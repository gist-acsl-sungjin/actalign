# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
from __future__ import annotations
import collections
import dataclasses
import logging
# import math
import pathlib
import os 
# import json
# import time
# import torch
# import PIL.Image
# import torchvision.transforms as transforms
import imageio
from libero.libero import benchmark
# from libero.libero import get_libero_path
# from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
# import scipy.interpolate as si
# import scipy.spatial.transform as st
import cv2
from scipy.spatial.transform import Rotation as R
# from datasets import load_dataset
# import datasets
# import einops
import threading
from queue import Queue, Empty
from utils import  _encode_image_b64,  load_api_key, get_prompt, _get_libero_vec_env, TASK_KEYWORD_MAPPLING, make_vec_env, raw_obs_to_numpy_obs, _get_libero_env
from utils import _quat2axisangle_batch, _quat2axisangle
import time 
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import multiprocessing as mp

"""
LIBERO Evaluation Script with Thinking Reset Functionality

This script includes a new reset mechanism to handle thinking state persistence between tasks:

1. client.reset_thinking() - Resets all batch thinking states (like warmup)
2. client.reset_thinking(batch_idx=i) - Resets thinking state for specific batch index

Usage:
- Use client.reset_thinking() when you want to clean up all thinking states (e.g., between tasks)
- Use client.reset_thinking(batch_idx=i) when you want to reset all environment's thinking states to a specific environment's thinking state specified by the batch_idx
- The reset is automatically called when all environments reach the second thinking stage

This replaces the previous hacky 200-step loop that was used to flush the model server's thinking state.
"""
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
LIBERO_TASK_INDEX_MAPPING = {
    0: 6,
    1: 2,
    2: 1,
    3: 4,
    4: 0,
    5: 8,
    6: 7,
    7: 5,
    8: 3,
    9: 9
}
TASK_MAPPING= {
    0:["place the alphabet soup in the basket and the tomato sauce as well","put both the can of soup and can of sauce in the basket"],
    1:["pick up both the cream cheese box and the butter and place them in the basket", "put both the box of cheese and the box of butter in the basket"],
    2:["switch the stove on and set the moka pot on the stove","turn on the cooktop and place the moka machine on it" ],
    3:["move the black bowl to the bottom drawer and close the drawer of the cabinet","put the middle bowl to the lowest drawer and close it"],
    4:["place two mugs on the plates, left one is the white mug and the right one is the yellow and white mug","put the pure white cup on the left plate and put the other one with the yellow handle on the right plate"],
    5:["grab the standing book and transfer it to the back compartment of the caddy","pick up the right book and put it in the rear part of the caddy"],
    6:["set the white mug on the plate with chocolate pudding to be placed to the right","put the pure white cup on the middle plate and put the brown chocolate to the right of the plate"],
    7:["put both the can of soup and the box of cheese in the basket","move the two objects, alphabet soup and cream cheese box, to the basket"],
    8:["transfer both moka pots from the table to the stove","put both the moka coffee makers on the cooktop"],
    9:["place the yellow and white mug inside the microwave, then shut the door","put the middle mug inside the microwave and close the door"],
}
mp.set_start_method("spawn", force=True)

# Create a session with connection pooling and retry logic for faster API calls
def create_optimized_session():
    """Create an optimized requests session with connection pooling and retry logic."""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=0.1,  # wait 0.1, 0.2, 0.4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # retry on these status codes
    )
    
    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=10,  # number of connection pools to cache
        pool_maxsize=20,  # maximum number of connections in each pool
    )
    
    # Mount adapter for both HTTP and HTTPS
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Set default timeout
    session.timeout = (5, 30)  # (connect_timeout, read_timeout)
    
    return session

# Global session for reuse
_api_session = None

def get_api_session():
    """Get or create the optimized API session."""
    global _api_session
    if _api_session is None:
        _api_session = create_optimized_session()
    return _api_session

def optimize_image_for_api(img, max_size=512, quality=85):
    """
    Optimize image for API transmission by resizing and compressing.
    
    Args:
        img: Input image array
        max_size: Maximum dimension size
        quality: JPEG quality (1-100)
    
    Returns:
        Optimized base64 encoded image string
    """
    import base64
    from io import BytesIO
    from PIL import Image
    
    # Convert numpy array to PIL Image
    if len(img.shape) == 3:
        pil_img = Image.fromarray(img)
    else:
        pil_img = Image.fromarray(img, mode='RGB')
    
    # Resize if too large
    if max(pil_img.size) > max_size:
        ratio = max_size / max(pil_img.size)
        new_size = tuple(int(dim * ratio) for dim in pil_img.size)
        pil_img = pil_img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to JPEG with specified quality
    buffer = BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality, optimize=True)
    
    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    action_horizon: int = 16
    action_dim: int = 7
    # TODO: load the data config from the training  to here
    history_length: int = 2
    # use the wrist image
    use_wrist_image: bool = False
    # Image downsampling steps for history (should match training config)
    image_down_sample_steps: list[int] = dataclasses.field(default_factory=lambda: [])#list[int] = dataclasses.field(default_factory=lambda: [3, 15])
    # State downsampling steps for history (should match training config)
    state_down_sample_steps: list[int] = dataclasses.field(default_factory=lambda:  []) #[3, 15])
    # Control frequency in Hz
    frequency: float = 10.0
    ## action down sample steps
    action_down_sample_steps: int = 3
    ## command latency
    command_latency: float = 0.01
    ## steps per inference
    steps_per_inference: int = 1#6
    ## temporal ensemble
    temporal_agg: bool = True
    ## batch size
    batch_size: int = 1
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_long"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50#50  # Number of rollouts per task
    ## visual ood
    visual_ood: int = 0 ## 0: no visual ood, 1: visual ood with visual-background(background and viewpoint), 2: visual ood with viusal scene(distractor object)
    ## semantic ood
    semantic_ood: int = 0 ## 0: no semantic ood, 1: semantic ood with rephrase, 2: semantic ood with object-property
    ## behavior ood
    behavior_ood: int = 0 ## 0: no behavior ood, 1: behavior composition (13 novel tasks)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos_reason_sample"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)
    
    #################################################################################################################
    # API Optimization parameters
    #################################################################################################################
    api_timeout: tuple[int, int] = (5, 25)  # (connect_timeout, read_timeout) in seconds
    api_verification_cooldown: float = 2.0  # Minimum seconds between verification calls
    api_image_quality: int = 85  # JPEG quality for image optimization (1-100)
    api_max_image_size: int = 512  # Maximum image dimension for API calls


def get_pi_obs_dict_batch(env_obs, previous_obs, instruction, args):
    """
    Get the observation dictionary for the pi0 model with reasoning 
    Returns:
        return_dict: Dictionary containing observations for the model
        previous_obs: Updated previous observations for next iteration
    """
    
    return_dict = {'prompt': instruction}
    
    for i in range(1):
        ## stack the list of images into the shape of (num_envs, 224, 224,3)
        return_dict[f'image_{i+1}'] = np.stack(env_obs['agentview_image'], axis=0)
        if args.use_wrist_image:
            ## stack the list of wrist images into the shape of (num_envs, 3, 224, 224)
            return_dict[f'image_wrist_{i+1}'] = np.stack(env_obs['robot0_eye_in_hand_image'], axis=0)
    
 
    # Add current state to history
    ## stack the list of states into the shape of (num_envs, 7)
    ## stack each element first 
    obs_pos = np.stack(env_obs['robot0_eef_pos'], axis=0)
    obs_quat = np.stack(env_obs['robot0_eef_quat'], axis=0)
    # breakpoint()
    obs_gripper = np.stack(env_obs['robot0_gripper_qpos'], axis=0)
    current_state = np.concatenate([

        obs_pos, 
        _quat2axisangle_batch(obs_quat),
        obs_gripper
    ], axis=1)
   
    
    
    return_dict['state'] = current_state
    
    return return_dict, previous_obs
def get_pi_obs_dict(env_obs, previous_obs, instruction, args):
    """
    Get the observation dictionary for the pi0 model with reasoning 
    Returns:
        return_dict: Dictionary containing observations for the model
        previous_obs: Updated previous observations for next iteration
    """
    
    return_dict = {'prompt': instruction}
    
   
    for i in range(1):
        return_dict[f'image_{i+1}'] = env_obs['agentview_image']
        if args.use_wrist_image:
   
            return_dict[f'image_wrist_{i+1}'] = env_obs['robot0_eye_in_hand_image']
    
   
    # Add current state to history
    current_state = np.concatenate([
        env_obs['robot0_eef_pos'],
        _quat2axisangle(env_obs['robot0_eef_quat']),
        env_obs['robot0_gripper_qpos']
    ])
   
    state_history = np.array([current_state])
    
    
    return_dict['state'] = state_history.flatten()
    
    return return_dict, previous_obs



def annotate_image_with_thoughts_batch(imgs, policy_return_dict, prev_img=None, response_thought=None):
    """
    Annotate the image with policy thoughts and status
    """
    annotated_imgs = []
    batch_size = len(imgs)
    is_thinking_batch = policy_return_dict.get('isthinking', [False] * batch_size)
    thoughts_batch = policy_return_dict.get('thought', ['' for _ in range(batch_size)])
 
    
    for i in range(batch_size):
        single_policy_return_dict = {}
        single_policy_return_dict['isthinking'] = is_thinking_batch[i]
        single_policy_return_dict['thought'] = thoughts_batch[i] if thoughts_batch != '' else ''
        if prev_img is None or all(img is None for img in prev_img):
            annotated_img = annotate_image_with_thoughts(imgs[i], single_policy_return_dict, response_thought=response_thought)
        else:
            annotated_img = annotate_image_with_thoughts(imgs[i], single_policy_return_dict, prev_img[i], response_thought=response_thought)
        annotated_imgs.append(annotated_img)
    return annotated_imgs

def annotate_image_with_thoughts(img, policy_return_dict, prev_img=None, response_thought=None):
    """
    Annotate the image with policy thoughts and status
    """
    # Create a copy of the image to avoid modifying the original
    annotated_img = img.copy()


    ## put another empty white image on the right that can put text on that 
    # Add thinking status
    is_thinking_flag = policy_return_dict.get('isthinking', False)
    status_text = "Thinking..." if is_thinking_flag else "Executing"
    
    # Add thoughts if available
    thoughts = policy_return_dict.get('thought', '')
    
    # Convert image to RGB for OpenCV
    if len(annotated_img.shape) == 2:
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_GRAY2RGB)
    elif annotated_img.shape[2] == 4:
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGBA2RGB)
    scaled_img = cv2.resize(annotated_img, None, fx=2.5, fy=2.5)
    
    # Add status text
    cv2.putText(
        scaled_img,
        f"Status: {status_text}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2
    )
    # Create white image same size as original
    if prev_img is None: 
        h, w = scaled_img.shape[:2]
        white_img = np.ones((h, w, 3), dtype=np.uint8) * 255
        
        # Combine original and white images side by side
        combined_img = np.hstack((scaled_img, white_img))
        scaled_img = combined_img
    else:
        ## use the right side of the previous image and put the current image on the left
        h, w = scaled_img.shape[:2]
        h_prev, w_prev = prev_img.shape[:2]
        scaled_img = np.hstack((scaled_img,prev_img[:, w_prev//2:]))
        
    # Add thoughts if available
    if response_thought:
        thoughts = response_thought[0]
    if thoughts:
       
        ## make the right side white
        h, w = scaled_img.shape[:2]
        ## annotated img is already combined with white image on the right
        # make the right side white
        scaled_img[:, w//2:] = 255
        # Split thoughts into lines for better readability
        # Split long lines into multiple lines if needed
        thought_lines = []
        for line in thoughts.split('\n'):
            # Split lines longer than 50 chars
            while len(line) > 63:
                split_idx = line[:63].rfind(' ')
                if split_idx == -1:
                    split_idx = 63
                thought_lines.append(line[:split_idx])
                line = line[split_idx:].lstrip()
            thought_lines.append(line)
            
        for i, line in enumerate(thought_lines):
            y_pos = 30 + i * 30
            cv2.putText(
                scaled_img,
                line,
                (w//2 + 10, y_pos), # Offset x by width to put text on white side
                cv2.FONT_HERSHEY_SIMPLEX,
                # cv2.FONT_HERSHEY_COMPLEX_SMALL,

                0.5,
                (0, 0, 0), # Black text
                1
            )
    return scaled_img




# Background verification state for test_reset_flags
_background_verification_thread: threading.Thread | None = None
_background_verification_result: Queue[int] = Queue(maxsize=1)
_background_verification_running = False
_background_verification_indices = set()  # Track which indices we're already verifying
_background_verification_thinking = False  # Track if any samples are still thinking
_background_verification_last_call = 0  # Track last verification call time
_background_verification_cooldown = 2.0  # Minimum seconds between verification calls
_background_verification_cancel_event = threading.Event()  # Event to signal cancellation


def test_reset_flags(reset_indices: np.ndarray, initial_image:np.ndarray, agent_images: list[list[np.ndarray]], wrist_images: list[list[np.ndarray]], instruction_text: list[str], args: Args) -> None:
    """
    Start background verification for given indices by calling GPT with their last replay images.
    The verification runs in a background thread and results can be checked with check_background_verification().
    
    Args:
        reset_indices: Array of indices to test
        replay_images: List of replay image sequences for each environment
        task_description: The task description for context
    """
    global _background_verification_thread, _background_verification_running, _background_verification_indices, _background_verification_thinking
    
    api_key =  load_api_key('openai')
    # Convert to set for comparison
    current_indices = set(reset_indices.astype(int))
    
    # Only start verification if we're not already verifying these exact indices
    if _background_verification_running and current_indices == _background_verification_indices:
        return  # Already verifying these indices
    
    # If we're verifying different indices, wait for current verification to finish
    if _background_verification_running:
        return  # Skip this call, let current verification finish
    
    # Clear any stale result
    try:
        while True:
            _background_verification_result.get_nowait()
    except Empty:
        pass
    
    _background_verification_running = True
    _background_verification_thinking = True
    _background_verification_indices = current_indices
    
    def _background_worker():
        global _background_verification_running, _background_verification_indices, _background_verification_thinking
        try:
            # Get optimized session
            session = get_api_session()
            
            # Test each index in parallel using threads
            results = {}
            threads = []
            
            def test_single_index(idx: int):
                try:
                    # Check for cancellation before making API call
                    if _background_verification_cancel_event.is_set():
                        return
                        
                    # Get the last image for this index
                    if idx < len(agent_images) and len(agent_images[idx]) > 0:
                        last_image = agent_images[idx][-1]
                        last_wrist_image = wrist_images[idx][-1]
                        
                        # Encode image to base64
                        b64_initial_image = optimize_image_for_api(initial_image, args.api_max_image_size, args.api_image_quality)
                        b64_image = optimize_image_for_api(last_image, args.api_max_image_size, args.api_image_quality)
                        b64_wrist_image = optimize_image_for_api(last_wrist_image, args.api_max_image_size, args.api_image_quality)
                        # Create prompt for GPT
                        subtask_goal = instruction_text[idx]
                        for keyword, mapping in TASK_KEYWORD_MAPPLING.items():
                            if keyword in subtask_goal:
                                subtask_goal = subtask_goal.replace(keyword,  mapping)
                        

                        prompt = get_prompt(subtask_goal, "verifier")
                       
                        # Check for cancellation again before making the request
                        if _background_verification_cancel_event.is_set():
                            return
                        base_url = "https://api.openai.com/v1/responses"
                        # Use optimized session with connection pooling
                        response = session.post(
                            base_url,
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": "gpt-4o-2024-08-06",
                                "input": [
                                    {
                                        "role": "user",
                                        "content": [
                                            {"type": "input_text", "text": prompt},
                                            {
                                                "type": "input_image",
                                                "image_url": f"data:image/jpeg;base64,{b64_initial_image}"
                                            },
                                            {
                                                "type": "input_image",
                                                "image_url": f"data:image/jpeg;base64,{b64_image}"
                                            },
                                            {
                                                "type": "input_image",
                                                "image_url": f"data:image/jpeg;base64,{b64_wrist_image}"
                                            }
                                        ]
                                    }
                                ],
                                "temperature": 0.0,
                                "max_output_tokens": 1000,
                            },
                            timeout=args.api_timeout  # Use configurable timeout
                        )
                        
                        # Check for cancellation after getting response
                        if _background_verification_cancel_event.is_set():
                            return
                    
                       
                        result_text = response.json()["output"][0]["content"][0]["text"].strip().upper()  # type: ignore[index]
                          
                        ## check the success from the result_text, get the last line in the text, 
                        verification_result = result_text.split("\n")[-1].strip()
                        print(verification_result)

                        
                        # Check if response indicates success
                        if "VERIFICATION" in verification_result:
                            if "SUCCESS" in verification_result :
                                results[idx] = True
                            else:
                                results[idx] = False
                        else:
                            raise ValueError(f"Invalid verification result: {verification_result}")
                            
                    else:
                        results[idx] = False
                        
                except Exception as e:
                    logging.warning(f"GPT verification failed for index {idx}: {e}")
                    results[idx] = False
            
            # Start threads for all indices with reduced timeout
            for idx in current_indices:
                # Check for cancellation before starting each thread
                if _background_verification_cancel_event.is_set():
                    break
                thread = threading.Thread(target=test_single_index, args=(int(idx),), daemon=True)
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete with shorter timeout
            for thread in threads:
                # Check for cancellation during thread joining
                if _background_verification_cancel_event.is_set():
                    break
                thread.join(timeout=20.0)  # Reduced from 30s to 20s timeout per thread
            
            # Only process results if not cancelled
            if not _background_verification_cancel_event.is_set():
                # Find the first successful index
                for idx in current_indices:
                    if results.get(int(idx), False):
                        try:
                            _background_verification_result.put_nowait(int(idx))
                        except Exception:
                            pass
                        break
                    
        except Exception as e:
            logging.warning(f"Background verification failed: {e}")
        finally:
            _background_verification_running = False
            _background_verification_thinking = False
            _background_verification_indices = set()
    
    _background_verification_thread = threading.Thread(target=_background_worker, daemon=True)
    _background_verification_thread.start()


def check_background_verification() -> int | None:
    """
    Check if background verification has completed and return the result.
    
    Returns:
        Index of successful environment if found, None otherwise
    """
    try:
        return _background_verification_result.get_nowait()
    except Empty:
        return None


def is_background_verification_running() -> bool:
    """
    Check if background verification is currently running.
    
    Returns:
        True if verification is running, False otherwise
    """
    return _background_verification_running


def is_background_verification_thinking() -> bool:
    """
    Check if background verification is still in thinking process.
    
    Returns:
        True if verification is still thinking, False otherwise
    """
    return _background_verification_thinking





def reset_background():
    """
    Cancel any remaining background API requests and reset all background verification state.
    This function forcefully cancels any ongoing background verification threads.
    """
    global _background_verification_running, _background_verification_indices, _background_verification_thinking, _background_verification_cancel_event
    
    # Signal cancellation to any running background threads
    _background_verification_cancel_event.set()
    
    # Wait for background thread to finish (with timeout)
    if _background_verification_thread and _background_verification_thread.is_alive():
        _background_verification_thread.join(timeout=5.0)  # Wait up to 5 seconds
    
    # Reset all state
    _background_verification_running = False
    _background_verification_indices = set()
    _background_verification_thinking = False
    _background_verification_cancel_event.clear()  # Reset the event for future use
    
    # Clear any stale results
    try:
        while True:
            _background_verification_result.get_nowait()
    except Empty:
        pass
    
    logging.info("Background verification cancelled and reset")




def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    ## load the hf dataset
    # hf_dataset = load_hf_dataset("/home/yilinw/.cache/huggingface/lerobot/physical-intelligence/libero-10")
    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
   
    # Start evaluation
    total_episodes, total_successes, total_fake_successes, total_episode_length, total_real_episode_length = 0, 0, 0, 0, 0
    ## check if behavior ood is enabled
    if args.behavior_ood == 1:
        task_range_in_suite = [10,23] # 13 novel tasks
    elif args.behavior_ood == 0:
        task_range_in_suite = [0,10]
        print('no behavior ood')
    else:
        raise ValueError(f"Unknown behavior task range: {args.behavior_ood}")
    
    for task_id in tqdm.tqdm(range(task_range_in_suite[0], task_range_in_suite[1])):
        ## check if visual ood is enabled
        if args.visual_ood in [1,2]:
            task_id = task_id + 13 + 10 * args.visual_ood
        elif args.visual_ood == 0:
            print('no visual ood')
        else:
            raise ValueError(f"Unknown visual ood: {args.visual_ood}")
        
        # Get task
        task = task_suite.get_task(task_id)
        

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        
        
        
        vec_env, task_description = _get_libero_vec_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.batch_size)
        
       
        ## create a single env for actual exe
        env_exec, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        ## check if semantic ood is enabled
        if args.semantic_ood in [1,2]:
            task_description = TASK_MAPPING[task_id][args.semantic_ood-1]
        elif args.semantic_ood == 0:
            print('no semantic ood')
        else:
            raise ValueError(f"Unknown semantic ood: {args.semantic_ood}")

        task_episodes, task_successes, task_fake_successes, task_episode_length, task_real_episode_length = 0, 0, 0, 0, 0

        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):#[22, 38, 41, 47, 49]):#range(args.num_trials_per_task)):

            
            #task_description = "put both the tomato sauce and the cream cheese box in the basket"
            logging.info(f"\nTask: {task_description}")
          
            
            
            vec_env.reset(np.arange(args.batch_size))
            env_exec.reset()
        
            action_plan = collections.deque()

            vec_env.set_init_state([initial_states[episode_idx]]*args.batch_size, np.arange(args.batch_size))
            obs_exec = env_exec.set_init_state(initial_states[episode_idx])
            # Setup
            t = [0 for _ in range(args.batch_size)]
            replay_images = [[] for _ in range(args.batch_size)]
            wrist_images = [[] for _ in range(args.batch_size)]
            agent_images = [[] for _ in range(args.batch_size)]
            ## create previous observation buffer
            previous_obs = {"image": None, "wrist_image": None, "state": None}
            
            # Initialize episode timing
            dt = 1.0 / args.frequency  # Time step in seconds
            
            logging.info(f"Starting episode {task_episodes+1}...")
            thinking_pose = [None] * args.batch_size
            thinking_counter = np.zeros(args.batch_size)

            ## create a list of action_plan for each env
            action_plan = [collections.deque() for _ in range(args.batch_size)]
            is_thinking = [False for _ in range(args.batch_size)]
            this_target_poses = [None ] * args.batch_size
            last_annotated_img = [None]*args.batch_size


            ## create each key in envs as a extended shape
            used_keys = ['agentview_image', 'robot0_eye_in_hand_image', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
            ## dummy steps to stablize the env
            for t in range(args.num_steps_wait):
                obs, reward, done, info = env_exec.step(LIBERO_DUMMY_ACTION)
            
            
            for _ in range(args.num_steps_wait):
                vec_obs, vec_reward, vec_done, vec_info = vec_env.step([LIBERO_DUMMY_ACTION]*args.batch_size, np.arange(args.batch_size))
            
            all_obs = raw_obs_to_numpy_obs(vec_obs)
            print('warm up')
            ## warm up the base model
            for _ in tqdm.tqdm(range(args.steps_per_inference), desc='Warm up', leave=False):
        
                obs_dict_np, previous_obs = get_pi_obs_dict_batch(all_obs, previous_obs, task_description, args)
        
                obs_dict_np['is_warm_up'] = np.array([True])
        
                action =client.infer(obs_dict_np)
            initial_image = vec_obs[0]['agentview_image'][::-1, ::-1]
            time_tracking = np.ones(args.batch_size) * (args.num_steps_wait + 1)
            env_dones = np.zeros(args.batch_size)
            action = np.zeros((args.batch_size, args.action_horizon, args.action_dim))
            thinking_time = np.zeros(args.batch_size)
            num_thinking_phases = 4
            env_reset_sim_state = None 
            reset_index = None 
            env_reset_sim_states = vec_env.get_sim_state(np.arange(args.batch_size))
     
            think_response = {'thought': None}
            used_steps = np.ones(args.batch_size) * (args.num_steps_wait + 1)
            used_step = time_tracking[0]
            limit = 200
            saved_exec_obs = []
            saved_real_exec_obs = []
            ## empty the action plan
            selected_action_plan = [] 
            batch_action_plan = [[] for _ in range(args.batch_size)]
            exec_done = False
            early_exit = False

            for phase in range(num_thinking_phases):
                # Cancel any remaining background verification from previous phase
                reset_background()

                ## check if there is any early exit
                if early_exit:
                    break
                ## compare the last phase videos 
                ## get the reset index
                ## reset all the envs 
                if env_reset_sim_state is not None:
                    ## move forward the action plan int the env_exec 
                    selected_imgs = replay_images[reset_index]
                    saved_exec_obs.extend(selected_imgs)
                    ## not save more than 520 steps of the exec_obs
                    if len(saved_exec_obs) > max_steps:
                        break
                    selected_action_plan = batch_action_plan[reset_index]

                    for selected_action in selected_action_plan:
                        exec_obs, exec_reward, exec_done, exec_info = env_exec.step(selected_action)
                    #     ## save the left half of the thinking
                        left_half_image = replay_images[reset_index][-1][:,int(args.resize_size* 2.5):,:]
                    #     ## concatenate the left half of the image with the exec_obs['agentview_image']
                        resized_size = int(args.resize_size* 2.5)
                        saved_img = np.concatenate([cv2.resize(exec_obs['agentview_image'][::-1, ::-1], (resized_size, resized_size)), left_half_image], axis=1)
                        saved_real_exec_obs.append(saved_img)
                        if exec_done: 
                            break
                    env_reset_sim_state = vec_env.get_sim_state(reset_index)[0]
                    vec_env.set_init_state(np.repeat(env_reset_sim_state.reshape(1, -1), args.batch_size, axis=0), np.arange(args.batch_size))
                    gripper_width = np.sum(np.abs(vec_obs[reset_index]['robot0_gripper_qpos']))
                    if gripper_width < 0.06:
                        dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
                    else:
                        dummy_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
                
                    all_obs = raw_obs_to_numpy_obs(vec_obs)
                 
                    ## reset all the variables 
                    action_plan = [collections.deque() for _ in range(args.batch_size)]
                    is_thinking = [False for _ in range(args.batch_size)]
                    thinking_pose = [None for _ in range(args.batch_size)]
                    thinking_counter = np.zeros(args.batch_size) ## starting from the second phase, we only count thinking once to end the thinking phase
                    thinking_time = np.zeros(args.batch_size)+ 1
                    reset_index = None
                   
                    
                    time_tracking = np.ones(args.batch_size) * used_step
                  
                    action = np.zeros((args.batch_size, args.action_horizon, args.action_dim))
                    this_target_poses = [None] * args.batch_size 
                    replay_images = [[] for _ in range(args.batch_size)]
                    wrist_images = [[] for _ in range(args.batch_size)]
                    agent_images = [[] for _ in range(args.batch_size)]
                    ## empty the action plan
                    selected_action_plan = [] 
                    batch_action_plan = [[] for _ in range(args.batch_size)]
                subtask_goals = ["" for _ in range(args.batch_size)]
                print('--------Begining the new thinking phase-----')
                print(f'start the {phase}th thinking phase')
                print('used step', used_step)
                print('-------------------------')
                # Reset background verification state for new phase
                ## reset the inference model to specific index 
                ## wait for the key to specify the model 
                while min(time_tracking) <= max_steps + args.num_steps_wait:
                

                    if not all(is_thinking):
                        ## do not update the observation history when thinking (assume the agent keeps still when thinking)
                    
                        obs_dict_np, previous_obs = get_pi_obs_dict_batch(all_obs, previous_obs, task_description, args)
                        
                 # Query model to get action or thinking if the current action queue is empty
                    if not action_plan[0]:
                        ## current queue is empty, so we need to infer the action
                   
                        policy_return_dict = client.infer(obs_dict_np)
                    for i in range(args.batch_size):
                       
                        if policy_return_dict.get('thought', False)  and subtask_goals[i] == "" and policy_return_dict['thought'][i] != "":
                            ## extract the string after 'Plan: ' and before "What I have done"
                          
                            plan_string= policy_return_dict['thought'][i].split('Now I need to do:')[-1] 
                            plan_string = plan_string.split('.\n')[0]
                            subtask_goals[i] = plan_string
                        if think_response['thought'] is not None and subtask_goals[i] == "" and think_response['thought'][i] != "":
                            plan_string = think_response['thought'][i].split('Now I need to do:')[-1]
                            plan_string = plan_string.split('.\n')[0]
                            subtask_goals[i] = plan_string
                   
                    annotated_imgs = annotate_image_with_thoughts_batch(all_obs['agentview_image'], policy_return_dict, last_annotated_img, think_response['thought'])
                
                    # Save annotated image for replay video
                    for i in range(args.batch_size):
                        
                        last_annotated_img[i] = annotated_imgs[i].copy()
                    ## if thinking, do not append the same annotated image to the replay_images buffer
                    if not all(is_thinking):
                        for i in range(args.batch_size):
                            if thinking_time[i] < 2 and len(replay_images[i]) < max_steps: 
                                replay_images[i].append(annotated_imgs[i]) 
                                wrist_images[i].append(all_obs['robot0_eye_in_hand_image'][i])
                                agent_images[i].append(all_obs['agentview_image'][i])
                     
                    
                    # breakpoint()
                    policy_is_thinking = policy_return_dict.get('isthinking', [False]*args.batch_size) 
                    ## synchronize the thinking, so if some envs are finished thinking, we are still querying the model until all envs are finished thinking
                    ## this is a bit hacky because in theory this would waste some compute on the inference side because you need to query the same observation multiple times
                    ## need to figure out a better way to do this instead of this hacky solution
   
                    # Update thinking states and actions for all envs at once
                    need_inference = np.array([not bool(plan) for plan in action_plan])
                    if np.any(need_inference):
                        # For envs that need inference, update thinking state and actions
                        is_thinking = np.where(need_inference, policy_is_thinking, is_thinking)
                        
                        # Handle thinking states
                        thinking_mask = need_inference & policy_is_thinking
                        if np.any(thinking_mask):
                            # Initialize thinking for new thinkers
                            new_thinkers = thinking_mask & (np.array(thinking_pose) == None)
                            if np.any(new_thinkers):
                                thinking_counter[new_thinkers] = 0
                                thinking_time[new_thinkers] += 1
                                ## something tricky here, needs to not update when there is redundant thinking
                                ## first case when it is not updating the thinking, it should not update 
                                ## when the time_tracking is the same as used_step, it should not update 
                                ## when the thinking is update but it has already more than 2, it should not update
                                # Get indices where thinking_time > 1 and it is the new update 
                                reset_indices = np.where((thinking_time > 1) * new_thinkers)[0]
                                ## remove those reset_indices where the time_tracking is the same as the used_step, this is exactly as the begining of the episode
                                ## there is one exception that is at the right begining, 
                                if np.all(time_tracking[reset_indices] == args.num_steps_wait+1):
                                    reset_indices_reduced = reset_indices
                                else: 
                                    reset_indices_reduced = reset_indices[np.where(time_tracking[reset_indices] != used_step)[0]]
                                ## remove those indices whose thinking_time is already more than 2 
                                reset_indices_reduced = reset_indices_reduced[np.where(thinking_time[reset_indices_reduced] <= 2)[0]]
                                
                                used_steps[reset_indices_reduced] = time_tracking[reset_indices_reduced].copy()
                                # Start background verification if not already running
                                if reset_indices_reduced.size > 0 and not is_background_verification_running():
                                    # Rate limiting: don't call too frequently
                                    global _background_verification_last_call
                                    current_time = time.time()
                                    if current_time - _background_verification_last_call >= args.api_verification_cooldown:
                                        _background_verification_last_call = current_time
                                        
                                        test_reset_flags(reset_indices_reduced, initial_image, agent_images, wrist_images, subtask_goals, args)
                                
                                
                                
                                
                               
                           
                                ## if time_tracking[reset_indices] is the same as used_steps[reset_indices], then we reduce the thinking_time by 1
                                reset_indices_invalid = np.where(time_tracking[reset_indices] == used_step)[0]
                                thinking_time[reset_indices_invalid] -= 1
                

                                # Loop over those envs and update the corresponding states
                                sim_states = vec_env.get_sim_state(np.arange(args.batch_size))
                               
                                for idx in reset_indices_reduced:
                                   
                                    env_reset_sim_states[idx] = sim_states[idx]#sim_states[sim_index]
                     
                            
                            # Update counters for thinking envs
                            thinking_counter[thinking_mask] += 1
                            limit = 200
                            used_steps[thinking_counter > limit] = time_tracking[thinking_counter > limit]
                           
                            
                            # Set thinking poses and actions
                            thinking_pose = np.where(thinking_mask, True, thinking_pose)
                          
                        
                        # Handle non-thinking states
                        non_thinking_mask = need_inference & ~np.array(policy_is_thinking)
                        if np.any(non_thinking_mask):
                            is_thinking[non_thinking_mask] = False
                            thinking_pose = np.where(non_thinking_mask, None, thinking_pose)
                       
                            # Expand non_thinking_mask to match action dimensions
                            expanded_mask = non_thinking_mask[:, np.newaxis, np.newaxis]  # Shape: (batch_size, 1, 1)
                            expanded_mask = np.broadcast_to(expanded_mask, policy_return_dict['actions'].shape)
                            
                            # Update actions only for non-thinking envs while preserving structure
                            action = np.where(expanded_mask, policy_return_dict['actions'], action)
                        
                 
                    condition = (thinking_time > 1) | (thinking_counter > limit) | (time_tracking >= max_steps + args.num_steps_wait)
                    # Check for background verification results
                    reset_flags = check_background_verification()
                    if reset_flags is not None and ((thinking_time[reset_flags] >1 ) or (thinking_counter[reset_flags] > limit) or (time_tracking[reset_flags] >= max_steps + args.num_steps_wait)):
                    
                        print(f"Background GPT verification succeeded with index {reset_flags}; resetting to next phase.")
                        ## if multiple reset_flags, randomly choose one 
                        reset_index = reset_flags #np.random.choice(reset_flags)
                        suffix = f"variation_{episode_idx}_thinking_phase_{phase}"
                        task_segment = task_description.replace(" ", "_")
                        # print(args.video_out_path)
                        imageio.mimwrite(
                            pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_reset_index_{reset_index}.mp4",
                            [np.asarray(x) for x in replay_images[reset_index]],
                            fps=10,
                        )
                        print('video saved')
                        env_reset_sim_state = env_reset_sim_states[reset_index]
                        used_step = used_steps[reset_index]
                        selected_action_plan = batch_action_plan[reset_index].copy()
                        think_response = client.reset_thinking(reset_index)
                        break
                    ## check if the background verification is thinking 
                    ## if all env ends action sampling, but background verification is thinking, then continue the loop until the background verification is finished
                    if is_background_verification_thinking() and np.all(condition):
                        continue
                    if np.all(condition):
                        print('all env ends action sampling')
                        for i in range(args.batch_size):
                            if time_tracking[i] >= max_steps + args.num_steps_wait:
                                used_steps[i] = time_tracking[i]
                        print('updated used_step', used_step)
                    
                        # Save a replay video of the sampled episode with thoughts
                        for i in range(args.batch_size):
                            # suffix = f"success_{task_episodes}" if done else f"v_{task_episodes}"
                            print('length of replay_images', len(replay_images[i]))
                            suffix = f"variation_{episode_idx}_thinking_phase_{phase}"
                            task_segment = task_description.replace(" ", "_")
                            # print(args.video_out_path)
                            imageio.mimwrite(
                                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_batch_{i}.mp4",
                                [np.asarray(x) for x in replay_images[i]],
                                fps=10,
                            )
                            print('video saved')
                        print('no successful episode, select the shortest one')
                        ## add some conditions to make sure it is at least 25 steps
                        ## filter out the ones that are less than 25 steps
                       
                        length_list = [len(replay_images[i]) for i in range(args.batch_size)]
                        ## filter out the ones that are less than 25 steps
                        filtered_length_list = [length for length in length_list if length >= 25]
                        ## if there is no length_list, then select the first one
                        if not filtered_length_list:
                            reset_index = 0
                        else:
                            min_length = min(filtered_length_list)
                            reset_index = length_list.index(min_length)
                        
                        
                        env_reset_sim_state = env_reset_sim_states[reset_index]
                        used_step = used_steps[reset_index]
                        selected_action_plan = batch_action_plan[reset_index].copy()
                        
                        think_response = client.reset_thinking(reset_index)
                        print("Reset completed. Moving to next task.")
                        
                        
                        break 
                    action_new = action.copy()
                    
                    for i in range(args.batch_size):
                        if not action_plan[i]: 
                            if thinking_pose[i] is not None:
                                ## pass in None action if the model is just thinking
                                # Don't try to use action[i] when thinking, just set this_target_poses to None
                                
                                this_target_poses[i] = None
                                
                                
                            else:
                                ## no temporal ensemble, so it is just executing the first k steps of predicted actions before next inference
                          
                                chunk= action_new[i][:5] 
                              
                                action_plan[i].extend(chunk)  
                                this_target_poses[i] = chunk
                                
                        ## if the current action queue is not empty, execute the action
                    # 2) Build a single batched action list to step the whole vector-env once
                    step_actions = []
                    exec_mask = [False] * args.batch_size  # True if we actually pop and execute a real action

                    for i in range(args.batch_size):
                        can_execute = (
                            this_target_poses[i] is not None
                            and action_plan[i]
                            and (time_tracking[i] < (max_steps + args.num_steps_wait + 1))
                            and (not env_dones[i])
                            and (thinking_time[i] < 2)
                        )

                        if can_execute:
                            actual_action = action_plan[i].popleft()
                            step_actions.append(actual_action.tolist() if hasattr(actual_action, "tolist") else actual_action)
                            batch_action_plan[i].append(actual_action)
                            exec_mask[i] = True
                    
                    if len(step_actions) != 0:

                    # 3) Single vectorized step
                        try:
                            sim_states = vec_env.get_sim_state(np.arange(args.batch_size))

                            
                        
                            vec_new_obs, vec_reward, vec_new_done, vec_info = vec_env.step(step_actions, np.where(exec_mask)[0])
                            # copy those obs to the index that is exec_mask is true
                            for idx_exec, idx_new in enumerate(np.where(exec_mask)[0]):
                                vec_obs[idx_new] = vec_new_obs[idx_exec].copy()
                                vec_done[idx_new] = vec_new_done[idx_exec]
                        except Exception as e:
                            ## recreate the envs from the sim_state 
                            envs = [] 
                        
                            vec_env= _get_libero_vec_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.batch_size)
                            
                            vec_env.set_init_state(sim_states, np.arange(args.batch_size))
                            for _ in range(10):
                                vec_obs, vec_reward, vec_done, vec_info = vec_env.step([LIBERO_DUMMY_ACTION]*args.batch_size, np.arange(args.batch_size))
                            vec_obs, vec_reward, vec_done, vec_info = vec_env.step(step_actions, np.arange(args.batch_size))
                    # 4) Update dones, optional logging, etc.
                    for i in range(args.batch_size):
                        if exec_mask[i]:
                            time_tracking[i] += 1
                        env_dones[i] = bool(vec_done[i])
                        if env_dones[i]:
                            print(f"successful episode, done {i}", env_dones[i])
                            early_exit = True
                            # Cancel any background verification since we're done
                            reset_background()
                            break 
                    all_obs = raw_obs_to_numpy_obs(vec_obs)
                   
                  
                            
                    if np.any(env_dones):
                        print('env_dones', env_dones)
                        done_index = np.where(env_dones)[0][0]
                    
                        suffix = f"variation_{episode_idx}_thinking_phase_{phase}"
                        task_segment = task_description.replace(" ", "_")
              
                        imageio.mimwrite(
                            pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_reset_index_{done_index}.mp4",
                            [np.asarray(x) for x in replay_images[done_index]],
                            fps=10,
                        )
                        print('video saved')
                        
                        break
                        
                
           ## end episode
            if np.any(env_dones):
                # break        # break
                done_index = np.where(env_dones)[0][0]
                saved_exec_obs.extend(replay_images[done_index])
                task_fake_successes += 1
                total_fake_successes += 1
               
                exec_env_t = len(saved_real_exec_obs) + args.num_steps_wait
                # ## repeat the last action plan for 50 steps 
                done_index = np.where(env_dones)[0][0]
                selected_action_plan = batch_action_plan[done_index].copy()
                if not selected_action_plan:
                    break
                if len(selected_action_plan) > 0:
                    last_action = selected_action_plan[-1]
                    selected_action_plan.extend([last_action]*10)
                for selected_action in selected_action_plan:
            
                    
                    exec_obs, exec_reward, exec_done, exec_info = env_exec.step(selected_action)
                    left_half_image =replay_images[done_index][-1][:,int(args.resize_size* 2.5):,:]
                    resized_size = int(args.resize_size* 2.5)
                    saved_img = np.concatenate([cv2.resize(exec_obs['agentview_image'][::-1, ::-1], (resized_size, resized_size)), left_half_image], axis=1)
                    saved_real_exec_obs.append(saved_img)
                    exec_env_t += 1
                    if exec_done: 
                        total_successes += 1
                        task_successes += 1
                        print('suceesful episode length', exec_env_t)
                        break
                    if exec_env_t >= max_steps + args.num_steps_wait:
                        break
            else:
                ## log all the wrong sample videos
                for i in range(args.batch_size):
                   
                    print('length of replay_images', len(replay_images[i]))
                    suffix = f"variation_{episode_idx}_thinking_phase_{phase}"
                    task_segment = task_description.replace(" ", "_")
                  
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_batch_{i}.mp4",
                        [np.asarray(x) for x in replay_images[i]],
                        fps=10,
                    )
                ## save the selected images 
                saved_exec_obs.extend(replay_images[reset_index])
                selected_action_plan = batch_action_plan[reset_index].copy()
                if len(selected_action_plan) > 0:
                    last_action = selected_action_plan[-1]
                    selected_action_plan.extend([last_action]*10)
                exec_env_t = len(saved_real_exec_obs) + args.num_steps_wait
                for selected_action in selected_action_plan:
     
                    
                    exec_obs, exec_reward, exec_done, exec_info = env_exec.step(selected_action)
                    left_half_image =replay_images[reset_index][-1][:,int(args.resize_size* 2.5):,:]
                    resized_size = int(args.resize_size* 2.5)
                    saved_img = np.concatenate([cv2.resize(exec_obs['agentview_image'][::-1, ::-1], (resized_size, resized_size)), left_half_image], axis=1)
                    saved_real_exec_obs.append(saved_img)
                    exec_env_t += 1
                    if exec_done: 
                        total_successes += 1
                        task_successes += 1
                        print('suceesful episode length', exec_env_t)
                        break
                    if exec_env_t >= max_steps + args.num_steps_wait:
                        break
                
       
                    
        
            
            suffix = f"reset_exec_{task_episodes}_success" if np.any(env_dones) else f"reset_exec_{task_episodes}_fail"
            suffix_real = f"real_exec_{task_episodes}_success" if exec_done else f"real_exec_{task_episodes}_fail"
            task_segment = task_description.replace(" ", "_")
            
            episode_length = len(saved_exec_obs)
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_length_{episode_length}.mp4",
                [np.asarray(x) for x in saved_exec_obs],
                fps=10,
            )
            real_episode_length = len(saved_real_exec_obs)
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix_real}_length_{real_episode_length}.mp4",
                [np.asarray(x) for x in saved_real_exec_obs],
                fps=10,
            )
            # Log current results
            total_episodes += 1
            task_episodes += 1
            if np.any(env_dones):
                logging.info(f"Reset Task Episode length: {episode_length}")
            else:
                episode_length = 510
                logging.info("Reset Task Episode length: 510")
            if exec_done:
                logging.info(f"Reset Task Episode length: {real_episode_length}")
            else:
                real_episode_length = 510
                logging.info("Reset Task Episode length: 510")
            task_episode_length += episode_length
            task_real_episode_length += real_episode_length
            total_episode_length += episode_length
            total_real_episode_length += real_episode_length
            
            logging.info(f" Reset Success: {np.any(env_dones)}")
            logging.info(f" Success: {exec_done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"Reset Task Episode length: {episode_length} ({total_episode_length/total_episodes:.1f})")
            logging.info(f"Real Task Episode length: {real_episode_length} ({total_real_episode_length/total_episodes:.1f})")
            logging.info(f"# reset successes : {total_fake_successes} ({total_fake_successes / total_episodes * 100:.1f}%)")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            # Clean up any remaining background verification
            reset_background()
            time.sleep(2)
        
        # Log final results
        logging.info(f"current task id: {task_id}")
        logging.info(f"current task reset success rate: {float(task_fake_successes) / float(task_episodes)}")
        logging.info(f"current total reset success rate: {float(total_fake_successes) / float(total_episodes)}")
        logging.info(f"current task reset episode length: {float(task_episode_length) / float(task_episodes)}")
        logging.info(f"current total reset episode length: {float(total_episode_length) / float(total_episodes)}")
        logging.info(f"current task real episode length: {float(task_real_episode_length) / float(task_episodes)}")
        logging.info(f"current total real episode length: {float(total_real_episode_length) / float(total_episodes)}")
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")
        
        env_exec.close()
        vec_env.close()

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total reset success rate: {float(total_fake_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
   





if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                        filename='eval_libero_100_reason_sample_10_libero_10_qualitative.log',
                        filemode='a')
    tyro.cli(eval_libero)
