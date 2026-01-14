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
import math
import pathlib
import os 
import json
import time
import torch
import PIL.Image
import torchvision.transforms as transforms

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import scipy.interpolate as si
import scipy.spatial.transform as st
import cv2
from scipy.spatial.transform import Rotation as R

from utils import _quat2axisangle,_get_libero_env
import einops
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
def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image
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
 
    history_length: int = 2
    # use the wrist image
    use_wrist_image: bool = False
    # Image downsampling steps for history (should match training config)
    image_down_sample_steps: list[int] = dataclasses.field(default_factory=lambda: [])
    # State downsampling steps for history (should match training config)
    state_down_sample_steps: list[int] = dataclasses.field(default_factory=lambda:  [])
    # Control frequency in Hz
    frequency: float = 10.0
    ## action down sample steps
    action_down_sample_steps: int = 3
    ## command latency
    command_latency: float = 0.01
    ## steps per inference
    steps_per_inference: int = 1
    ## temporal ensemble
    temporal_agg: bool = False
   
    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_long"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50#50#50  # Number of rollouts per task
    semantic_ood: int = 0 ## 0: no semantic ood, 1: semantic ood with rephrase, 2: semantic ood with object-property
    visual_ood: int = 0 ## 0: no visual ood, 1: visual ood with visual-background(background and viewpoint), 2: visual ood with viusal scene(distractor object)
    behavior_ood: int = 0 ## 0: no behavior ood, 1: behavior composition (13 novel tasks)
    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos_reason"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)

def get_pi_obs_dict(env_obs, previous_obs, instruction, args):
    """
    Get the observation dictionary for the pi0 model with reasoning 
    Returns:
        return_dict: Dictionary containing observations for the model
        previous_obs: Updated previous observations for next iteration
    """
    
    return_dict = {'prompt': instruction}
    
    # Handle image history with downsampling
    if previous_obs['image'] is None:
        # Initialize image history buffer
        previous_obs['image'] = []
        if args.use_wrist_image:
            previous_obs['wrist_image'] = []
        previous_obs['image_timestamps'] = []
    
    # Add current image to history
    previous_obs['image'].append(env_obs['agentview_image'])
    if args.use_wrist_image:
        previous_obs['wrist_image'].append(env_obs['robot0_eye_in_hand_image'])
    previous_obs['image_timestamps'].append(env_obs['timestamp'])
    
    # Calculate target indices for image history
    # Similar to training code: [current_idx] + [current_idx - down_sample_steps[i] for i in range(history_length-1)]
    current_idx = len(previous_obs['image']) - 1
    image_target_idx = np.array([current_idx] + 
                              [current_idx - step for step in args.image_down_sample_steps])
    
    # Ensure we have enough history
    # max_history_length = max(args.image_down_sample_steps) + 1 if args.image_down_sample_steps else 1
    if args.image_down_sample_steps:
        if len(previous_obs['image']) < max(args.image_down_sample_steps) + 1:
            # If we don't have enough history, duplicate the earliest available frame
            earliest_idx = max(0, min(image_target_idx))
            for i in range(len(image_target_idx)):
                if image_target_idx[i] < 0:
                    image_target_idx[i] = earliest_idx
    
    # Add images to return dict using target indices
    for i in range(len(image_target_idx)):
        return_dict[f'image_{i+1}'] = previous_obs['image'][image_target_idx[i]]
        if args.use_wrist_image:
            return_dict[f'image_wrist_{i+1}'] = previous_obs['wrist_image'][image_target_idx[i]]
    
    # Handle robot state history
    if previous_obs['state'] is None:
        # Initialize state history buffer
        previous_obs['state'] = []
        previous_obs['state_timestamps'] = []
    
    # Add current state to history
    current_state = np.concatenate([
        env_obs['robot0_eef_pos'],
        _quat2axisangle(env_obs['robot0_eef_quat']),
        env_obs['robot0_gripper_qpos']
    ])
    previous_obs['state'].append(current_state)
    previous_obs['state_timestamps'].append(env_obs['timestamp'])
    
    # Calculate target indices for state history
    current_idx = len(previous_obs['state']) - 1
    state_target_idx = np.array([current_idx] + 
                              [current_idx - step for step in args.state_down_sample_steps])
    
    # Ensure we have enough history
    if args.state_down_sample_steps:
        if len(previous_obs['state']) < max(args.state_down_sample_steps) + 1:
            # If we don't have enough history, duplicate the earliest available state
            earliest_idx = max(0, min(state_target_idx))
            for i in range(len(state_target_idx)):
                if state_target_idx[i] < 0:
                    state_target_idx[i] = earliest_idx
    
    # Get states at target indices
    state_history = np.array([previous_obs['state'][idx] for idx in state_target_idx])
    
    # For rotation, use SLERP interpolation
    if args.state_down_sample_steps:
        rot_history = state_history[:, 3:6]  # Get rotation part
        rot_interpolator = st.Slerp(
            times=np.arange(len(rot_history)),
            rotations=st.Rotation.from_rotvec(rot_history)
        )
        interpolated_rot = st.Rotation.as_rotvec(rot_interpolator(np.linspace(0, len(rot_history)-1, len(state_target_idx))))
        
        # For position and gripper, use linear interpolation
        pos_history = state_history[:, :3]  # Get position part
        gripper_history = state_history[:, 6:]  # Get gripper part
        
        pos_interpolator = si.interp1d(
            x=np.arange(len(pos_history)),
            y=pos_history,
            axis=0,
            assume_sorted=True
        )
        gripper_interpolator = si.interp1d(
            x=np.arange(len(gripper_history)),
            y=gripper_history,
            axis=0,
            assume_sorted=True
        )
        
        interpolated_pos = pos_interpolator(np.linspace(0, len(pos_history)-1, len(state_target_idx)))
        interpolated_gripper = gripper_interpolator(np.linspace(0, len(gripper_history)-1, len(state_target_idx)))
        
        # Combine interpolated states
        interpolated_state = np.concatenate([
            interpolated_pos,
            interpolated_rot,
            interpolated_gripper
        ], axis=1)
        
        # Flatten the state history into a single vector
        return_dict['state'] = interpolated_state.flatten()
    else:
        return_dict['state'] = state_history.flatten()
    
    return return_dict, previous_obs


def annotate_image_with_thoughts(img, policy_return_dict, prev_img=None):
    """
    Annotate the image with policy thoughts and status
    """
    # Create a copy of the image to avoid modifying the original
    annotated_img = img.copy()
    ## rgb to bgr
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

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




def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
   
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
    total_episodes, total_successes, total_lengths = 0, 0, 0

    ## check if behavior ood is enabled
    if args.behavior_ood == 1:
        task_range_in_suite = [10,23] # 13 novel tasks
    elif args.behavior_ood == 0:
        task_range_in_suite = [0,10]
        print('no behavior ood')
    else:
        raise ValueError(f"Unknown behavior task range: {args.behavior_ood}")
    for task_id in tqdm.tqdm(range(task_range_in_suite[0], task_range_in_suite[1])):#[52,53]):#range(num_tasks_in_suite)[:1]):
        
        
        ## check if visual ood is enabled
        if args.visual_ood in [1,2]:
            task_id = task_id + 13 + 10 * args.visual_ood
        elif args.visual_ood == 0:
            print('no visual ood')
        else:
            raise ValueError(f"Unknown visual ood: {args.visual_ood}")


        
        # Get task
        task = task_suite.get_task(task_id)
        # Start episodes
        task_episodes, task_successes, task_lengths = 0, 0, 0
        

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        ## check if semantic ood is enabled
        if args.semantic_ood in [1,2]:
            task_description = TASK_MAPPING[task_id][args.semantic_ood-1]
        elif args.semantic_ood == 0:
            print('no semantic ood')
        else:
            raise ValueError(f"Unknown semantic ood: {args.semantic_ood}")

        
        length_episodes = []
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            
            
            
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            replay_wrist_images = []
            ## create previous observation buffer
            previous_obs = {"image": None, "wrist_image": None, "state": None}
            
            # Initialize episode timing
            dt = 1.0 / args.frequency  # Time step in seconds
            
            
            logging.info(f"Starting episode {task_episodes+1}...")
            iter_idx = 0 
            inference_idx = args.steps_per_inference
            thinking_pose = None 
            all_time_actions = np.zeros((max_steps, max_steps + args.action_horizon, args.action_dim))
            
            action_plan = collections.deque()
            is_thinking = False
            last_annotated_img = None
            
            while t < max_steps + args.num_steps_wait:
                
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]) 
                
                    
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    img = image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                

                    wrist_img = image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    
                    if t == args.num_steps_wait:

                        ## warm up phase (not sure to keep or not)
                        for _ in tqdm.tqdm(range(args.steps_per_inference), desc='Warm up', leave=False):
                            env_obs = {
                                "agentview_image": img,
                                "robot0_eye_in_hand_image": wrist_img,
                                "robot0_eef_pos": obs["robot0_eef_pos"],
                                "robot0_eef_quat": obs["robot0_eef_quat"],
                                "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
                                "timestamp": 0.0 + (t-args.num_steps_wait) * dt
                            }
                    
                            obs_dict_np, previous_obs = get_pi_obs_dict(env_obs, previous_obs, task_description, args)
                    
                            obs_dict_np['is_warm_up'] = np.array([True])
                    
                            
                            action = client.infer(obs_dict_np)
                            
                            action_horizon = 16
                        
                        t += 1

                    env_obs = {
                        "agentview_image": img,
                        "robot0_eye_in_hand_image": wrist_img,
                        "robot0_eef_pos": obs["robot0_eef_pos"],
                        "robot0_eef_quat": obs["robot0_eef_quat"],
                        "robot0_gripper_qpos": obs["robot0_gripper_qpos"],
                        "timestamp": 0.0 + (t-args.num_steps_wait-1) * dt
                    }
                    if not is_thinking:
                        ## do not update the observation history when thinking (assume the agent keeps still when thinking)
                        obs_dict_np, previous_obs = get_pi_obs_dict(env_obs, previous_obs, task_description, args)
                        
            #     # Query model to get action or thinking if the current action queue is empty
                    if not action_plan:
                        ## current queue is empty, so we need to infer the action
                        policy_return_dict = client.infer(obs_dict_np)
                        
                    ## for every step, annotate the image with thoughts if there is any thought generated 
                    annotated_img = annotate_image_with_thoughts(img, policy_return_dict, last_annotated_img)
                    ## rgb to bgr
                
                    # Save annotated image for replay video
                    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)
                    last_annotated_img = annotated_img
                    ## if thinking, do not append the same annotated image to the replay_images buffer
                    if not is_thinking:

                        replay_images.append(annotated_img) 
                        replay_wrist_images.append(wrist_img)
                    
                    ## if the current action queue is empty, start to process the thinking or action 
                    if not action_plan:
                        ## policy should return empty action if thinking
                        if policy_return_dict.get('isthinking', False):
                            is_thinking = True
                            if thinking_pose is None:
                                print('Thinking...')
                                thinking_counter = 0 
                        
                    
                            if thinking_counter > 800:
                                t = max_steps + args.num_steps_wait
                            thinking_counter += 1 

                            thinking_pose = np.zeros(7)
                            action = None
                        
                        else:
                            ## the current inference is action not thinking 
                            is_thinking = False
                    
                            if thinking_pose is not None:
                                thinking_pose = None ## reset the thinking pose to None and starts the action execution
                            action = policy_return_dict['actions']
                            
                    ## inferred action is not None, so we can update the inference buffer
                    if thinking_pose is None: ## executing the action 
                        all_time_actions[[iter_idx],iter_idx:iter_idx + action_horizon] = action
                        

                    if not action_plan: 
                        if thinking_pose is not None:
                            ## pass in None action if the model is just thinking
                            this_target_poses = action
                            # print('this_target_poses shape for thinking is not none', this_target_poses.shape)
                        elif args.temporal_agg:
                            # temporal ensemble
                            ## current action sequence N x 16 x 7
                            ## blending the last 8 steps for temporal ensemble with exponential weighted average
                            action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + action_horizon]
                            
                            for i in range(8): ## action_horizon 
                                ensemble_num = 8
                                actions_for_curr_step = action_seq_for_curr_step[max(0, iter_idx - ensemble_num + 1):iter_idx + 1, i]
                                actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                                actions_for_curr_step = actions_for_curr_step[actions_populated]
                                ## exponential weighted average for temporal ensemble
                                k = -0.01
                                exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
                                exp_weights = exp_weights / exp_weights.sum()
                                weighted_rotvec = R.from_rotvec(np.array(actions_for_curr_step)[:, 3:6]).mean(weights=exp_weights).as_rotvec()
                                weighted_action = (actions_for_curr_step * exp_weights[:, np.newaxis]).sum(axis=0, keepdims=True)
                                weighted_action[0][3:6] = weighted_rotvec
                            
                                action_plan.extend(weighted_action)
                                this_target_poses = weighted_action
                            
                        else:
                            ## no temporal ensemble, so it is just executing the first k steps of predicted actions before next inference
                            action_new = action.copy()
                            
                            this_target_poses = action_new[:5]
                            
                            action_plan.extend(action_new[:5])
                            this_target_poses = action
                    ## if the current action queue is not empty, execute the action
                    if this_target_poses is not None:
                
                        actual_action = action_plan.popleft()
                        
                        obs, reward, done, info = env.step(actual_action.tolist())
                        t += 1
                        iter_idx += 1
                        if done:
                            task_successes += 1
                            total_successes += 1
                            break
                    if done:
                        ## end episode
                        break 
                    

                
        


            
            
            length_episodes.append(t-args.num_steps_wait)
            
            
            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode with thoughts
            suffix = f"success_{task_episodes}" if done else f"failure_{task_episodes}"
            task_segment = task_description.replace(" ", "_")
            print(args.video_out_path)
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_special_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_wrist_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_wrist_images],
                fps=10,
            )
            print('video saved')
            episode_length = len(replay_images)
            task_lengths += episode_length
            total_lengths += episode_length
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"Episode length: {episode_length}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task average length: {float(task_lengths) / float(task_episodes)}")
        logging.info(f"Current total average length: {float(total_lengths) / float(total_episodes)}")
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total average length: {float(total_lengths) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s', 
                                                filename='eval_libero_single_reason.log',
                                                                                                            filemode='a')
    tyro.cli(eval_libero)
