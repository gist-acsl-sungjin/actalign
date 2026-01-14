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
import json 
import os 
import numpy as np
import math
import pathlib
import cv2
import base64
import tqdm
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from vec_env import SampleVectorEnv
import torch
#from datasets import load_dataset
from torchvision import transforms
import PIL
#import datasets
from openpi_client import image_tools

RESIZE_SIZE = 224
MODEL_NAME = "gpt-4o"
RPC_TIMEOUT = 120.0  # seconds

# TASK_KEYWORD_MAPPLING = {
#     "alphabet soup" : "alphabet soup can (blue cylindrical can)",
#     "cream cheese" : "cream cheese (blue rectangular box)",
#     "butter" : "butter (red rectangular box)",
#     "tomato sauce": "tomato sauce (red and green cylindrical can)",
#     "chocolate pudding": "chocolate pudding (brown rectangular box)",
    
# }
TASK_KEYWORD_MAPPLING = {
    "alphabet soup" : "alphabet soup can (blue cylindrical can)",
    "cream cheese" : "cream cheese (blue rectangular box)",
    "butter" : "butter (red rectangular box)",
    "tomato sauce": "tomato sauce (red and green cylindrical can)",
    "chocolate pudding": "chocolate pudding (brown rectangular box)",
    "Pick up": "grasp",
    "the yellow and white mug" : "the mug that has yellow and white color on itself",
    "middle mug" : "the mug that has yellow and white color on itself",
    "left" : "right",
    "right" : "left",
    # "moka pot": "the handle (under the lid) of the moka pot",
    
}



def _encode_image_b64(img: np.ndarray) -> str:
    """
    Encode an image array to base64 string for GPT API calls.
    
    Args:
        img: Image array (H, W, C) in RGB format
        
    Returns:
        Base64 encoded string of the image
    """
    try:
        # Ensure 3 channels
        if img.ndim == 2:
            img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img3 = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        else:
            img3 = img
            
        # Encode to JPEG
        ok, buf = cv2.imencode('.jpg', img3)
        if not ok:
            # fallback: convert color space and retry
            img_bgr = cv2.cvtColor(img3, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode('.jpg', img_bgr)
            if not ok:
                raise RuntimeError("Image encoding failed")
        return base64.b64encode(buf.tobytes()).decode('utf-8')
    except Exception:
        # As last resort, use a white image to avoid crashing
        blank = np.ones((64, 64, 3), dtype=np.uint8) * 255
        ok, buf = cv2.imencode('.jpg', blank)
        return base64.b64encode(buf.tobytes()).decode('utf-8')
    
def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description

def make_vec_env(envs):
    return SampleVectorEnv(envs)

def _get_libero_vec_env(task, resolution, seed, env_num):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    # envs = [OffScreenRenderEnv(**env_args) for _ in range(batch_size)] 
    env = SampleVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args, horizon=5000) for _ in range(env_num)]
        )
    env.seed([seed]*env_num)
    return env, task_description

def raw_obs_to_numpy_obs(obs):
    """
    Prepare the tensor observations as input for the algorithm.
    """
    env_num = len(obs)

    data = {
    }

    all_obs_keys = obs[0].keys()
    
    for obs_name in all_obs_keys:
        data[obs_name] = []

    for k in range(env_num):
        for obs_name in all_obs_keys:
            if 'image' in obs_name:
                data[obs_name].append(image_tools.resize_with_pad(np.ascontiguousarray(obs[k][obs_name][::-1, ::-1]), RESIZE_SIZE, RESIZE_SIZE))
            else:
                data[obs_name].append(
                    obs[k][obs_name]
                )


    return data

def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

 

def _quat2axisangle_batch(quats):
    """
    Batch version of _quat2axisangle.
    Expects quats with shape (B, 4) in (x, y, z, w) order.
    Returns axis-angle vectors with shape (B, 3).
    """
    # q = np.asarray(quats, dtype=np.float64).copy()
    q = quats.copy()
    # clip quaternion
    q[:, 3] = np.clip(q[:, 3], -1.0, 1.0)

    den = np.sqrt(1.0 - q[:, 3] * q[:, 3])
    mask = np.isclose(den, 0.0)

    out = np.zeros((q.shape[0], 3), dtype=q.dtype)
    # out[~mask] = (q[~mask, :3] * 2.0 * np.arccos(q[~mask, 3])) / den[~mask] 
    out[~mask] = (q[~mask, :3] * 2.0 * np.arccos(q[~mask, 3])[:, None]) / den[~mask, None]

    return out

def get_prompt(instruction_text: str, query_type: str) -> str:
    ## load the prompt from the file
    with open(f"examples/libero/prompt_template_{query_type}.txt", "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt.format(TASK_INSTRUCTION=instruction_text)



def load_api_key(provider: str) -> str:
    with open("examples/libero/api_config.json") as f:
        config = json.load(f)
    api_key = config[provider]["api_key"]
    print(f"Loaded API key for {provider}: {api_key[:10]}...")
    return api_key
