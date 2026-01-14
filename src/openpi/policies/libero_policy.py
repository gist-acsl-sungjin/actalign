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
import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "state": np.random.rand(24), ## UMI:48
        "image_1": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_2": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "image_3": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
   
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        mask_padding = self.model_type == _model.ModelType.PI0  # We don't mask for pi0-FAST.

        # Get the state. We are padding from 8 to the model action dim.
        # For pi0-FAST, we don't pad the state (action_dim = 7, which is < 8, so pad is skipped).
        state = transforms.pad_to_dim(data["state"], self.action_dim)
        

        history_length = 1
        while True:
            if f"image_{history_length + 1}" not in data:
                break
            history_length += 1
        
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        image_dict, image_mask_dict = {}, {}
        for i in range(history_length):
            ## if data[f"image_{i + 1}"] has 4 dimensions, then need to iterate over to parse each image
            
            if data[f"image_{i + 1}"].ndim == 4:
                image_list = []
                for j in range(data[f"image_{i + 1}"].shape[0]):
                    
                    image = _parse_image(data[f"image_{i + 1}"][j])
                    image_list.append(image)
                image_dict[f"{i}_rgb"] = np.stack(image_list, axis=0)
                ## get batch size of the true
                image_mask_dict[f"{i}_rgb"] = np.ones(data[f"image_{i + 1}"].shape[0], dtype=np.bool_)
            else:
                
                image = _parse_image(data[f"image_{i + 1}"])
                image_dict[f"{i}_rgb"] = image
                image_mask_dict[f"{i}_rgb"] = np.True_
            if 'image_wrist_{}'.format(i + 1) in data.keys():
                ## if data[f"image_wrist_{i + 1}"] has 4 dimensions, then need to iterate over to parse each image
                if data[f"image_wrist_{i + 1}"].ndim == 4:
                    image_list = []
                    for j in range(data[f"image_wrist_{i + 1}"].shape[0]):
                      
                        image = _parse_image(data[f"image_wrist_{i + 1}"][j])
                        image_list.append(image)
                    image_dict[f"wrist_{i}_rgb"] = np.stack(image_list, axis=0)
                    image_mask_dict[f"wrist_{i}_rgb"] = np.ones(data[f"image_wrist_{i + 1}"].shape[0], dtype=np.bool_)
                else:
                 
                    image = _parse_image(data['image_wrist_{}'.format(i + 1)])
                    image_dict[f"wrist_{i}_rgb"] = image
                    image_mask_dict[f"wrist_{i}_rgb"] = np.True_
        if 'reference_image' in data.keys():
            if data['reference_image'].ndim == 4:
                image_list = []
                for j in range(data['reference_image'].shape[0]):
                    image = _parse_image(data['reference_image'][j])
                    image_list.append(image)
                image_dict['reference_rgb'] = np.stack(image_list, axis=0)
                image_mask_dict['reference_rgb'] = np.ones(data['reference_image'].shape[0], dtype=np.bool_)
            else:
                image = _parse_image(data['reference_image'])
                image_dict['reference_rgb'] = image
                image_mask_dict['reference_rgb'] = np.True_
        add_prompt_info = None
        if 'condition' in data.keys():
            if data['condition'] is None:
                image_dict['start_rgb'] = np.zeros_like(image_dict['0_rgb'])
                image_mask_dict['start_rgb'] = np.False_
            else:
                image_dict['start_rgb'] = _parse_image(data['condition']['episode_start_image'])
                image_mask_dict['start_rgb'] = np.True_
                add_prompt_info = '. Objects are located at ' + str(data['condition']['detect']) + '.'

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": image_dict,
            "image_mask": image_mask_dict
        }

        

        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if add_prompt_info is not None:
            inputs["prompt"] += add_prompt_info

        if 'thought' in data.keys():
            inputs['thought'] = data['thought']
            inputs['act_with_outdated_thought'] = data['act_with_outdated_thought']
            inputs['think_with_outdated_thought'] = data['think_with_outdated_thought']

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        # return {"actions": np.asarray(data["actions"][:, :7])} 
        ## check the action dimension 
        ## if actions are 2d with shape (action_prediction_horizon, 7)
        ## if actions are 3d with shape (batch_size, action_prediction_horizon, 7)
        if data["actions"].ndim == 3:
            data.update({"actions": np.asarray(data["actions"][:, :, :7])})
        else:
            data.update({"actions": np.asarray(data["actions"][:, :7])})
        return data
