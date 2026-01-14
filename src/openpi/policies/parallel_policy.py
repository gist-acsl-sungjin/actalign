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
from collections.abc import Sequence
import logging
import pathlib
from typing import Any, TypeAlias, Optional
import threading
import time

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
try:
    # only used for onboard testing
    from pynput import keyboard
except ImportError:
    pass
import numpy as np
from openpi_client import base_policy as _base_policy
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models.pi0_fuse import Pi0Fuse
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        # Make a batch and convert to jax.Array.
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)

        self._rng, sample_rng = jax.random.split(self._rng)
        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng, _model.Observation.from_dict(inputs), **self._sample_kwargs)[0],
        }

        # Unbatch and convert to np.ndarray.
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        return self._output_transform(outputs)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class ReasoningPolicy(BasePolicy):
    def __init__(
        self,
        model: Pi0Fuse,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        use_ref_img: bool = False,
        initial_scene_plan: str = '',
    ):
        self._prefill = nnx_utils.module_jit(model.prefill,
                                             static_argnames=('temprature',))
        self._act = nnx_utils.module_jit(model.act)
        self._reason = nnx_utils.module_jit(model.reason,
                                           static_argnames=('temprature',))
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._rng = rng or jax.random.key(0)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}

        ## binary variable for thinking mode or acting mode
        self._is_thinking = []  # List of boolean values for each batch element
        self._lock = threading.Lock()
        ## whether to use reference image 
        self._use_ref_img = use_ref_img
        ## initial scene plan
        self._initial_scene_plan: str = initial_scene_plan
        ## action placeholder
        self._action_placehoser = np.zeros((model.action_horizon, model.action_dim))
        ## temperature
        self._temperature = float(0)

        # Batch state management - will be initialized on first inference
        self._batch_thoughts = None
        self._batch_ref_imgs = None  # Will be initialized as list of arrays
        self._batch_scene_plans = None
        self._batch_instructions = None
        self._batch_interventions = None
        self._batch_robot_questions = None
        self._batch_user_answers = None
        self._current_batch_size = 0
        
        # Initialize last_thoughts for the thought property
        self.last_thoughts = []
        
        # Test flag for debugging
        self._test_mode = False

    def start(self):
        """
        Start a new rollout.
        Reset all batch thought states.
        """
        self._reset_batch_states()
        print("Start a new rollout.")
    
    def _reset_batch_states(self):
        """Reset all batch thought states using vectorized operations"""
        if self._batch_thoughts is not None:
            batch_size = len(self._batch_thoughts)
            # Reset all states in parallel using vectorized operations
            self._batch_thoughts = [None] * batch_size
            self._batch_ref_imgs = None # [None] * batch_size
            self._batch_scene_plans = [self._initial_scene_plan] * batch_size
            self._batch_instructions = [None] * batch_size
            self._batch_interventions = [None] * batch_size
            self._batch_robot_questions = [None] * batch_size
            self._batch_user_answers = [None] * batch_size
            self._is_thinking = [False] * batch_size

    def _reset_thinking_states(self, batch_idx: Optional[int] = None):
        """
        Reset thinking states for specific batch indices or all batch states.
        
        Args:
            batch_idx: If provided, reset only the specified batch index.
                      If None, reset all batch states (like warmup).
        """
        if self._batch_thoughts is None:
            print("No batch thoughts to reset")
            return  # Nothing to reset
        
        if batch_idx is None:
            # Reset all batch states (like warmup)
            print("Resetting all batch thinking states...")
            self._reset_batch_states()
            print("Reset all batch thinking states completed")
            print(f"Thinking state after reset: {self._is_thinking}")
        else:
            # Reset all batch states to the value of the specific batch index
            if 0 <= batch_idx < len(self._batch_thoughts):
                print(f"Resetting all batch states to match batch index {batch_idx}...")
                # Set entire batch to value at batch_idx
                # Direct assignment is fine since we're just copying primitive values
                # (None, strings, booleans) which are immutable in Python
                self._batch_thoughts = [self._batch_thoughts[batch_idx]] * len(self._batch_thoughts)
                self._batch_scene_plans = [self._batch_scene_plans[batch_idx]] * len(self._batch_scene_plans)
                self._batch_instructions = [self._batch_instructions[batch_idx]] * len(self._batch_instructions)
                self._batch_interventions = [self._batch_interventions[batch_idx]] * len(self._batch_interventions)
                self._batch_robot_questions = [self._batch_robot_questions[batch_idx]] * len(self._batch_robot_questions)
                self._batch_user_answers = [self._batch_user_answers[batch_idx]] * len(self._batch_user_answers)
                self._is_thinking = [self._is_thinking[batch_idx]] * len(self._is_thinking)
                
                # Reset reference image for this batch index if it exists
                if self._batch_ref_imgs is not None and batch_idx < self._batch_ref_imgs.shape[0]:
                    # Set to None or initial state - this will be re-initialized on next inference
                    pass
                
                print(f"Reset thinking states for batch index {batch_idx} completed")
                print(f"Thinking state after reset: {self._is_thinking}")
            else:
                print(f"Warning: Invalid batch index {batch_idx}, no reset performed")

    def _monitor_intervention(self):
        """
        Listen to the intervention signal from the user.

        If 'i' key is pressed, add the intervention.
        """
        def on_press(key):
            try:
                # When the 'i' key is pressed, add the intervention
                if key.char == 'i' and self._batch_interventions is not None:
                    # Add intervention to all batch elements
                    for i in range(len(self._batch_interventions)):
                        if self._batch_interventions[i] is None:
                            self._batch_interventions[i] = "User Intervention: I don't want the orange-flavored vodka. Please Add the lemon-flavored vodka instead.\n"
                    
                    print("\n'I' key pressed! Adding intervention to all batch elements...")
                    return False
            except AttributeError:
                # For non-character keys (like 'Esc', 'Shift', etc.),
                # we don't do anything.
                pass
        while True:
            with keyboard.Listener(on_press=on_press) as listener:
                listener.join()
                time.sleep(2)

    def _prepare_obs_batch(self, obs: dict, batch_size: int) -> dict:
        """
        Update the `obs` dict's 'thought' and 'ref_img' fields for batch processing.
        
        Args:
            obs: Dictionary containing batch of observations
            batch_size: Number of observations in the batch
        
        Returns:
            Updated observation dictionary with batch-processed thoughts and reference images
        """
        assert 'prompt' in obs, "The observation must contain a 'prompt' key."
        instructions = obs['prompt']  # Shape: (batch_size,)
        
        # Initialize or resize batch thought states
        self._ensure_batch_states(batch_size)
        
        # Get the instruction (use first element if it's a list, otherwise use as is)
        if isinstance(instructions, (list, tuple, np.ndarray)):
            instruction = instructions[0]  # Use the same instruction for all batch elements
        else:
            instruction = instructions
        
        # Prepare thoughts for all batch elements in parallel using vectorized operations
        instruction_prefix = 'Instruction: ' + instruction + '\n'
        
        # Initialize all needed elements in parallel
        for i in range(batch_size):
            if self._batch_instructions[i] is None:
                self._batch_instructions[i] = instruction_prefix
                self._batch_thoughts[i] = instruction_prefix + self._batch_scene_plans[i]
        
        # Collect all thoughts (already initialized or existing)
        batch_thoughts = self._batch_thoughts.copy()
        
        obs['thought'] = [[batch_thought] for batch_thought in batch_thoughts]
        
        # Handle reference images in parallel
        if 'image_1' in obs:
            images = obs['image_1']  # Shape: (batch_size, H, W, C)
            if self._batch_ref_imgs is None or self._batch_ref_imgs.shape[0] != batch_size:
                if len(images.shape) > 3:
                    # Batch of images - make a copy as array
                    self._batch_ref_imgs = np.array(images, copy=True)
                else:
                    # Single image - tile to batch and make a copy
                    self._batch_ref_imgs = np.stack([images] * batch_size, axis=0)
            
            if self._use_ref_img:
                obs['reference_image'] = self._batch_ref_imgs
        
        # Set flags in parallel using list multiplication
        obs['act_with_outdated_thought'] = [False] * batch_size
        obs['think_with_outdated_thought'] = [False] * batch_size
        
        return obs
    
    def _ensure_batch_states(self, batch_size: int):
        """
        Ensure batch states are initialized with the correct size.
        If batch size changes, resize the states accordingly.
        
        Args:
            batch_size: Required batch size
        """
        if self._batch_thoughts is None:
            # Initialize for the first time using vectorized operations
            # print(f"Initializing batch states with size: {batch_size}")
            self._batch_thoughts = [None] * batch_size
            self._batch_ref_imgs = None  # Force re-init as array in _prepare_obs_batch
            self._batch_scene_plans = [self._initial_scene_plan] * batch_size
            self._batch_instructions = [None] * batch_size
            self._batch_interventions = [None] * batch_size
            self._batch_robot_questions = [None] * batch_size
            self._batch_user_answers = [None] * batch_size
            self._is_thinking = [False] * batch_size
            self._current_batch_size = batch_size
        elif self._current_batch_size != batch_size:
            # Resize if batch size changed using vectorized operations
            # print(f"Resizing batch states from {self._current_batch_size} to {batch_size}")
            if batch_size > self._current_batch_size:
                # Expand using list extension (parallel operation)
                expansion_size = batch_size - self._current_batch_size
                self._batch_thoughts.extend([None] * expansion_size)
                self._batch_scene_plans.extend([self._initial_scene_plan] * expansion_size)
                self._batch_instructions.extend([None] * expansion_size)
                self._batch_interventions.extend([None] * expansion_size)
                self._batch_robot_questions.extend([None] * expansion_size)
                self._batch_user_answers.extend([None] * expansion_size)
                self._is_thinking.extend([False] * expansion_size)
            else:
                # Shrink using slice operations (parallel operation)
                self._batch_thoughts = self._batch_thoughts[:batch_size]
                self._batch_scene_plans = self._batch_scene_plans[:batch_size]
                self._batch_instructions = self._batch_instructions[:batch_size]
                self._batch_interventions = self._batch_interventions[:batch_size]
                self._batch_robot_questions = self._batch_robot_questions[:batch_size]
                self._batch_user_answers = self._batch_user_answers[:batch_size]
                self._is_thinking = self._is_thinking[:batch_size]
            self._batch_ref_imgs = None  # Force re-init as array in _prepare_obs_batch
            self._current_batch_size = batch_size
    
    def _update_thought_batch(self, model_output: str, obs: dict, batch_idx: int):
        """
        Update the thought and ref_img fields for a specific batch element.
        Optimized for parallel processing.
        
        Args:
            model_output: Model output for this batch element
            obs: Original observation dictionary
            batch_idx: Index of the batch element to update
        """
        # Update scene plan if not a question
        ## manually update the text plan to ensure the text plan is always correct 
        # if self._batch_scene_plans[batch_idx]  == self._initial_scene_plan:
        #     if 'alphabet' in self._batch_instructions[batch_idx] and ( 'tomato' in model_output or 'cream cheese' in model_output):
        #         self._batch_scene_plans[batch_idx] =  ('Plan: 1. Pick up the can of alphabet soup.\n2. Place the can of alphabet soup in the basket.\n3. Pick up the butter.\n4. Place the butter in the basket.\n'
        #             'What I have done: Nothing.\n'
        #             'Now I need to do: pick up the can of alphabet soup.\n')
        #     elif 'cream cheese' in self._batch_instructions[batch_idx] and ('alphabet' in model_output or ('butter' in model_output)):
        #         self._batch_scene_plans[batch_idx] =  ('Plan: 1. pick up the cream cheese box.\n2. place the cream cheese box in the basket.\n3. pick up the tomato sauce.\n4. place the tomato sauce in the basket.\n'
        #             'What I have done: Nothing.\n'
        #             'Now I need to do: pick up the cream cheese box.\n')
        #     else:
        #         self._batch_scene_plans[batch_idx] = model_output
        if 'Robot Question:' not in model_output: 
        # else:
            self._batch_scene_plans[batch_idx] = model_output
        
        ## TODO: not add the question part now since we don't use question, user intervention, robot question in our policy currently
        # # Update robot question and user answer if not already set
        # if (self._batch_robot_questions[batch_idx] is None and 
        #     'Robot Question' in model_output):
        #     self._batch_robot_questions[batch_idx] = 'Robot Question: Which cocktail would you like? We have three options: Mojito, Mount Fuji, and Vodka Sunrise.\n'
        #     self._batch_user_answers[batch_idx] = 'User Answer: I want a cup of Vodka Sunrise.\n'
        
        # Build updated thought using efficient string operations
        # intervention = self._batch_interventions[batch_idx] or ''
        # robot_question = self._batch_robot_questions[batch_idx] or ''
        # user_answer = self._batch_user_answers[batch_idx] or ''
        
        self._batch_thoughts[batch_idx] = (
            self._batch_instructions[batch_idx] + 
            # robot_question + user_answer + intervention + 
            self._batch_scene_plans[batch_idx]
        )

        # Update reference image efficiently
        if 'image_1' in obs:
            images = obs['image_1']
            if len(images.shape) > 3:
                self._batch_ref_imgs[batch_idx] = np.array(images[batch_idx], copy=True)
            else:
                self._batch_ref_imgs[batch_idx] = np.array(images, copy=True)
        
        print(f'Updated prefix for batch element {batch_idx}:')
        print(self._batch_thoughts[batch_idx])
        print('-----------------------------------\n')
        return self._batch_scene_plans[batch_idx]
    
    def _clear_jax_cache_if_needed(self, new_batch_size: int):
        """
        Clear JAX compilation cache if batch size changes to prevent compilation mismatches.
        
        Args:
            new_batch_size: The new batch size being used
        """
        if hasattr(self, '_last_compiled_batch_size') and self._last_compiled_batch_size != new_batch_size:
            print(f"Clearing JAX cache due to batch size change: {self._last_compiled_batch_size} -> {new_batch_size}")
            # Clear JAX compilation cache
            jax.clear_caches()
            self._last_compiled_batch_size = new_batch_size
        elif not hasattr(self, '_last_compiled_batch_size'):
            self._last_compiled_batch_size = new_batch_size

    def _warmup(self, inputs: dict) -> dict:
        """warmup jit compilation"""
        print('Warmup...')
        # Get batch size from the first observation field
        batch_size = None
        for key, value in inputs.items():
            if hasattr(value, 'shape') and len(value.shape) > 0:
                batch_size = value.shape[0]
                # print(f"Warmup detected batch size: {batch_size} from field '{key}'")
                break
        
        if batch_size is None:
            print("Warning: Could not detect batch size from inputs, using batch_size=1 for warmup")
            batch_size = 1
        
        # print(f"Using batch size {batch_size} for warmup")
        
        # Clear JAX cache if batch size changed
        # self._clear_jax_cache_if_needed(batch_size)
        
        inputs = self._prepare_obs_batch(inputs, batch_size)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x), inputs)
        inputs = _model.FuseObservation.from_dict(inputs)

        prefill_rng, think_rng, act_rng, self._rng = jax.random.split(self._rng, 4)

        # prefill and decide to act or think
        processed_inputs, prefix_cache, first_suffix_token, \
            last_logit, prefix_mask, prefix_positions, to_act = self._prefill(prefill_rng, inputs)
        
        # forward both think and act
        actions = self._act(act_rng, processed_inputs, prefix_cache,
                            prefix_mask, prefix_positions)
        suffix_tokens = self._reason(think_rng, last_logit,
                                    prefix_cache, prefix_mask, prefix_positions)
        
        outputs = {
            "state": inputs.state,
            "actions": actions,
            "tokenized_suffix": suffix_tokens,
        }

        self._reset_batch_states()
        # Convert to np.ndarray
        outputs = jax.tree.map(lambda x: np.asarray(x), outputs)
        transformed = self._output_transform(outputs)
        return transformed

    @override
    def infer(self, obs: dict) -> dict:
        """
        Infer actions from batch observations.
        
        Args:
            obs: Dictionary containing batch of observations. Each field should have batch dimension as first dimension.
                 Expected keys: 'prompt', 'image_1', 'state', etc.
        
        Returns:
            Dictionary containing batch of outputs with same batch dimension as input.
        """
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        ## remember the last observation 
        # Handle warmup case
        warmup = inputs.pop('is_warm_up', False)
        if warmup:
            return self._warmup(inputs)
        
        # Handle reset thinking case
        reset_thinking = inputs.pop('reset_thinking', False)
        if reset_thinking:
            batch_idx = inputs.pop('batch_idx', None)
            print(f"Processing reset thinking request with batch_idx: {batch_idx}")
            self._reset_thinking_states(batch_idx)
            # Return properly formatted response to acknowledge reset
            # The client expects actions and isthinking fields
            reset_response = {
                'actions': np.zeros((1, self._action_placehoser.shape[0], self._action_placehoser.shape[1])),
                'isthinking': [True, True],
                'thought': self._batch_thoughts,
              
            }
            return reset_response
          
        
        # Get batch size from the first observation field
        batch_size = None
        for key, value in inputs.items():
            if hasattr(value, 'shape') and len(value.shape) > 0:
                batch_size = value.shape[0]
               
                break
        
        if batch_size is None:
            raise ValueError("Could not determine batch size from input observations")
        
     
        
        # Prepare batch observations
        batch_inputs = self._prepare_obs_batch(inputs, batch_size)
        batch_prefixes = batch_inputs['thought']
        
        # Apply input transforms
        batch_inputs = self._input_transform(batch_inputs)
        
        # Convert to jax.Array (already has batch dimension)
        batch_inputs = jax.tree.map(lambda x: jnp.asarray(x), batch_inputs)
        batch_inputs = _model.FuseObservation.from_dict(batch_inputs)

        # Split RNG for each batch element
        prefill_rng, think_rng, act_rng, self._rng = jax.random.split(self._rng, 4)

        # Prefill and decide to act or think for each batch element
        (
            processed_inputs,
            prefix_cache,
            first_suffix_token,
            last_logit,
            prefix_mask,
            prefix_positions,
            to_act
        ) = self._prefill(prefill_rng, batch_inputs, temprature=self._temperature)
        
        # Handle each batch element's decision
        to_act = jnp.asarray(to_act)  # Shape: (batch_size,)
        to_think = ~to_act  # Shape: (batch_size,)
        
        # Check if any batch elements need to think due to initial scene plan
        initial_scene_plan_mask = jnp.array([
            scene_plan == self._initial_scene_plan 
            for scene_plan in self._batch_scene_plans
        ])
        to_think = to_think | initial_scene_plan_mask
        to_act = ~to_think
        
        # Update thinking state for each batch element
        self.is_thinking = to_think.tolist()

        # Initialize outputs with placeholders
        actions = jnp.tile(self._action_placehoser[np.newaxis, ...], (batch_size, 1, 1))
        # default: '<eos>' token - single token per batch element
        suffix_tokens = jnp.ones((batch_size, 1), dtype=jnp.int32)
        
        # Process actions for batch elements that should act
        if jnp.any(to_act):
            act_indices = jnp.where(to_act)[0]
            print(f'Batch elements {act_indices} are acting...')
            act_actions = self._act(act_rng, processed_inputs, prefix_cache, 
                                   prefix_mask, prefix_positions)
            actions = actions.at[act_indices].set(act_actions[act_indices])
        
        # Process reasoning for batch elements that should think
        if jnp.any(to_think):
            think_indices = jnp.where(to_think)[0]
            print(f'Batch elements {think_indices} are thinking...')
            for id in think_indices:
                print(f'----Prefix for batch element {id}----')
                print(batch_prefixes[id][0])
                print('----Thnking------')
            
            # Process the entire batch to avoid batch size mismatch issues
            suffix_tokens = self._reason(
                think_rng, last_logit, prefix_cache, 
                prefix_mask, prefix_positions, 
                temprature=self._temperature
            )
           
        
        outputs = {
            "state": batch_inputs.state,
            "actions": actions,
            "tokenized_suffix": suffix_tokens,
        }

        # Convert to np.ndarray
        outputs = jax.tree.map(lambda x: np.asarray(x), outputs)
        transformed = self._output_transform(outputs)

        # Update thoughts for batch elements that were thinking
        if jnp.any(to_think):
            think_indices = jnp.where(to_think)[0]
            think_thoughts = [transformed['thoughts'][think_idx] for think_idx in think_indices]
            assert transformed['thoughts'] != ['']
            self.last_thoughts = transformed['thoughts']
            # Update thoughts for each thinking batch element
            for i, batch_idx in enumerate(think_indices):
                batch_idx = batch_idx.item()

                if self._batch_scene_plans[batch_idx] == self._initial_scene_plan:
                    update_thoughts = self._update_thought_batch(think_thoughts[i], inputs, batch_idx)

                    transformed['thoughts'][batch_idx] = update_thoughts
                else: 
                    self._update_thought_batch(think_thoughts[i], inputs, batch_idx)

            
            # If all batch elements were thinking, return thinking response
            if jnp.all(to_think):
                # Set all thinking states to False after thinking is complete
                self.is_thinking = [False] * batch_size
                return {
                    'isthinking': np.full(batch_size, True, dtype=bool),
                    'thought': transformed['thoughts']
                }
            
            else:
                # Set thinking states to False for elements that just finished thinking
                ## deal with the situation where one batch element is thinking and the other is not
               
                for idx in think_indices:
                    self._is_thinking[idx.item()] = False
                new_response = transformed.copy()
                ## true where thinking_indices is 
                new_response['isthinking'] =  np.array([True if i in think_indices else False for i in range(batch_size)])
              
                return new_response
   

        return transformed
    
    @property
    def thought(self) -> list[str]:
        with self._lock:
            return self.last_thoughts
    
    @property
    def is_thinking(self) -> list[bool]:
        with self._lock:
            return self._is_thinking
    
    @is_thinking.setter
    def is_thinking(self, value: list[bool] | bool):
        with self._lock:
            if isinstance(value, bool):
                # Convert single boolean to list for backward compatibility
                if self._current_batch_size > 0:
                    self._is_thinking = [value] * self._current_batch_size
                else:
                    self._is_thinking = [value]
            else:
                # Direct list assignment
                self._is_thinking = value

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
