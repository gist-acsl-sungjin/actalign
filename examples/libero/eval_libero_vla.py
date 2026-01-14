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
import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
from utils import _quat2axisangle,_get_libero_env

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
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

@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    semantic_ood: int = 0 ## 0: no semantic ood, 1: semantic ood with rephrase, 2: semantic ood with object-property
    visual_ood: int = 0 ## 0: no visual ood, 1: visual ood with visual-background(background and viewpoint), 2: visual ood with viusal scene(distractor object)
    behavior_ood: int = 0 ## 0: no behavior ood, 1: behavior composition (13 novel tasks)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos_vla"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


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

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        ## check if semantic ood is enabled
        if args.semantic_ood in [1,2]:
            task_description = TASK_MAPPING[task_id][args.semantic_ood-1]
        elif args.semantic_ood == 0:
            print('no semantic ood')
        else:
            raise ValueError(f"Unknown semantic ood: {args.semantic_ood}")
       
        # Start episodes
        task_episodes, task_successes, task_lengths = 0, 0, 0
        
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
            q_values = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        output = client.infer(element)
                        action_chunk = output["actions"]
                        if 'q_values' in output:
                            q_values.append(output["q_values"])
                            print('q_values', q_values[-1])
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            ## record the length of the episode 
            print('success', done)
            episode_length = len(replay_images)
            task_lengths += episode_length
            total_lengths += episode_length
            logging.info(f"Episode length: {episode_length}")
            suffix = f"success_{task_episodes}" if done else f"failure_{task_episodes}"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            ## save the q_values
            if len(q_values) > 0:
                np.save(pathlib.Path(args.video_out_path) / f"q_values_{task_segment}_{suffix}.npy", q_values)
                print('q_values saved')
                print('q_values', min(q_values), max(q_values))
            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
          
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task average length: {float(task_lengths) / float(task_episodes)}")
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total average length: {float(total_lengths) / float(total_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                                    filename='eval_libero_vla.log',
                                                            filemode='a')
    tyro.cli(eval_libero)
