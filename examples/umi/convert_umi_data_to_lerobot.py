"""
Usage:
uv run examples/umi/convert_umi_data_to_lerobot.py --data_dir /path/to/your/data, --repo_name ${REPO_NAME} 
"""
import json
import os
import shutil

import numpy as np
import tyro
import zarr
from imagecodecs_numcodecs import register_codecs
from lerobot.common.datasets.lerobot_dataset import (LEROBOT_HOME,
                                                     LeRobotDataset)
from umi_replay_buffer import ReplayBuffer
from PIL import Image

register_codecs()

REPO_NAME = "umi/your-repo-name"  # Name of the output dataset, also used for the Hugging Face Hub


def main(data_dir: str | None = None, vl_data_path_list: str | None = None, vl_number_per_task: str | None = None, vl_task_list: str | None = None, repo_name: str = REPO_NAME, *, push_to_hub: bool = False):
    """
    Convert the UMI dataset to the LeRobot format.

    Args:
        data_dir: Path to the UMI dataset directory.
        repo_name: Name of the output dataset, also used for the Hugging Face Hub.
        push_to_hub: Whether to push the dataset to the Hugging Face Hub.
    """

    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=60,
        features={
            "image": {
                "dtype": "image",
                "shape": (224, 224, 3),
                "names": ["height", "width", "channel"],
            },
            "eef_pos": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["eef_pos"],
            },
            "eef_rot_axis_angle": {
                "dtype": "float32",
                "shape": (3,),
                "names": ["eef_rot_axis_angle"],
            },
            "gripper_width": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_width"],
            },
            "demo_start_pose": {
                "dtype": "float32",
                "shape": (6,),
                "names": ["demo_start_pose"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["actions"],
            },
            "episode_start_idx": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["episode_start_idx"],
            },
            "episode_end_idx": {
                "dtype": "int64",
                "shape": (1,),
                "names": ["episode_end_idx"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    if data_dir is not None:
        with zarr.ZipStore(os.path.join(data_dir, 'dataset.zarr.zip'), mode='r') as zip_store:
            replay_buffer = ReplayBuffer.copy_from_store(
                src_store=zip_store, 
                store=zarr.MemoryStore()
            )

        with open(os.path.join(data_dir, "instruction.txt"), "r") as f:
            instructions = f.readlines()

        episode_ends = replay_buffer.episode_ends[:]
        for i in range(len(episode_ends)):
            start_idx = 0 if i == 0 else episode_ends[i-1]
            end_idx = episode_ends[i]
            for current_idx in range(start_idx, end_idx):
                eef_pos, eef_rot_axis_angle, gripper_width = replay_buffer["robot0_eef_pos"][current_idx], replay_buffer["robot0_eef_rot_axis_angle"][current_idx], replay_buffer["robot0_gripper_width"][current_idx]
                actions = np.concatenate([replay_buffer["robot0_eef_pos"][current_idx], replay_buffer["robot0_eef_rot_axis_angle"][current_idx], replay_buffer["robot0_gripper_width"][current_idx]])
                dataset.add_frame(
                    {
                        "image": replay_buffer["camera0_rgb"][current_idx],  # Image.fromarray(replay_buffer["camera0_rgb"][current_idx]).save('bb.png')
                        "eef_pos": eef_pos,
                        "eef_rot_axis_angle": eef_rot_axis_angle,
                        "gripper_width": gripper_width,
                        "demo_start_pose": replay_buffer["robot0_demo_start_pose"][current_idx],
                        "actions": actions,
                        "episode_start_idx": start_idx,
                        "episode_end_idx": end_idx,
                    }
                )
            dataset.save_episode(task=instructions[i].strip())

    if vl_data_path_list is not None:
        vl_number_per_task = vl_number_per_task.split(',')
        vl_number_per_task_list = [int(number) for number in vl_number_per_task]
        vl_task_list = vl_task_list.split(',')
        vl_data_path_list = vl_data_path_list.split(',')
        assert len(vl_task_list) == len(vl_number_per_task_list), f"vl_task_list and vl_number_per_task_list should have the same length, but got {len(vl_task_list)} and {len(vl_number_per_task_list)}"
        print('total vl image number', sum(vl_number_per_task_list), f'{len(vl_task_list)} vl task')
        for i, vl_data_path in enumerate(vl_data_path_list):
            print(f'vl data path {i}: ', vl_data_path)

        vl_episode_start_idx = 0 if data_dir is None else replay_buffer.episode_ends[-1]
        for single_vl_task, single_vl_number in zip(vl_task_list, vl_number_per_task_list):
            print(f'vl task: {single_vl_task}, data number: {single_vl_number}')
            for i in range(single_vl_number):
                for vl_data_path in vl_data_path_list:
                    image_path = os.path.join(vl_data_path, single_vl_task, f"{i}.png")
                    image = np.array(Image.open(image_path).convert("RGB"))  # Image.fromarray(image).save('aa.png')
                    dataset.add_frame(
                        {
                            "image": image,
                            "eef_pos": np.zeros((3,), dtype=np.float32),
                            "eef_rot_axis_angle": np.zeros((3,), dtype=np.float32),
                            "gripper_width": np.zeros((1,), dtype=np.float32),
                            "demo_start_pose": np.zeros((6,), dtype=np.float32),
                            "actions": np.zeros((7,), dtype=np.float32),
                            "episode_start_idx": vl_episode_start_idx + len(vl_data_path_list) * i,
                            "episode_end_idx": vl_episode_start_idx + len(vl_data_path_list) * (i + 1),
                        }
                    )
                dataset.save_episode(task='This is vision-language data.')
            vl_episode_start_idx += len(vl_data_path_list) * single_vl_number

    # Consolidate the dataset, skip computing stats since we will do that later
    dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
