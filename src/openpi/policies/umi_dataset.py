from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import os
import json
import torch
import copy
import scipy.interpolate as si
import scipy.spatial.transform as st
from openpi.policies.pose_util import pose_to_mat, mat_to_pose10d, mat_to_pose
from openpi.policies.pose_repr_util import convert_pose_mat_rep
from openpi.training.config import UMIDataConfig


def _get_thought(thoughts: list[dict], step: int) -> dict[str, str|None]:
    for thought in thoughts:
        end_step = thought['end_step']
        if end_step == -1:
            end_step = 1e9
        if thought['start_step'] <= step < end_step:
            return thought

    raise ValueError(f"No thought found for step {step}")


class UMIDataset(LeRobotDataset):
    def __init__(self, data_config: UMIDataConfig, action_horizon: int):
        super().__init__(data_config.repo_id, local_files_only=data_config.local_files_only)
        self.data_config = data_config

        # umi related variables
        self.image_hisory_length = len(data_config.image_down_sample_steps) + 1
        self.image_down_sample_steps = data_config.image_down_sample_steps
        self.state_hisory_length = len(data_config.state_down_sample_steps) + 1
        self.state_down_sample_steps = data_config.state_down_sample_steps
        self.action_horizon = action_horizon
        self.action_down_sample_steps = data_config.action_down_sample_steps

        # how to calculate the state
        # we merge historical state into one single vector
        self.getitem_type = data_config.getitem_type
        # whether to enable reasoning training
        self.use_reasoning = data_config.use_reasoning
        # whether to use the reference image
        self.use_reference_image = data_config.use_reference_image
        # training or computing norm stats
        # we need this flag to skip vision-language data
        # when computing norm stats
        self.is_computing_norm_stats = data_config.is_computing_norm_stats
        # probability to predict reasoning content
        # used for reasoning segments
        self.pred_reasoning_prob = 0.7
        # wheter to use outdated reasoning content
        self.use_outdated_reasoning = data_config.use_outdated_reasoning

        self.low_dim_keys = ["eef_pos", "eef_rot_axis_angle", "gripper_width", "demo_start_pose"]
        self.low_dim_features = {}
        for key in self.low_dim_keys:
            self.low_dim_features[key] = torch.stack(self.hf_dataset[key]).numpy().astype(np.float32)
        self.low_dim_features['gripper_width'] = self.low_dim_features['gripper_width'][:, None]
        self.actions = torch.stack(self.hf_dataset['actions']).numpy().astype(np.float32)

        episode_ends = self.hf_dataset['episode_end_idx']
        self.episode_ends = np.unique(episode_ends)

        if data_config.reasoning_json_path is not None:
            with open(data_config.reasoning_json_path, 'r') as f:
                _loaded = json.load(f)
                self.reasoning = {}
                for k, v in _loaded.items():
                    if k.isdigit():
                        self.reasoning[int(k)] = v
                    else:
                        self.reasoning[k] = v

        if data_config.create_train_val_split:
            # only for *creating* train/val split
            # need to be false when training
            assert data_config.use_val_dataset
            self.create_train_val_split()
        if data_config.use_val_dataset:
            self.indices = self.get_indices('train')
        else:
            self.indices = list(range(len(self.hf_dataset)))

        self.rdm = np.random.RandomState(self.data_config.seed)

    def create_train_val_split(self):
        episode_num = len(self.episode_ends)
        val_num = int(episode_num * self.data_config.val_ratio)
        np.random.seed(self.data_config.seed)
        train_episode_idx = np.random.choice(episode_num, episode_num - val_num, replace=False)
        train_episode_idx = np.sort(train_episode_idx)
        val_episode_idx = np.setdiff1d(np.arange(episode_num), train_episode_idx)
        os.makedirs(self.data_config.norm_stats_dir, exist_ok=True)
        with open(os.path.join(self.data_config.norm_stats_dir, 'train_val_split.json'), 'w') as f:
            json.dump({'train_episode_idx': train_episode_idx.tolist(), 'val_episode_idx': val_episode_idx.tolist()}, f)

    def get_indices(self, split):
        with open(os.path.join(self.data_config.norm_stats_dir, 'train_val_split.json'), 'r') as f:
            split_idx = json.load(f)[f'{split}_episode_idx']
        indices = []
        for idx in split_idx:
            start_idx = 0 if idx == 0 else self.episode_ends[idx-1]
            end_idx = self.episode_ends[idx]
            if self.is_computing_norm_stats:
                if (
                    self.data_config.reasoning_json_path is not None and
                    idx in self.reasoning['vision_language_episode_idx']
                ):
                    continue
            indices += list(range(start_idx, end_idx))
        return indices
    
    def get_val_dataset(self):
        val_set = copy.copy(self)
        val_set.indices = self.get_indices('val')
        val_set.pred_reasoning_prob = 1.0
        return val_set

    def __len__(self):
        return len(self.indices)
    
    def get_prob(
            self,
            start_step: int,
            end_step: int,
            now_step: int,
            start_prob: float=0.8,
            end_prob: float=0.4,
        ) -> float:
        """
        linerly interpolate the probability from start_prob to end_prob
        """
        # from start_prob -> end_prob linearly
        assert start_step <= now_step < end_step
        return start_prob - (start_prob - end_prob) * (now_step - start_step) / (end_step - start_step)

    def __getitem__(self, idx: int) -> dict:
        """
        return_dict['thought'] is a list of strings
            - if the length is 1, it only contains the latest reasoning content
            - if the length is 2, it contains the latest reasoning content and the updated reasoning content
        """
        idx = self.indices[idx]
        low_dim_dict, return_dict = {}, {}
        current_idx_item = self.hf_dataset[idx]
        start_idx, end_idx = current_idx_item['episode_start_idx'].item(), current_idx_item['episode_end_idx'].item()
        
        freeze_action = False

        if self.use_reasoning:
            episode_id = current_idx_item['episode_index'].item()
            reasonings = self.reasoning[episode_id]['segments']
            return_dict['act_with_outdated_thought'], return_dict['think_with_outdated_thought'] = False, False

            # ========== vision-language data ==========
            if episode_id in self.reasoning['vision_language_episode_idx']:
                # (we have no action or state)
                assert not self.is_computing_norm_stats

                # for vision-language data, segments are different (instruction, reasoning) pairs
                reasoning_idx = self.rdm.randint(0, len(reasonings))
                return_dict['thought'] = [reasonings[reasoning_idx]['content'], reasonings[reasoning_idx]['updated_content']]
                return_dict['image_1'] = self.hf_dataset[idx]['image']
                freezing_action = [0., 0., 0., 1., 0., 0., 0., 1., 0, 0.08623]
                return_dict['actions'] = torch.tensor(freezing_action, dtype=torch.float32).repeat(self.action_horizon, 1)
                return_dict['action_is_pad'] = torch.tensor([True] * self.action_horizon)
                if self.getitem_type == 'necessary':
                    # an rest state for umi
                    # we assume the reasoning happens when the robot is at the start pose
                    # but it is not used as we do not supervise action for vision-language data
                    return_dict['state'] = torch.tensor([0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.08623, 0.08623, 0.08623], dtype=torch.float32)
                copy_key = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
                for key in copy_key:
                    return_dict[key] = current_idx_item[key]
                return return_dict

            # ========== robot data ==========
            # interval of the first segment
            episode_start_interval = self.reasoning[episode_id]['episode_start_interval']
            # use relative indexing within the episode to obtain reasoning content
            reasoning_dict = _get_thought(reasonings, idx - start_idx)
            # to get reference image
            reference_start_step, reference_end_step = reasoning_dict['reference_start_step'], reasoning_dict['reference_end_step']

            # ========== human-robot interaction segments ==========
            # robot asks question, user answers
            if 'user_answer' in reasoning_dict.keys():
                # probabilities of the four cases
                content_prob, updated_content_w_instruction_prob, user_answer_prob, user_answer_update_w_instruction_prob = 0.2, 0.2, 0.2, 0.4
                random_number = self.rdm.rand()
                # ===== case 1: input previous reasoning, output robot answer =====
                if random_number < content_prob:
                    ref_interval_start_step, ref_interval_end_step = idx - start_idx, idx - start_idx + 1
                    return_dict['thought'] = [reasoning_dict['content'], reasoning_dict['updated_content']]
                # ===== case 2: input robot answer, output static action =====
                # robot stays still when waiting for the user to answer
                elif random_number < content_prob + updated_content_w_instruction_prob:
                    ref_interval_start_step, ref_interval_end_step = idx - start_idx, idx - start_idx + 1
                    return_dict['thought'] = [reasoning_dict['updated_content_w_instruction']]
                    freeze_action = True
                # ===== case 3: input user answer, output reasoning after user answer =====
                elif random_number < content_prob + updated_content_w_instruction_prob + user_answer_prob:
                    ref_interval_start_step, ref_interval_end_step = idx - start_idx, idx - start_idx + 1
                    return_dict['thought'] = [reasoning_dict['user_answer'], reasoning_dict['user_answer_update']]
                # ===== case 4: input reasoning after user answer, output action =====
                else:
                    ref_interval_start_step, ref_interval_end_step = reasoning_dict['start_step'], idx - start_idx + 1
                    return_dict['thought'] = [reasoning_dict['user_answer_update_w_instruction']]

            # ========== user intervention segments ==========
            # user interrupts the robot
            elif 'user_intervention' in reasoning_dict.keys():
                # probabilities of the three cases
                content_prob, user_intervention_prob, user_intervention_update_w_instruction_prob = 0.3, 0.4, 0.3
                random_number = self.rdm.rand()
                # ===== case 1: input previous reasoning, output action =====
                if random_number < content_prob:
                    ref_interval_start_step, ref_interval_end_step = reference_start_step, reference_end_step
                    return_dict['thought'] = [reasoning_dict['content']]
                # ===== case 2: input user intervention, output reasoning after user intervention =====
                elif random_number < content_prob + user_intervention_prob:
                    ref_interval_start_step, ref_interval_end_step = reference_start_step, reference_end_step
                    return_dict['thought'] = [reasoning_dict['user_intervention'], reasoning_dict['user_intervention_update']]
                # ===== case 3: input reasoning after user intervention, output action =====
                else:
                    ref_interval_start_step, ref_interval_end_step = reasoning_dict['start_step'], idx - start_idx + 1
                    return_dict['thought'] = [reasoning_dict['user_intervention_update_w_instruction']]
            
            # ========== visual grounding robot data ==========
            # for visual grounding tasks, there are multiple reasoning contents
            # these reasoning contents are related to spatial, attribute, and semantic properties
            elif 'possible_contents' in reasoning_dict.keys():
                # sample one reasoning content
                possible_content_idx = self.rdm.randint(0, len(reasoning_dict['possible_contents']))
                chosen_content = reasoning_dict['possible_contents'][possible_content_idx]
                # ========== first segment ==========
                # where the robot reasons to identify the object to move to
                if chosen_content['updated_content'] is not None:
                    # we only learn action from visual grounding robot data
                    # otherwise the model can only output reasoning content in the robot data
                    # and ignore visuon-language data
                    pred_thought_prob = 0.
                    # case 1: ====== input previous reasoning, output updated reasoning ======
                    if self.rdm.rand() < pred_thought_prob:
                        # To train the model to reason
                        # the reference image is the current image
                        # as the first reasoning should happen at the first step of the episode
                        ref_interval_start_step, ref_interval_end_step = idx - start_idx, idx - start_idx + 1
                        return_dict['thought'] = [chosen_content['content'], chosen_content['updated_content']]
                    # case 2: ====== input updated reasoning, output action ======
                    else:
                        # To train the model to act
                        # the reference image is from step 0 (the start step of the first segment)
                        # to the current step
                        ref_interval_start_step, ref_interval_end_step = reference_start_step, idx - start_idx + 1
                        return_dict['thought'] = [chosen_content['updated_content_w_instruction']]
                    # we do not supervise the (previous reasoning -> action) case
                    # as it is the first segment, previous reasoning is an 'empty' reasoning
                # ========= second segment ==========
                # where the robot acts to move to the object
                else:
                    ref_interval_start_step, ref_interval_end_step = reference_start_step, reference_end_step
                    return_dict['thought'] = [chosen_content['content']]

            # ========== normal reasoning segments ==========
            elif reasoning_dict['updated_content'] is not None:
                reasoning_end_step = reasoning_dict['end_step'] if reasoning_dict['end_step'] != -1 else end_idx - start_idx
                # prob to use previous thought
                prev_reasoning_prob = self.get_prob(reasoning_dict['start_step'], reasoning_end_step, idx - start_idx)
                if self.rdm.rand() < prev_reasoning_prob:
                    # case 1: ====== input previous reasoning, output updated reasoning ======
                    # case 1.1: the first segment
                    if (idx - start_idx) < episode_start_interval[1]:
                        # reference image is the current image
                        ref_interval_start_step, ref_interval_end_step = idx - start_idx, idx - start_idx + 1
                        return_dict['thought'] = [reasoning_dict['content'], reasoning_dict['updated_content']]
                    # case 1.2: the non-first segment
                    elif self.rdm.rand() < self.pred_reasoning_prob:
                        # reference image can be sampled from the interval where the last reasoning occurred
                        ref_interval_start_step, ref_interval_end_step = reference_start_step, reference_end_step
                        return_dict['thought'] = [reasoning_dict['content'], reasoning_dict['updated_content']]
                    # case 2: ====== input previous reasoning, output action ======
                    # the robot should be able to act even the reasoning is outdated
                    else:
                        ref_interval_start_step, ref_interval_end_step = reference_start_step, reference_end_step
                        return_dict['thought'] = [reasoning_dict['content']]
                        # the action is with outdated thought
                        # we should not supervise the model to predict <BEGIN_OF_ACTION> token
                        return_dict['act_with_outdated_thought'] = True
                else:
                    # case 3: ====== input updated reasoning, output action ======
                    ref_interval_start_step, ref_interval_end_step = reasoning_dict['start_step'], idx - start_idx + 1
                    return_dict['thought'] = [reasoning_dict['updated_content_w_instruction']]
                    # for the last segment, the robot should stop moving to indicate finish
                    if reasoning_dict['end_step'] == -1:
                        freeze_action = True
            # ========== acting segments ==========
            else:
                # ========== acting segments with outdated reasoning ==========
                if 'outdated_content' in reasoning_dict.keys() and self.use_outdated_reasoning:
                    outdate_prob = self.get_prob(reasoning_dict['start_step'], reasoning_dict['end_step'], idx - start_idx, 0.4, 0.0)
                    # case 1: ====== input outdated reasoning, output <BEGIN_OF_REASONING> ======
                    if self.rdm.rand() < outdate_prob:
                        ref_interval_start_step, ref_interval_end_step = reasoning_dict['outdated_reference_start_step'], reasoning_dict['outdated_reference_end_step']
                        return_dict['thought'] = [reasoning_dict['outdated_content'], reasoning_dict['content']]
                        # we only supervise the <BEGIN_OF_REASONING> token
                        return_dict['think_with_outdated_thought'] = True
                    # case 2: ====== input latest reasoning, output action ======
                    else:
                        ref_interval_start_step, ref_interval_end_step = reference_start_step, reference_end_step
                        return_dict['thought'] = [reasoning_dict['content']]
                # ========== normal acting segments ==========
                else:
                    ref_interval_start_step, ref_interval_end_step = reference_start_step, reference_end_step
                    return_dict['thought'] = [reasoning_dict['content']]
            
            # ========== get reference image ==========
            if self.use_reference_image:
                reference_idx = self.rdm.randint(start_idx + ref_interval_start_step, start_idx + ref_interval_end_step)
                return_dict['reference_image'] = self.hf_dataset[reference_idx]['image']
        
        # for pi0, use_reasoning is False
        # for visual grounding tasks, we set prompt_from_task to False
        elif self.data_config.reasoning_json_path is not None:
            # pi0's instructions come from the 'content' of the 0-th segment
            # i.e., 'Instruction: ...'
            episode_id = current_idx_item['episode_index'].item()
            if 'possible_contents' in self.reasoning[episode_id]['segments'][0].keys():
                possible_contents = self.reasoning[episode_id]['segments'][0]['possible_contents']
                content_idx = self.rdm.randint(0, len(possible_contents))
                return_dict['prompt'] = possible_contents[content_idx]['content'].strip()
            else:
                return_dict['prompt'] = self.reasoning[episode_id]['segments'][0]['content'].strip()

        # get image and image history
        image_target_idx = np.array([idx] + [idx - self.image_down_sample_steps[history_idx] for history_idx in range(self.image_hisory_length - 1)])
        image_target_idx = np.clip(image_target_idx[::-1], start_idx, end_idx - 1)
        for i in range(self.image_hisory_length):
            return_dict['image_{}'.format(i + 1)] = self.hf_dataset[int(image_target_idx[i])]['image']
        
        # get state features and state history
        state_target_idx = np.array([idx] + [idx - self.state_down_sample_steps[history_idx] for history_idx in range(self.state_hisory_length - 1)])
        if self.use_reference_image and self.use_reasoning:
            # we randomly drop the history state
            # as the robot stays still while reasoning when rolling out
            # but this does not happen in the training data
            drop_history_prob = 0.2
            if self.rdm.rand() < drop_history_prob:
                _dummy_start_idx = idx
            else:
                _dummy_start_idx = reference_idx

            _dummy_start_idx = min(_dummy_start_idx, end_idx - 5)

            state_target_idx = np.clip(state_target_idx[::-1], _dummy_start_idx, end_idx - 1)
            interpolation_start = max(int(state_target_idx[0]) - 5, _dummy_start_idx)
        else:
            state_target_idx = np.clip(state_target_idx[::-1], start_idx, end_idx - 1)
            interpolation_start = max(int(state_target_idx[0]) - 5, start_idx)
        interpolation_end = min(int(state_target_idx[-1]) + 2 + 5, end_idx)
        for key in self.low_dim_keys:
            input_arr = self.low_dim_features[key]
            if 'rot' in key:
                slerp = st.Slerp(
                    times=np.arange(interpolation_start, interpolation_end),
                    rotations=st.Rotation.from_rotvec(input_arr[interpolation_start: interpolation_end]))
                output = st.Rotation.as_rotvec(slerp(state_target_idx))
            else:
                interp = si.interp1d(
                    x=np.arange(interpolation_start, interpolation_end),
                    y=input_arr[interpolation_start: interpolation_end],
                    axis=0, assume_sorted=True)
                output = interp(state_target_idx)
            low_dim_dict[key] = output

        # get action chunk
        slice_end = min(end_idx, idx + (self.action_horizon - 1) * self.action_down_sample_steps + 1)
        actions = self.actions[idx: slice_end: self.action_down_sample_steps]
        action_is_pad = torch.tensor([False] * actions.shape[0] + [True] * (self.action_horizon - actions.shape[0]))
        return_dict['action_is_pad'] = action_is_pad
        padding = np.repeat(actions[-1:], self.action_horizon - actions.shape[0], axis=0)
        actions = np.concatenate([actions, padding], axis=0)

        # calculate relative pose to start pose
        pose_mat = pose_to_mat(np.concatenate([low_dim_dict['eef_pos'], low_dim_dict['eef_rot_axis_angle']], axis=-1))
        start_pose = low_dim_dict['demo_start_pose'][0]
        start_pose += np.random.normal(scale=[0.05,0.05,0.05,0.05,0.05,0.05],size=start_pose.shape)
        start_pose_mat = pose_to_mat(start_pose)
        rel_obs_pose_mat = convert_pose_mat_rep(
            pose_mat,
            base_pose_mat=start_pose_mat,
            pose_rep='relative',
            backward=False)
        rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
        low_dim_dict['eef_rot_axis_angle_wrt_start'] = rel_obs_pose[:,3:]
        del low_dim_dict['demo_start_pose']

        # calculate relative pose to previous pose for obs and action
        action_mat = pose_to_mat(actions[..., :6])
        obs_pose_mat = convert_pose_mat_rep(
            pose_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep='relative',
            backward=False)
        action_pose_mat = convert_pose_mat_rep(
            action_mat, 
            base_pose_mat=pose_mat[-1],
            pose_rep='relative',
            backward=False)

        obs_pose = mat_to_pose10d(obs_pose_mat)
        action_pose = mat_to_pose10d(action_pose_mat)    
        action_gripper = actions[..., 6:7]
        final_action = np.concatenate([action_pose, action_gripper], axis=-1)
        if freeze_action:
            # repeat final_action[0] for the rest of the action horizon
            return_dict['actions'] = torch.from_numpy(np.repeat(final_action[:1], self.action_horizon, axis=0).astype(np.float32))
        else:
            return_dict['actions'] = torch.from_numpy(final_action.astype(np.float32))

        low_dim_dict['eef_pos'] = obs_pose[:,:3]
        low_dim_dict['eef_rot_axis_angle'] = obs_pose[:,3:]

        # concat low dim features to get state
        if self.getitem_type == 'default':
            key_sequence = ['eef_pos', 'eef_rot_axis_angle', 'eef_rot_axis_angle_wrt_start', 'gripper_width']
            return_dict['state'] = torch.from_numpy(np.concatenate([low_dim_dict[key].flatten() for key in key_sequence], axis=-1).astype(np.float32))
        elif self.getitem_type == 'necessary':
            return_dict['state'] = torch.from_numpy(np.concatenate([low_dim_dict['eef_pos'][:-1].flatten(), low_dim_dict['eef_rot_axis_angle'][:-1].flatten(), \
                                                                    low_dim_dict['eef_rot_axis_angle_wrt_start'][-1], low_dim_dict['gripper_width'].flatten()], axis=-1).astype(np.float32))
        elif self.getitem_type == 'shortest':
            history_rel_pose = mat_to_pose(obs_pose_mat)[:-1]
            history_rel_pose = np.concatenate([history_rel_pose, low_dim_dict['gripper_width'][:-1]], axis=-1)
            current_start_rel_pose = mat_to_pose(rel_obs_pose_mat)[-1]
            return_dict['state'] = torch.from_numpy(np.concatenate([history_rel_pose.flatten(), current_start_rel_pose, low_dim_dict['gripper_width'][-1]], axis=-1).astype(np.float32))
        else:
            raise ValueError('getitem_type should be one of default, necessary, shortest')
        
        copy_key = ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
        for key in copy_key:
            return_dict[key] = current_idx_item[key]
        return return_dict
