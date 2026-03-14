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
"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Protocol, TypeAlias
import copy
import os
import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0 as pi0
import openpi.models.pi0_fuse as pi0_fuse
import openpi.models.pi0_fast as pi0_fast
import openpi.models.tokenizer as _tokenizer
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.policies.libero_policy_pi as libero_policy_pi

import openpi.policies.umi_policy as umi_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms


ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # If true, will disable syncing the dataset from the Hugging Face Hub. Allows training on local-only datasets.
    local_files_only: bool = False

    def create_val_config(self) -> 'DataConfig':
        """
        Create a config for the validation dataset.
        The validation dataset may have different transforms than the training dataset.

        E.g.: postfix (actions) should not appear in the input for the validation dataset.
        """
        new_inputs = copy.deepcopy(self.model_transforms.inputs)
        for i in range(len(new_inputs)):
            if isinstance(new_inputs[i], _transforms.TokenizeFASTInputs):
                new_inputs[i] = dataclasses.replace(new_inputs[i], validation=True)
        new_model_transforms = dataclasses.replace(self.model_transforms, inputs=new_inputs)
        return dataclasses.replace(self, model_transforms=new_model_transforms)

@dataclasses.dataclass(frozen=True)
class UMIDataConfig(DataConfig):
    state_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    image_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    action_down_sample_steps: int = 3
    getitem_type: str = "default"
    use_val_dataset: bool = False
    val_ratio: float = 0.05
    # used to *create* train/val split
    # set to False when training
    create_train_val_split: bool = False
    norm_stats_dir: str | None = None
    seed: int = 42
    use_reasoning: bool = False
    use_reference_image: bool = True
    is_computing_norm_stats: bool = False
    reasoning_json_path: str | None = None
    use_outdated_reasoning: bool = True
@dataclasses.dataclass(frozen=True)
class LiberoReasonDataConfig(DataConfig):
    state_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    image_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    action_down_sample_steps: int = 1
    getitem_type: str = "default"
    use_val_dataset: bool = False
    val_ratio: float = 0.05
    # used to *create* train/val split
    # set to False when training
    create_train_val_split: bool = False
    norm_stats_dir: str | None = None
    seed: int = 42
    use_reasoning: bool = False
    use_reference_image: bool = True
    use_wrist_image: bool = False
    use_history: bool = True
    is_computing_norm_stats: bool = False
    reasoning_json_path: str | None = None
    use_outdated_reasoning: bool = True
class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            _tokenizer.FASTTokenizer(model_config.max_token_len),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )
            case _model.ModelType.PI0_FUSE:
                if model_config.eval_type == 'default':
                    print('use default policy transform')
                    return _transforms.Group(
                        inputs=[
                            _transforms.InjectDefaultPrompt(self.default_prompt),
                            _transforms.ResizeImages(224, 224),
                            _transforms.FuseTokenizePrompt(
                                _tokenizer.FusePaligemmaTokenizer(model_config.max_token_len),
                            ),
                        ],
                        outputs=[
                            _transforms.ExtractThoughts(
                                _tokenizer.FusePaligemmaTokenizer(model_config.max_token_len),
                            ),
                        ]
                    )
                elif model_config.eval_type == 'parallel':                
            ## TODO: add config to switch between parallel and sequential policy
            ## during inference time, we need to switch to the batch version 
                    print('use parallel policy transform')
                    # case _model.ModelType.PI0_FUSE:
                    return _transforms.Group(
                        inputs=[
                            _transforms.InjectDefaultPrompt(self.default_prompt),
                            _transforms.ResizeImages(224, 224),
                            _transforms.FuseTokenizePromptBatch(
                                _tokenizer.FusePaligemmaTokenizer(model_config.max_token_len),
                            ),
                        ],
                        outputs=[
                            _transforms.ExtractThoughtsBatch(
                                _tokenizer.FusePaligemmaTokenizer(model_config.max_token_len),
                            ),
                        ]
                    )

@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class SimpleDataConfig(DataConfigFactory):
    # Factory for the data transforms.
    data_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=GroupFactory)
    # Factory for the model transforms.
    model_transforms: tyro.conf.Suppress[GroupFactory] = dataclasses.field(default_factory=ModelTransformFactory)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=self.data_transforms(model_config),
            model_transforms=self.model_transforms(model_config),
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotAlohaDataConfig(DataConfigFactory):
    # If true, will convert joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions: bool = True
    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # If true, this will convert the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model. People who
    # use standard Aloha data should set this to true.
    adapt_to_pi: bool = True

    # Repack transforms.
    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {"cam_high": "observation.images.top"},
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action",)

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[aloha_policy.AlohaInputs(action_dim=model_config.action_dim, adapt_to_pi=self.adapt_to_pi)],
            outputs=[aloha_policy.AlohaOutputs(adapt_to_pi=self.adapt_to_pi)],
        )
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class LeRobotLiberoDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/wrist_image": "wrist_image",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[libero_policy_pi.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy_pi.LiberoOutputs()],
        )
        # # Use delta actions (not for gripper)
        # delta_action_mask = _transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[_transforms.DeltaActions(delta_action_mask)],
        #     outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        # )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
@dataclasses.dataclass(frozen=True)
class LeRobotLiberoReasonDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Make inputs look like they come from the Libero environment

        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[libero_policy.LiberoInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[libero_policy.LiberoOutputs()],
        )
 

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
          
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
    
    def _get_norm_stats_dir(self, assets_dir: epath.Path, asset_id: str | None) -> str:
        if asset_id is None :
            return None
        return str(assets_dir / asset_id )

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or LiberoReasonDataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            norm_stats_dir = self._get_norm_stats_dir(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotUMIDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:


        # Prepare data for policy training
        # Convert images to uint8 numpy arrays, add masks
        data_transforms = _transforms.Group(
            inputs=[umi_policy.UMIInputs(action_dim=model_config.action_dim, model_type=model_config.model_type)],
            outputs=[umi_policy.UMIOutputs()],
        )
        # Use delta actions (not for gripper)
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms include things like tokenizing the prompt and action targets
        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs),
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
    
    def _get_norm_stats_dir(self, assets_dir: epath.Path, asset_id: str | None, getitem_type: str | None) -> str:
        if asset_id is None or getitem_type is None:
            return None
        return str(assets_dir / asset_id / getitem_type)

    def create_base_config(self, assets_dirs: pathlib.Path) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or UMIDataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id, self.base_config.getitem_type),
            norm_stats_dir=self._get_norm_stats_dir(epath.Path(self.assets.assets_dir or assets_dirs), asset_id, self.base_config.getitem_type),
        )
    
    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None, getitem_type: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id / getitem_type)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"
    # directory for load checkpoint when eval 
    policy_dir: str | None = None
    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    val_batch_size: int = 12
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000
    # How often (in steps) to evaluate the model. Only used if use_val_dataset is True.
    val_interval: int = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If true, will use the validation dataset for training.
    use_val_dataset: bool = False
    val_ratio: float = 0.05
    # If true, will create a train/val split from the dataset. It's used only when compute norm stats.
    create_train_val_split: bool = False

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


@dataclasses.dataclass(frozen=True)
class UMITrainConfig(TrainConfig):
    repo_id: str = tyro.MISSING
    # umi related
    state_down_sample_steps: list[int] = tyro.MISSING
    image_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    action_down_sample_steps: int = tyro.MISSING

    # how to calculate the state
    getitem_type: str = tyro.MISSING
    # enable reasoning
    use_reasoning: bool = False
    # whether to use the reference image
    use_reference_image: bool = True
    # whether to use outdated reasoning
    use_outdated_reasoning: bool = True
    is_computing_norm_stats: bool = False
    data: DataConfigFactory = dataclasses.field(init=False)
    reasoning_json_path: str | None = None
    prompt_from_task: bool = True

    # If true, set the `decay_step` to the number of training steps.
    lr_decay_till_end: bool = True

    def __post_init__(self):
       
        super().__post_init__()
         ## process the reasoning_json_path to append the repo_id 
        if self.reasoning_json_path is not None:
            self.reasoning_json_path = os.path.join(self.reasoning_json_path, repo_id, "cot.json")
        object.__setattr__(self, 'data', LeRobotUMIDataConfig(
            repo_id=self.repo_id,
            base_config=UMIDataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=self.prompt_from_task,
                state_down_sample_steps=self.state_down_sample_steps,
                image_down_sample_steps=self.image_down_sample_steps,
                action_down_sample_steps=self.action_down_sample_steps,
                getitem_type=self.getitem_type,
                use_val_dataset=self.use_val_dataset,
                val_ratio=self.val_ratio,
                create_train_val_split=self.create_train_val_split,
                seed=self.seed,
                use_reasoning=self.use_reasoning,
                use_reference_image=self.use_reference_image,
                is_computing_norm_stats=self.is_computing_norm_stats,
                reasoning_json_path=self.reasoning_json_path,
                use_outdated_reasoning=self.use_outdated_reasoning,
            ),
        ))
        if self.lr_decay_till_end:
            assert isinstance(self.lr_schedule, _optimizer.CosineDecaySchedule), "Only CosineDecaySchedule is supported for lr_decay_till_end"
            object.__setattr__(self, 'lr_schedule', dataclasses.replace(self.lr_schedule, decay_steps=self.num_train_steps))


@dataclasses.dataclass(frozen=True)
class LiberoReasonTrainConfig(TrainConfig):
    repo_id: str = tyro.MISSING
    # umi related
    state_down_sample_steps: list[int] = tyro.MISSING
    image_down_sample_steps: list[int] = dataclasses.field(default_factory=list)
    action_down_sample_steps: int = tyro.MISSING

    # how to calculate the state
    getitem_type: str = tyro.MISSING
    # enable reasoning
    use_reasoning: bool = False
    # whether to use the reference image
    use_reference_image: bool = True
    # whether to use the wrist image
    use_wrist_image: bool = False
    # whether to use history state
    use_history: bool = True
    # whether to use outdated reasoning
    use_outdated_reasoning: bool = True
    is_computing_norm_stats: bool = False
    data: DataConfigFactory = dataclasses.field(init=False)
    reasoning_json_path: str | None = None
    prompt_from_task: bool = True

    # If true, set the `decay_step` to the number of training steps.
    lr_decay_till_end: bool = True

    def __post_init__(self):
        super().__post_init__()
        ## process the reasoning_json_path to append the repo_id 
        if self.reasoning_json_path is not None:
            object.__setattr__(self, 'reasoning_json_path',os.path.join(self.reasoning_json_path, self.repo_id, "cot_simple.json"))
        object.__setattr__(self, 'data', LeRobotLiberoReasonDataConfig(
            repo_id=self.repo_id,
            base_config=LiberoReasonDataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=self.prompt_from_task,
                state_down_sample_steps=self.state_down_sample_steps,
                image_down_sample_steps=self.image_down_sample_steps,
                action_down_sample_steps=self.action_down_sample_steps,
                getitem_type=self.getitem_type,
                use_val_dataset=self.use_val_dataset,
                val_ratio=self.val_ratio,
                create_train_val_split=self.create_train_val_split,
                seed=self.seed,
                use_wrist_image=self.use_wrist_image,
                use_history=self.use_history,
                use_reasoning=self.use_reasoning,
                use_reference_image=self.use_reference_image,
                is_computing_norm_stats=self.is_computing_norm_stats,
                reasoning_json_path=self.reasoning_json_path,
                use_outdated_reasoning=self.use_outdated_reasoning,
            ),
        ))
        if self.lr_decay_till_end:
            assert isinstance(self.lr_schedule, _optimizer.CosineDecaySchedule), "Only CosineDecaySchedule is supported for lr_decay_till_end"
            object.__setattr__(self, 'lr_schedule', dataclasses.replace(self.lr_schedule, decay_steps=self.num_train_steps))

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    #
    # Inference Aloha configs.
    #
    TrainConfig(
        name="pi0_aloha",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
        ),
    ),
    TrainConfig(
        name="pi0_aloha_towel",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="fold the towel",
        ),
    ),
    TrainConfig(
        name="pi0_aloha_tupperware",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            assets=AssetsConfig(asset_id="trossen"),
            default_prompt="open the tupperware and put the food on the plate",
        ),
    ),
    #
    # Inference DROID configs.
    #
    TrainConfig(
        name="pi0_droid",
        model=pi0.Pi0Config(action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    TrainConfig(
        name="pi0_fast_droid",
        model=pi0_fast.Pi0FASTConfig(action_dim=8, action_horizon=10),
        data=SimpleDataConfig(
            assets=AssetsConfig(asset_id="droid"),
            data_transforms=lambda model: _transforms.Group(
                inputs=[droid_policy.DroidInputs(action_dim=model.action_dim, model_type=ModelType.PI0_FAST)],
                outputs=[droid_policy.DroidOutputs()],
            ),
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
    ),
    #
    # Fine-tuning Libero configs.
    
    TrainConfig(
        name="pi0_libero",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="xxx/libero",
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_10",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="libero-10-r",
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_100",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="libero-100-r",
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
    TrainConfig(
        name="pi0_libero_100_basket",
        model=pi0.Pi0Config(),
        data=LeRobotLiberoDataConfig(
            repo_id="libero-100-basket-r",
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),


    TrainConfig(
        name="pi0_fast_libero",
        model=pi0_fast.Pi0FASTConfig(action_dim=7, action_horizon=10, max_token_len=180),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
    ),


    LiberoReasonTrainConfig(
        name="pi0_libero_reason_wrist_image_no_history",
        model=pi0_fuse.Pi0FuseConfig(action_horizon=16, max_token_len=415,),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        use_reasoning=True,
        use_wrist_image=True,
        use_history=False,
        repo_id="libero-10-r",
        state_down_sample_steps=[],
        action_down_sample_steps=1,
        getitem_type="necessary",
        use_val_dataset=True,
        use_outdated_reasoning=True,
        prompt_from_task=True,
    ),

    LiberoReasonTrainConfig(
        name="pi0_libero_100_basket_reason_wrist_image_no_history",
        model=pi0_fuse.Pi0FuseConfig(action_horizon=16, max_token_len=415,),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        use_reasoning=True,
        use_wrist_image=True,
        use_history=False,
        repo_id="libero-100-basket-r",
        state_down_sample_steps=[],
        action_down_sample_steps=1,
        getitem_type="necessary",
        use_val_dataset=True,
        use_outdated_reasoning=True,
        prompt_from_task=True,
    ),
    
    LiberoReasonTrainConfig(
        name="pi0_libero_100_reason_wrist_image_no_history",
        model=pi0_fuse.Pi0FuseConfig(action_horizon=16, max_token_len=415,),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        use_reasoning=True,
        use_wrist_image=True,
        use_history=False,
        repo_id="libero-100-r",
        state_down_sample_steps=[],
        action_down_sample_steps=1,
        getitem_type="necessary",
        use_val_dataset=True,
        use_outdated_reasoning=True,
        prompt_from_task=True,
    ),

    LiberoReasonTrainConfig(
        name="pi0_libero_reason_wrist_image_no_history_parallel",
        model=pi0_fuse.Pi0FuseConfig(action_horizon=16, max_token_len=415, eval_type="parallel"),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        use_reasoning=True,
        use_wrist_image=True,
        use_history=False,
        repo_id="libero-10-r",
        state_down_sample_steps=[],
        action_down_sample_steps=1,
        getitem_type="necessary",
        use_val_dataset=True,
        use_outdated_reasoning=True,
        prompt_from_task=True,
    ),
    
    LiberoReasonTrainConfig(
        name="pi0_libero_100_basket_reason_wrist_image_no_history_parallel",
        model=pi0_fuse.Pi0FuseConfig(action_horizon=16, max_token_len=415, eval_type="parallel"),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        use_reasoning=True,
        use_wrist_image=True,
        use_history=False,
        repo_id="libero-100-basket-r",
        state_down_sample_steps=[],
        action_down_sample_steps=1,
        getitem_type="necessary",
        use_val_dataset=True,
        use_outdated_reasoning=True,
        prompt_from_task=True,
    ),
        LiberoReasonTrainConfig(
        name="pi0_libero_100_reason_wrist_image_no_history_parallel",
        model=pi0_fuse.Pi0FuseConfig(action_horizon=16, max_token_len=415, eval_type="parallel"),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        use_reasoning=True,
        use_wrist_image=True,
        use_history=False,
        repo_id="libero-100-r",
        state_down_sample_steps=[],
        action_down_sample_steps=1,
        getitem_type="necessary",
        use_val_dataset=True,
        use_outdated_reasoning=True,
        prompt_from_task=True,
    ),




    UMITrainConfig(
        name="onetwovla_cocktail",
        model=pi0_fuse.Pi0FuseConfig(action_horizon=16, max_token_len=410),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        use_reasoning=True,
        repo_id="umi/cocktail",
        state_down_sample_steps=[3, 15],
        action_down_sample_steps=3,
        getitem_type="necessary",
        use_val_dataset=True,
        use_outdated_reasoning=True,
    ),
    UMITrainConfig(
        name="pi0_visual_grounding",
        model=pi0.Pi0Config(action_horizon=16, max_token_len=40),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        use_reasoning=False,
        repo_id="umi/wild_move_to_no_vl",
        state_down_sample_steps=[3, 15],
        action_down_sample_steps=3,
        getitem_type="necessary",
        use_val_dataset=True,
        prompt_from_task=False,
    ),
    UMITrainConfig(
        name="onetwovla_visual_grounding",
        model=pi0_fuse.Pi0FuseConfig(action_horizon=16, max_token_len=100),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        use_reasoning=True,
        repo_id="umi/wild_move_to",
        state_down_sample_steps=[3, 15],
        action_down_sample_steps=3,
        getitem_type="necessary",
        use_val_dataset=True,
        use_reference_image=False,
        use_outdated_reasoning=True,
    ),
    TrainConfig(
        name="pi0_fast_libero_low_mem_finetune",
        model=pi0_fast.Pi0FASTConfig(paligemma_variant="gemma_2b_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="physical-intelligence/libero",
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_fast_base/params"),
        num_train_steps=30_000,
        freeze_filter=pi0_fast.Pi0FASTConfig(
            action_dim=7, action_horizon=10, max_token_len=180, paligemma_variant="gemma_2b_lora"
        ).get_freeze_filter(),
        ema_decay=None,
    ),
    #
    # Fine-tuning Aloha configs.
    #
    # This is a test config that is used to illustate how train on a custom LeRobot dataset.
    # For instuctions on how to convert and train on your own Aloha dataset see examples/aloha_real/README.md
    TrainConfig(
        name="pi0_aloha_pen_uncap",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="physical-intelligence/aloha_pen_uncap_diverse",
            assets=AssetsConfig(
                assets_dir="s3://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="trossen",
            ),
            default_prompt="uncap the pen",
            repack_transforms=_transforms.Group(
                inputs=[
                    _transforms.RepackTransform(
                        {
                            "images": {
                                "cam_high": "observation.images.cam_high",
                                "cam_left_wrist": "observation.images.cam_left_wrist",
                                "cam_right_wrist": "observation.images.cam_right_wrist",
                            },
                            "state": "observation.state",
                            "actions": "action",
                        }
                    )
                ]
            ),
            base_config=DataConfig(
                local_files_only=False,  # Set to True for local-only datasets.
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    # This config is used to demonstrate how to train on a simple simulated environment.
    TrainConfig(
        name="pi0_aloha_sim",
        model=pi0.Pi0Config(),
        data=LeRobotAlohaDataConfig(
            repo_id="lerobot/aloha_sim_transfer_cube_human",
            default_prompt="Transfer cube",
            use_delta_joint_actions=False,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=20_000,
    ),
    #
    # Debugging configs.
    #
    TrainConfig(
        name="debug",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        save_interval=100,
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
    TrainConfig(
        name="debug_restore",
        data=FakeDataConfig(),
        batch_size=2,
        model=pi0.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"),
        weight_loader=weight_loaders.CheckpointWeightLoader("./checkpoints/debug/debug/9/params"),
        overwrite=True,
        exp_name="debug",
        num_train_steps=10,
        wandb_enabled=False,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
