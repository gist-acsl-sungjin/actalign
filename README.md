# Do What You Say: Steering Vision-Language-Action Models via Runtime Reasoning-Action Alignment Verification

<p align="center">
  <a href="https://yilin-wu98.github.io/steering-reasoning-vla/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://www.arxiv.org/abs/2510.16281"><img src="https://img.shields.io/badge/arXiv-2510.16281-b31b1b" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/nvidia/libero_r_dataset"><img src="https://img.shields.io/badge/🤗-Datasets-yellow" alt="Datasets"></a>
  <a href="https://github.com/NVlabs/Libero-10-r"><img src="https://img.shields.io/badge/Reasoning-Benchmark-green" alt="Reasoning Benchmark"></a>
  <img src="https://img.shields.io/badge/🤗-Models_(TBA)-c9a227" alt="Models TBA">
</p>

<p align="center">
<a href="https://yilin-wu98.github.io/">Yilin Wu</a><sup>2*</sup>,
<a href="https://anqili.github.io/">Anqi Li</a><sup>1</sup>,
<a href="https://robot-learning.cs.utah.edu/thermans">Tucker Hermans</a><sup>1</sup>,
<a href="https://scholar.google.com/citations?user=T_mJiHoAAAAJ&hl=en">Fabio Ramos</a><sup>1</sup>,
<a href="https://www.cs.cmu.edu/~abajcsy/">Andrea Bajcsy</a><sup>2†</sup>,
<a href="https://scholar.google.com/citations?user=q0kUpsoAAAAJ&hl=en">Claudia D'Arpino</a><sup>1†</sup>
</p>

<p align="center">
<sub><sup>1</sup>NVIDIA &nbsp; <sup>2</sup>Carnegie Mellon University &nbsp;&nbsp; *Work done during an internship at NVIDIA &nbsp; †Equal Advising</sub>
</p>

This repository contains the code and instructions for training and benchmarking SEAL-VLA. 





## 🛠️ Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:NVlabs/actalign.git seal-vla

# Or if you already cloned the repo:
git submodule update --init --recursive
```


If you want to run inside a docker container, build the image first:
If you are running experiments on the cluster, you can build a docker image first:
```bash
DOCKER_BUILDKIT=1 docker build  --rm      --tag seal-vla:latest       --file examples/libero/Dockerfile   .
```

This image can be used for both training and evaluation. 
Then you run the following commands to add the dependency for the policy. You can also directly run the following commands for installation without docker container.

We manage Python dependencies with [uv](https://docs.astral.sh/uv/). If you haven't installed `uv`, please follow [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. 

Run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

> NOTE: `GIT_LFS_SKIP_SMUDGE=1` is needed to pull LeRobot as a dependency.

For more details, refer to the original [openpi repository](https://github.com/Physical-Intelligence/openpi.git). 



## 📝 Configuration 
Open the config_scripts/config.sh and specify some credentials to run the training and evaluaiton.

Fill in the following terms:

1. Replace `<PROJECT_FOLDER>` with the absolute path in your system to store the cloned repo. e.g. /home/$USER/seal-vla
2. Replace `<WANDB_API_KEY>` with the wandb api key to automatically upload the log to your wandb.
3. Replace `<HF_TOKEN>` with the huggingface token of your account. This is for downloading datasets and models.

In the config_scripts/config.sh, there are also some default paths for storing data. If you change the path to store the data, you can update config.sh to match your settings. 

Specifically, 

1.`$LEROBOT_HOME` is to store the datasets used for training. 

2.`$OPENPI_DATA_HOME` is to store the base pre-trained model from openpi. 

3.`$UV_CACHE_DIR` is to store the cache for uv when creating the environments. The default uv cache directory is /home/$USER/.cache but if your home directory storage is small, you need to specify a new directory for uv cache. 

4.`$HF_HOME` is for huggingface cache.

Add openpi token to run the VLM verification.

In `examples/libero/api_config.json` file, add the api key in the following format
```json
{
    "openai": {
        "api_key": "your api key"
    },
}
```
## 📷 Datasets \& Models

### 1.Download the dataset and place them under `$LEROBOT_HOME/`. Replace the `<repo id>` with any dataset listed in the table below.
```bash
# 1) Make sure Git LFS is installed
git lfs install

# 2) Clone the dataset
git clone https://huggingface.co/datasets/<repo id>

# 3) Pull the large files (LFS objects)
cd <name>
git lfs pull

```

The provided reasoning datasets are available at [nvidia/libero_r_dataset](https://huggingface.co/datasets/nvidia/libero_r_dataset):

| Path                  | Description                                                                                                 |
| --------------------- | ----------------------------------------------------------------------------------------------------------- |
| `libero-10-r/`        | LIBERO-10 with reasoning annotations.                                                                       |
| `libero-100-basket-r/`| LIBERO-10 and pick-and-place-into-basket tasks from LIBERO-90 with reasoning annotations.                   |
| `libero-100-r/`       | LIBERO-10 and LIBERO-90 with reasoning annotations.                                                         |

The reasoning traces are provided in cot_simple.json in each dataset folder. Therefore, to get the original demonstration-only dataset, you can just ignore the cot_simple.json file. The original demonstration-only datasets are preprocessed to remove no-ops, unsuccessful trajectories, and had its image observations flipped back upright according to the script OpenVLA authors provided.

### 2.Download the pre-trained models and place them under `$PROJECT_FOLDER/checkpoints/`. 

The pre-trained models will be released soon.

The pre-trained models are:

| Repo ID                                                     | Description                                                   |
| ----------------------------------------------------------- | ------------------------------------------------------------- |
| TBD/pi0_libero_10                                      | The pi0 model finetuned on libero-10 dataset.                 |
| TBD/pi0_libero_10_reason_wrist_image_no_history        | The pi0 model finetuned on libero-10 reasoning dataset.       |
| TBD/pi0_libero_100_basket                              | The pi0 model finetund on libero-100-basket dataset.          |
| TBD/pi0_libero_100_basket_reason_wrist_image_no_history| The pi0 model finetund on libero-100-basket reasoning dataset.|
| TBD/pi0_libero_100                                     | The pi0 model finetund on libero-100 dataset.                 |
| TBD/pi0_libero_100_reason_wrist_image_no_history       | The pi0 model finetuned on libero-100 reasoning dataset.      |
| TBD/vgps                                               | The q function trained with IQL on libero-100 dataset.        |


```bash
# 1) Make sure Git LFS is installed
git lfs install

# 2) Clone the dataset
git clone https://huggingface.co/<repo id>

# 3) Pull the large files (LFS objects)
cd <name>
git lfs pull

```
You can also train your models and the checkpoints will automatically stored in the `$PROJECT_FOLDER/checkpoints/` folder. If you want to evaluate your own model with provided evaluation scripts. Remmber to update the `launch_server_vla_reason.sh` file in eval_scripts folder to specify your checkpoint path. You should replace the `ckpt_dir` with your `$PROJECT_FOLDER/checkpoints/<config_name>/<exp_name>/<training_iters>` for the model you have trained. 

## 🚀 Training SEAL

To train the models in this repository, you will need an NVIDIA GPU with at least the following specifications. These estimations assume a single GPU, but you can also use multiple GPUs with model parallelism to reduce per-GPU memory requirements by configuring `fsdp_devices` in the training config. Please also note that the current training script does not yet support multi-node training.

| Mode               | Memory Required | Example GPU        |
| ------------------ | --------------- | ------------------ |
| Fine-Tuning (Full) | > 70 GB         | A100 (80GB) / H100 |


### Available config_names for pi0 or pi0-reason model with different datasets:

1.```pi0_libero_10```: finetune pi0 with demonstration-only dataset of LIBERO-10

2.```pi0_libero_100_basket```: finetune pi0 with demonstration-only dataset of LIBERO-10 and pick-place-into-basket tasks in LIBERO-90

3.```pi0_libero_100```: finetune pi0 with demonstration-only dataset of LIBERO-10 and LIBERO-90

4.```pi0_libero_10_reason```: finetune pi0 with reasoning-annotated dataset of LIBERO-10

5.```pi0_libero_100_basket_reason```: finetune pi0 with reasoning-annotated dataset of LIBERO-10 and pick-place-into-basket tasks in LIBERO-90

6.```pi0_libero_100_reason```: finetune pi0 with reasoning-annotated dataset of LIBERO-10 and LIBERO-90


### Before training a model, we need to compute the normalization statistics for the training data. Run the script below with the name of config:
```bash
bash train_scripts/compute_norm_stats.sh -m <config_name>
```

### To train a pi0 or pi0-reason model:

1) launch a new run or overwrite the previous runs for this configuration
```bash
bash train_scripts/train.sh -c <config_name> -o true
```

2) resume a previous run for this configuration
```bash
bash train_scripts/train.sh -c <config_name> -r true
```





## 🦾 Evaluation 

The first time to launch the evaluation, we need to create two venv environment to run the policy inference and the simulation for libero-environment. After the venv has been installed, we no longer need to go through this installation steps for the following evaluation.

### Policy Inference environment (Note that we modify the environment just to make the vgps checkpoint can be loaded with pi0 model)

Clone the v-gps repo
```bash 
cd $PROJECT_FOLDER/..
git clone https://github.com/nakamotoo/V-GPS.git v-gps

```
Install: 

If you are running evaulation on the cluster, you can build a docker image first:
```bash
DOCKER_BUILDKIT=1 docker build  --rm      --tag seal-vla:latest       --file examples/libero/Dockerfile   .
```

Once you are inside the container, you can run the following command once to build the virtual envs for policy and libero. 
After you have built the virtual env, the next time you can just launch docker image and run the evaluation commands below directly. 
```bash
source config_scripts/config.sh
uv venv --python 3.11 policy/.venv/
source policy/.venv/bin/activate
uv pip sync requirements_policy.txt
uv pip install -e packages/openpi-client
uv pip install -e ../v-gps
uv pip install -e .
uv pip install -e ../v-gps/octo

```

### LIBERO-Environment
```bash
source config_scripts/config.sh
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero
```
We provide the following scripts to run the different baselines as well as our method.

### Our Method

### 1. Run our method with test-time sampling and verification 
On the server side, run the following command to launch the policy server.

You may change the port `-p` to any port that is free.

You may change the number of samples `-n` to any integer number. 

You can change the port `-m` to models trained from different datasets. You can choose from (`libero-10`, `libero-100-basket`, `libero-100`)
```bash 
source eval_scripts/env.sh
bash eval_scripts/launch_server_vla_reason.sh -p 8000 -n 10 -m <model_name>
```
On the client side, run the following command to launch different evaluation by passing different values to the corresponding flags.

The port flag `-p` should match the port specified on the server side. 

The number of samples `-n` should match the server side. 

The semantic ood environments flag `-s`: (default 0 for no semantic ood; 1 for Semantic-OOD-Rephrase; 2 for Semantic-OOD-Object-Property). 

The visual ood environment flag `-v`: (deafult 0 for no visual ood; 1 for Visual-OOD-Viewpoint, 2 for Visual-OOD-Scene)

The novel behavior composition task flag `-b`: (default 0 for original 10 tasks in libero-10; 1 for 13 novel behavior composition tasks ):

```bash
bash eval_scripts/launch_libero_vla_reason_sample.sh -p 8000  -n 10
```


### Baselines
### 2. Run the base reasoning vla model built on top of OneTwoVLA

On the server side, run the following command to launch the policy server. 

You may change the port `-p` to any port that is free

You can change the port `-m` to models trained from different datasets. You can choose from (`libero-10`, `libero-100-basket`, `libero-100`)
```bash 
source eval_scripts/env.sh
bash eval_scripts/launch_server_vla_reason.sh -p 8000 -m <model_name>
```
On the client side, run the following command to launch different evaluation.

The meaning of different flags can be found in the previous section [Our Method](#our-method)
```bash
bash eval_scripts/launch_libero_vla_reason.sh -p 8000
```
### 3. Run the pi-0 with vgps to steer the VLA policy

On the server side, run the following command, change the number of samples with the flag `-n` and specify the port `-p`.

You can change the port `-m` to models trained from different datasets. You can choose from (`libero-10`, `libero-100-basket`, `libero-100`):
```bash
source eval_scripts/env.sh
bash eval_scripts/launch_server_vla.sh -p 8000 -u true -n 10 -m <model_name>
```

On the client side, run the following command to launch different evaluation.

The meaning of different flags can be found in the previous section [Our Method](#our-method)

```bash
bash eval_scripts/launch_libero_vla.sh -p 8000 
```

### 4. Run the pi-0 without test-time steering
On the server side, run the following command and specify the port `-p`.
You can change the port `-m` to models trained from different datasets. You can choose from (`libero-10`, `libero-100-basket`, `libero-100`):
```bash
source eval_scripts/env.sh
bash eval_scripts/launch_server_vla.sh -p 8000 -m <model_name>
```

On the client side, run the following command to launch different evaluation.

The meaning of different flags can be found in the previous section [Our Method](#our-method)

```bash
bash launch_libero_vla.sh -p 8000
```

## Acknowledgements

This repository builds upon the [codebase](https://github.com/Fanqi-Lin/OneTwoVLA) from 
[OneTwoVLA: A Unified Vision-Language-Action Model with Adaptive Reasoning](https://arxiv.org/abs/2505.11917),
which builds upon the codebase of [Openpi](https://github.com/Physical-Intelligence/openpi). 

## License 

See License files. 

## Citation

If you found this work useful, please cite the following paper:

```bibtex
@article{wu2025saysteeringvisionlanguageactionmodels,
      title={Do What You Say: Steering Vision-Language-Action Models via Runtime Reasoning-Action Alignment Verification}, 
      author={Yilin Wu and Anqi Li and Tucker Hermans and Fabio Ramos and Andrea Bajcsy and Claudia P\'{e}rez-D'Arpino},
      year={2025},
      eprint={2510.16281},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2510.16281}, 
}
```