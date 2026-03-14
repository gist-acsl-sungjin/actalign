#!/bin/bash
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
# shellcheck disable=SC1091
source "config_scripts/config.sh"
while getopts ":b:v:r:o:c:" opt; do
  case $opt in
    b)
      batch_size="$OPTARG"
      ;;
    v)
      val_batch_size="$OPTARG"
      ;;
    r)
      resume="$OPTARG"
      ;;
    o)
      overwrite="$OPTARG"
      ;;
    c)
      config="$OPTARG"
      ;;
    \?)
      echo "Invalid option -$OPTARG" >&2
      echo "Usage: $0 [-b batch_size] [-v val_batch_size] [-r resume] [-o overwrite]">&2
      exit 1
      ;;
  esac
done

# the json file can be downloaded from https://huggingface.co/datasets/Richard-Nai/onetwovla-dataset/tree/main/cocktail
# ensure the dataset's path is $LEROBOT_HOME/umi/cocktail


if [ "$resume" = true ]; then
  resume="--resume"
else
  resume=""
fi
if [ "$overwrite" = true ]; then
  overwrite="--overwrite"
else
  overwrite=""
fi

if [ "$config" = "pi0_libero_100" ]; then
  config="pi0_libero_100"
  reasoning_json_path=""
  batch_size=160
  val_batch_size=96
elif [ "$config" = "pi0_libero_10" ]; then
  config="pi0_libero_10"
  reasoning_json_path=""
  batch_size=32 # base: 160
  val_batch_size=16 # base: 96
elif [ "$config" = "pi0_libero_100_basket" ]; then
  config="pi0_libero_100_basket"
  reasoning_json_path=""
  batch_size=160
  val_batch_size=96
elif [ "$config" = "pi0_libero_100_reason" ]; then
  config="pi0_libero_100_reason_wrist_image_no_history"
  reasoning_json_path="--reasoning_json_path $PROJECT_FOLDER/../lerobot"
  batch_size=128
  val_batch_size=64
elif [ "$config" = "pi0_libero_10_reason" ]; then
  config="pi0_libero_reason_wrist_image_no_history"
  reasoning_json_path="--reasoning_json_path $PROJECT_FOLDER/../lerobot"
  batch_size=128
  val_batch_size=64
elif [ "$config" = "pi0_libero_100_basket_reason" ]; then
  config="pi0_libero_100_basket_reason_wrist_image_no_history"
  reasoning_json_path="--reasoning_json_path $PROJECT_FOLDER/../lerobot"
  batch_size=128
  val_batch_size=64
fi


# reasoning_json_path="$PROJECT_FOLDER/../lerobot"

# base train script
# GIT_LFS_SKIP_SMUDGE=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py $config \
# --exp-name=$config --batch-size=$batch_size  --val-batch-size=$val_batch_size \
# $reasoning_json_path $resume $overwrite

# changed script
export XLA_FLAGS="--xla_gpu_enable_triton_softmax_fusion=true --xla_gpu_graph_level=0"
export JAXTYPING_DTYPE_CHECK=off
export JAX_TRACEBACK_FILTERING=off
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export GIT_LFS_SKIP_SMUDGE=1

uv run scripts/train.py $config \
    --exp-name="${config}_a30_8gpu" \
    --batch-size=$batch_size \
    --val-batch-size=$val_batch_size \
    --fsdp-devices=8 \
    --overwrite $resume
