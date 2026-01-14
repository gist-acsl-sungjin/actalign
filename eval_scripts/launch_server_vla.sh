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
num_samples=1
checkpoint_iter=219999
temperature=1.0
use_vgps_critic=false
while getopts ":p:n:i:t:u:m:" opt; do
  case $opt in
    p)
      port="$OPTARG"
      ;;
    n)
      num_samples="$OPTARG"
      ;;
    i)
      checkpoint_iter="$OPTARG"
      ;;
    t)
      temperature="$OPTARG"
      ;;
    u)
      use_vgps_critic="$OPTARG"
      ;;
    m)
      model_type="$OPTARG"
      ;;
    \?)
      echo "Invalid option -$OPTARG" >&2
      echo "Usage: $0 [-p port] [-n num_samples] [-i checkpoint_iter] [-t temperature] [-u use_vgps_critic] [-m model_type]">&2
      exit 1
      ;;
  esac
done

## if use_vgps_critic is true, then use the vgps-based critic
if [ "$use_vgps_critic" = true ]; then
  use_vgps_critic="--use-vgps-critic"
  eval_type="parallel"
else
  use_vgps_critic=""
  eval_type="default"
fi
if [ "$model_type" = "libero-100" ]; then
  config="pi0_libero_100"
  ckpt_dir="checkpoints/pi0_libero_100"
elif [ "$model_type" = "libero-10" ]; then
  config="pi0_libero_10"
  ckpt_dir="checkpoints/pi0_libero_10"
elif [ "$model_type" = "libero-100-basket" ]; then
  config="pi0_libero_100_basket"
  ckpt_dir="checkpoints/pi0_libero_100_basket"
fi
python scripts/serve_policy.py \
 --num-samples $num_samples --port $port --checkpoint-step $checkpoint_iter \
 --action-temp $temperature  $use_vgps_critic --eval-type $eval_type policy:checkpoint \
 --policy.config=$config --policy.dir=$ckpt_dir
