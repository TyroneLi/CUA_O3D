#!/bin/sh
set -x

exp_dir=$1
config=$2
model_dir=$3

mkdir -p ${exp_dir}

export PYTHONPATH=.
python -u run/baseline_prob_seg.py \
  --config=${config} \
  save_path ${exp_dir} \
  init ${model_dir} \
  2>&1 | tee -a ${exp_dir}/distill-$(date +"%Y%m%d_%H%M").log

