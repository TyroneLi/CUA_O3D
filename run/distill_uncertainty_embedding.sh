#!/bin/sh
set -x

exp_dir=$1
config=$2
model_dir=${exp_dir}/model

mkdir -p ${exp_dir}

export PYTHONPATH=.
python -u run/distill_uncertainty_embedding.py \
  --config=${config} \
  save_path ${exp_dir} \
  resume ${model_dir}/model_last.pth.tar \
  2>&1 | tee -a ${exp_dir}/distill-$(date +"%Y%m%d_%H%M").log