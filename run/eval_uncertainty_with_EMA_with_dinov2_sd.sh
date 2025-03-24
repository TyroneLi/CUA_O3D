#!/bin/sh
set -x

exp_dir=$1
config=$2
feature_type=$3

mkdir -p ${exp_dir}
result_dir=${exp_dir}/result_eval
model_path=${exp_dir}/model/ema_model_best.pth.tar
# model_path=${exp_dir}/model/model_best.pth.tar
# model_path=${exp_dir}/model/model_best.pth.tar

export PYTHONPATH=.
python -u run/evaluate_uncertainty_with_EMA_with_dinov2_sd.py \
  --config=${config} \
  feature_type ${feature_type} \
  save_folder ${result_dir} \
  model_path ${model_path} \
  2>&1 | tee -a ${exp_dir}/eval-$(date +"%Y%m%d_%H%M").log
  