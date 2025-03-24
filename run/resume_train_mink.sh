set -x

exp_dir=$1
config=$2
model_dir=${exp_dir}/model

mkdir -p ${exp_dir}

export PYTHONPATH=.
python -u run/train_mink.py \
  --config=${config} \
  save_path ${exp_dir} \
  resume ${model_dir}/model_last.pth.tar \
  2>&1 | tee -a ${exp_dir}/train_mink-$(date +"%Y%m%d_%H%M").log