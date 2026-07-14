#!/bin/bash
#SBATCH -J unetpp_off
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/unetpp_official_%j.out

set -e

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

mkdir -p runs/logs

echo "===== UNET++ OFFICIAL DEEP SUPERVISION ====="
which python
python --version
nvidia-smi

python -m py_compile \
  src/official_baseline_common.py \
  src/train_unetpp_official.py

python src/train_unetpp_official.py \
  --run-name unetpp_official_ds_e200_bs4 \
  --image-size 512 \
  --batch-size 4 \
  --val-batch-size 4 \
  --epochs 200 \
  --lr 3e-4 \
  --early-stopping-patience 30 \
  --num-workers 2 \
  --seed 42

cat runs/unetpp_official_ds_e200_bs4/test_metrics.json
cat runs/unetpp_official_ds_e200_bs4/summary_metrics.json

