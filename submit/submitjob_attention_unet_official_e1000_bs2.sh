#!/bin/bash
#SBATCH -J attunet_off
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/attention_unet_official_%j.out
#SBATCH -e runs/logs/attention_unet_official_%j.out

set -Eeuo pipefail

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
conda activate caries-baselines
export PYTHONNOUSERSITE=1

mkdir -p runs/logs

echo "===== ATTENTION U-NET OFFICIAL 2D ====="
which python
python --version
nvidia-smi

python -m py_compile \
  src/official_baseline_common.py \
  src/train_attention_unet_official.py

python src/train_attention_unet_official.py \
  --run-name attention_unet_official_2d_e1000_bs2 \
  --image-size 512 \
  --batch-size 2 \
  --val-batch-size 2 \
  --epochs 1000 \
  --lr 1e-4 \
  --weight-decay 1e-6 \
  --lr-decay-epochs 250 \
  --lr-decay-gamma 0.1 \
  --val-every-epochs 10 \
  --num-workers 2 \
  --seed 42

cat runs/attention_unet_official_2d_e1000_bs2/test_metrics.json
cat runs/attention_unet_official_2d_e1000_bs2/summary_metrics.json
