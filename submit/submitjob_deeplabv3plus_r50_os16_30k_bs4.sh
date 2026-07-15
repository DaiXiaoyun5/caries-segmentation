#!/bin/bash
#SBATCH -J deeplabv3p
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/deeplabv3plus_r50_%j.out
#SBATCH -e runs/logs/deeplabv3plus_r50_%j.out

set -Eeuo pipefail

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
conda activate caries-baselines
export PYTHONNOUSERSITE=1

mkdir -p runs/logs .cache/torch
export TORCH_HOME=/share/home/u2515283028/caries_project/.cache/torch

echo "===== DEEPLABV3+ RESNET50 OS16 ====="
which python
python --version
nvidia-smi

python - <<'PY'
import segmentation_models_pytorch as smp
print("segmentation_models_pytorch:", smp.__version__)
PY

python -m py_compile \
  scripts/setup/prepare_official_baseline_weights.py \
  src/official_baseline_common.py \
  src/train_deeplabv3plus_official.py

python scripts/setup/prepare_official_baseline_weights.py \
  --model deeplab \
  --verify-only

python src/train_deeplabv3plus_official.py \
  --run-name deeplabv3plus_r50_os16_30k_bs4 \
  --image-size 512 \
  --batch-size 4 \
  --val-batch-size 4 \
  --total-iters 30000 \
  --val-interval-iters 1000 \
  --lr 0.0025 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 1

cat runs/deeplabv3plus_r50_os16_30k_bs4/test_metrics.json
cat runs/deeplabv3plus_r50_os16_30k_bs4/summary_metrics.json
