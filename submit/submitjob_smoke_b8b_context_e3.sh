#!/bin/bash
#SBATCH -J smoke_b8b
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b8b_context_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

echo "===== ENV ====="
which python
python --version
nvidia-smi

python -m py_compile src/train_resnet34_unet_brr_b8b_context_e200.py

echo "===== SMOKE RUN B8B-context: 3 epochs ====="
python src/train_resnet34_unet_brr_b8b_context_e200.py \
  --run-name smoke_b8b_context_e3 \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 2 \
  --epochs 3 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 0 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --context-branch-channels 128 \
  --context-dropout 0.0 \
  --context-init-scale 0.10

echo "===== DONE SMOKE B8B-context ====="
