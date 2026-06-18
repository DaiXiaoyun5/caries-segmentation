#!/bin/bash
#SBATCH -J smoke_b6
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_b6_component_%j.out

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

echo "===== CHECK SCRIPT ====="
python -m py_compile src/train_resnet34_unet_brr_b6_component_recall_e200.py

echo "===== SMOKE RUN B6: 3 epochs, batch size 2 ====="
python src/train_resnet34_unet_brr_b6_component_recall_e200.py \
  --run-name smoke_b6_brr_rbsg_component_recall_e3 \
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
  --component-weight 0.15 \
  --component-topk-ratio 0.30 \
  --component-gamma 2.0 \
  --component-min-area 3 \
  --component-max-count 64

echo "===== DONE SMOKE B6 ====="
