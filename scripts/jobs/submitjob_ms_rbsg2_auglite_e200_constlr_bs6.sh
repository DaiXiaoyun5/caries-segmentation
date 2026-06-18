#!/bin/bash
#SBATCH -J ms_rbsg2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/ms_rbsg2_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_ms_rbsg2_e200.py

echo "===== RUN MS-RBSG-2: RBSG at decoder 1/8 and 1/4 + final B2 RBSG ====="
python src/train_resnet34_unet_brr_ms_rbsg2_e200.py \
  --run-name ms_rbsg2_1_8_1_4_auglite_e200_constlr_bs6 \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 6 \
  --epochs 200 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --ms-boundary-weight 0.05 \
  --ms-rim-weight 0.025

echo "===== DONE MS-RBSG-2 ====="
