#!/bin/bash
#SBATCH -J b2_pbd
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b2_prebd_only_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b2_preaux_e200.py

echo "===== RUN B2-prebd-only: pre_boundary=0.05, pre_rim=0.00 ====="
python src/train_resnet34_unet_brr_b2_preaux_e200.py \
  --run-name b2_prebd_only_pb005_pr000_auglite_e200_constlr_bs6 \
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
  --pre-boundary-weight 0.05 \
  --pre-rim-weight 0.00

echo "===== DONE B2-prebd-only ====="
