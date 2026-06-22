#!/bin/bash
#SBATCH -J b4_lshnw
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b4_lshnw_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b4_lshnw_e200.py

echo "===== RUN B4: BRR-RBSG + Lesion-size and Hard-negative Weighted Loss ====="
python src/train_resnet34_unet_brr_b4_lshnw_e200.py \
  --run-name b4_brr_rbsg_lshnw_auglite_e200_constlr_bs6 \
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
  --small-ref-ratio 0.01 \
  --small-gamma 0.5 \
  --small-max-weight 2.0 \
  --rim-hard-weight 0.5

echo "===== DONE B4 ====="
