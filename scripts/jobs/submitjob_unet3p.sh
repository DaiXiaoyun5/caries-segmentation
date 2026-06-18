#!/bin/bash
#SBATCH -J unet3p
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/unet3p_%j.out

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project

srun python src/train_unet3p.py \
  --run-name unet3p_baseline \
  --epochs 30 \
  --batch-size 4 \
  --num-workers 2 \
  --image-size 512 \
  --lr 3e-4 \
  --base-channels 64 \
  --deep-supervision 1
