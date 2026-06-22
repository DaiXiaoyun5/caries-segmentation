#!/bin/bash
#SBATCH -J unetpp_v1_fix
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/unetpp_v1_fix_%j.out

module load anaconda3/4.12.0
source activate caries-train

cd /share/home/u2515283028/caries_project

srun python src/train_unetpp.py \
  --run-name unetpp_baseline_v1_fix \
  --epochs 20 \
  --batch-size 4 \
  --num-workers 2 \
  --image-size 512 \
  --lr 1e-3 \
  --encoder-name resnet34 \
  --encoder-weights imagenet
