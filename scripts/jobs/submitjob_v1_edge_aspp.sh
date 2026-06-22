#!/bin/bash
#SBATCH -J unetpp_v1_edge_aspp
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/unetpp_v1_edge_aspp_%j.out

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project

srun python src/train_method_v1_edge_aspp.py \
  --run-name unetpp_v1_edge_aspp \
  --epochs 30 \
  --batch-size 4 \
  --num-workers 2 \
  --image-size 512 \
  --lr 3e-4 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --lambda-edge 0.3 \
  --edge-stat-samples 800 \
  --edge-vis-threshold 0.2
