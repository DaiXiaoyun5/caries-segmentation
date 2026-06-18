#!/bin/bash
#SBATCH -J u3p_edge_dual
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/res34_u3p_edge_dual_decoder_%j.out

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project

srun python src/train_resnet34_unet3p_edge_dual_decoder.py \
  --run-name resnet34_unet3p_edge_dual_decoder_e50_bs6 \
  --epochs 50 \
  --batch-size 6 \
  --num-workers 2 \
  --image-size 512 \
  --lr 3e-4 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --cat-channels 64 \
  --deep-supervision 1 \
  --lambda-edge 0.3 \
  --lambda-center 0.5 \
  --heatmap-sigma 15
