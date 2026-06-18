#!/bin/bash
#SBATCH -J unetpp_simam
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/unetpp_simam_v1_e30_bs6_fix_%j.out

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /share/home/u2515283028/caries_project

echo "============================================================"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Current dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Experiment: UNet++ + SimAM, e30 bs6"
echo "============================================================"

srun python src/train_unetpp_simam.py \
  --run-name unetpp_simam_v1_e30_bs6_fix \
  --epochs 30 \
  --batch-size 6 \
  --num-workers 2 \
  --image-size 512 \
  --lr 3e-4 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --simam-lambda 1e-4

echo "============================================================"
echo "Job finished at: $(date)"
echo "============================================================"
