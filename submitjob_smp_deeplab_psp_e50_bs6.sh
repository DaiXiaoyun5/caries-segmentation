#!/bin/bash
#SBATCH -J smp_base
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/smp_deeplab_psp_e50_bs6_%j.out

set -e

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
echo "Task: train DeepLabV3+ and PSPNet, then measure Params/GFLOPs"
echo "============================================================"

echo "GPU info:"
nvidia-smi

echo "============================================================"
echo "Train DeepLabV3+ ResNet34 e50 bs6"
echo "============================================================"

srun python src/train_smp_segmentation.py \
  --run-name deeplabv3plus_resnet34_e50_bs6 \
  --model-type deeplabv3plus \
  --epochs 50 \
  --batch-size 6 \
  --num-workers 2 \
  --image-size 512 \
  --lr 3e-4 \
  --encoder-name resnet34 \
  --encoder-weights imagenet

echo "============================================================"
echo "Train PSPNet ResNet34 e50 bs6"
echo "============================================================"

srun python src/train_smp_segmentation.py \
  --run-name pspnet_resnet34_e50_bs6 \
  --model-type pspnet \
  --epochs 50 \
  --batch-size 6 \
  --num-workers 2 \
  --image-size 512 \
  --lr 3e-4 \
  --encoder-name resnet34 \
  --encoder-weights imagenet

echo "============================================================"
echo "Measure Params / GFLOPs"
echo "============================================================"

python src/measure_model_complexity.py

echo "============================================================"
echo "Job finished at: $(date)"
echo "============================================================"
