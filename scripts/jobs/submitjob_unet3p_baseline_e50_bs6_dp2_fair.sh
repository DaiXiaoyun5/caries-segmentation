#!/bin/bash
#SBATCH -J unet3p_dp2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:2
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/unet3p_baseline_e50_bs6_dp2_fair_%j.out

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
echo "Experiment: fair UNet3+ baseline e50 bs6, 2 GPUs, DataParallel"
echo "============================================================"

echo "GPU info:"
nvidia-smi

echo "============================================================"
echo "Check training script:"
ls -lh src/train_unet3p_dp2.py
grep -n -A12 -B4 "model = UNet3Plus" src/train_unet3p_dp2.py
echo "============================================================"

srun python src/train_unet3p_dp2.py \
  --run-name unet3p_baseline_e50_bs6_dp2_fair \
  --epochs 50 \
  --batch-size 6 \
  --num-workers 4 \
  --image-size 512 \
  --lr 3e-4 \
  --base-channels 64 \
  --deep-supervision 1

echo "============================================================"
echo "Job finished at: $(date)"
echo "============================================================"
