#!/bin/bash
#SBATCH -J u3p_simam_scse
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/res34_u3p_simam_scse_hd1_%j.out

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

cd /share/home/u2515283028/caries_project

echo "============================================================"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Current dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Experiment: Edge + SimAM + SCSE-hd1"
echo "============================================================"

srun python src/train_resnet34_unet3p_edge_simam_scse_hd1.py \
  --run-name resnet34_unet3p_edge_simam_scse_hd1_e50_bs6 \
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
  --edge-kernel-size 3 \
  --edge-stat-samples 800 \
  --edge-vis-threshold 0.2 \
  --simam-lambda 1e-4 \
  --simam-position all \
  --scse-reduction 16

echo "============================================================"
echo "Job finished at: $(date)"
echo "============================================================"
