#!/bin/bash
#SBATCH -J res34_u3p_cbsa_ppm
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/%x_%j.out

cd /share/home/u2515283028/caries_project

mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

echo "===== Job Info ====="
echo "job name: $SLURM_JOB_NAME"
echo "job id  : $SLURM_JOB_ID"
echo "node    : $SLURMD_NODENAME"
echo "pwd     : $(pwd)"
echo "python  : $(which python)"
python -V

echo "===== GPU Info ====="
nvidia-smi || true

echo "===== Start Training: ResNet34-UNet3+ + CBSA + PPM-lite ====="

srun python -u src/train_resnet34_unet3p_cbsa_ppm.py \
  --run-name resnet34_unet3p_cbsa_ppm_e50_bs6 \
  --image-size 512 \
  --batch-size 6 \
  --epochs 50 \
  --lr 3e-4 \
  --num-workers 4 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --cat-channels 64 \
  --deep-supervision 1 \
  --cbsa-position all \
  --cbsa-lambda 1e-4 \
  --cbsa-alpha 0.2 \
  --cbsa-beta 0.1 \
  --ppm-hidden 64 \
  --ppm-pool-sizes 1,2,3,6 \
  --ppm-dropout 0.0 \
  --ppm-res-scale 0.1

echo "===== Done: ResNet34-UNet3+ + CBSA + PPM-lite ====="
