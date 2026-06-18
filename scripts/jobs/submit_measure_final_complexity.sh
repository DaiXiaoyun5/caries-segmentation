#!/bin/bash
#SBATCH -J final_complexity
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/final_complexity_%j.out

set -e
PROJECT_ROOT="/share/home/u2515283028/caries_project"
cd "$PROJECT_ROOT"
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

srun python src/measure_final_model_complexity.py --image-size 512 --warmup 10 --repeat 50
