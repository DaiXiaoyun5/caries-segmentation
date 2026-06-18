#!/bin/bash
#SBATCH -J inspect_smp_decoder
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/inspect_smp_decoder_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

echo "===== Inspect SMP decoder structure ====="
python src/inspect_smp_decoder_structure.py

echo "===== Done ====="
