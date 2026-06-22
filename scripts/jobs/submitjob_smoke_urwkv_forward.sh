#!/bin/bash
#SBATCH -J smoke_urwkv
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_urwkv_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

module load http-proxy || true

echo "===== ENV ====="
which python
python --version
python -m pip --version

echo "===== RUN SMOKE ====="
python src/smoke_urwkv_forward.py

echo "===== DONE ====="
