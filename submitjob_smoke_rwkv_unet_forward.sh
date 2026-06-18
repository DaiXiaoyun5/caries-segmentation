#!/bin/bash
#SBATCH -J smoke_rwkvunet
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/smoke_rwkv_unet_%j.out

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

echo "===== CHECK IMPORTS ====="
python - <<'PY'
import torch
import timm
import einops
import h5py
import scipy
import skimage
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("timm:", timm.__version__)
print("einops ok")
print("h5py/scipy/skimage ok")
PY

echo "===== RUN RWKV-UNET SMOKE ====="
python src/smoke_rwkv_unet_forward.py

echo "===== DONE ====="
