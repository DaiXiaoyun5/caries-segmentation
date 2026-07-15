#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=/share/home/u2515283028/caries_project
ENV_NAME=caries-baselines

cd "${PROJECT_ROOT}"
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

module load http-proxy 2>/dev/null || true
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

mkdir -p \
    runs/logs \
    .cache/huggingface \
    .cache/torch \
    .cache/official_baselines \
    external_assets \
    external_models

export HF_HOME=${PROJECT_ROOT}/.cache/huggingface
export TORCH_HOME=${PROJECT_ROOT}/.cache/torch
export HF_HUB_DISABLE_TELEMETRY=1

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1

python -m pip install --upgrade "pip==24.0" "setuptools==69.5.1" "wheel==0.43.0"
python -m pip install \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
python -m pip install \
    -c configs/official_baselines_constraints.txt \
    "numpy==1.26.4" \
    "Pillow==10.4.0" \
    "scipy==1.12.0" \
    "matplotlib==3.8.4" \
    "opencv-python==4.9.0.80" \
    "segmentation-models-pytorch==0.3.4" \
    "timm==0.9.7" \
    "transformers==4.49.0" \
    "einops==0.7.0" \
    "yacs==0.1.8" \
    "PyYAML==6.0.1" \
    "thop==0.1.1.post2209072238"
python -m pip check

python - <<'PY'
import torch
import torchvision
import transformers
from transformers import SegformerForSemanticSegmentation
import segmentation_models_pytorch as smp
import timm
import yacs
import einops

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("transformers:", transformers.__version__)
if transformers.__version__ != "4.49.0":
    raise RuntimeError(
        "Expected transformers 4.49.0 with torch 2.0.1, got "
        + transformers.__version__
    )
print("SegformerForSemanticSegmentation import: OK")
print("segmentation-models-pytorch:", smp.__version__)
print("timm:", timm.__version__)
print("yacs/einops: OK")
print("CUDA available on this node:", torch.cuda.is_available())
print("Login-node CUDA=False is expected; training jobs verify the allocated GPU.")
PY

# Download on the network-enabled login node, then prove that the exact model
# loaders can reopen both checkpoints without Hugging Face network access.
python scripts/setup/prepare_official_baseline_weights.py --model all
python scripts/setup/prepare_official_baseline_weights.py \
    --model all \
    --verify-only

echo "External baseline environment is ready: ${ENV_NAME}"
