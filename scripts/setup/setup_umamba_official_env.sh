#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=/share/home/u2515283028/caries_project
ASSET_DIR="${PROJECT_ROOT}/external_assets"
ASSET_ZIP="${ASSET_DIR}/U-Mamba-main.zip"
WHEEL_DIR="${PROJECT_ROOT}/external_assets/umamba_wheels"
CAUSAL_CONV1D_WHEEL="${WHEEL_DIR}/causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
MAMBA_SSM_WHEEL="${WHEEL_DIR}/mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
EXTERNAL_MODELS="${PROJECT_ROOT}/external_models"
SOURCE_ROOT="${EXTERNAL_MODELS}/U-Mamba-main"
ENV_NAME=umamba-caries
EXPECTED_SHA256=7d03c731ddb30061d46f1f40d42885ffce868c8581dffcce38794f20c4f8decc

cd "${PROJECT_ROOT}"
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

module load http-proxy 2>/dev/null || true
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

mkdir -p "${ASSET_DIR}" "${EXTERNAL_MODELS}" runs/logs umamba_caries/predictions

require_nonempty_file() {
    local required_path="$1"
    if [[ ! -s "${required_path}" ]]; then
        echo "Missing or empty required wheel: ${required_path}" >&2
        exit 2
    fi
    echo "Using local wheel: ${required_path}"
}

if [[ ! -f "${ASSET_ZIP}" ]]; then
    echo "Missing official source archive: ${ASSET_ZIP}" >&2
    echo "Upload the supplied U-Mamba-main.zip to external_assets first." >&2
    exit 2
fi

echo "${EXPECTED_SHA256}  ${ASSET_ZIP}" | sha256sum -c -
require_nonempty_file "${CAUSAL_CONV1D_WHEEL}"
require_nonempty_file "${MAMBA_SSM_WHEEL}"

if [[ ! -f "${SOURCE_ROOT}/umamba/setup.py" ]]; then
    if [[ -e "${SOURCE_ROOT}" ]]; then
        echo "Incomplete source directory already exists: ${SOURCE_ROOT}" >&2
        echo "Move it aside manually, then rerun this setup script." >&2
        exit 2
    fi
    python -m zipfile -e "${ASSET_ZIP}" "${EXTERNAL_MODELS}"
fi

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx "${ENV_NAME}"; then
    conda create -n "${ENV_NAME}" python=3.10 -y
fi
conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1

python -m pip install --upgrade "pip==24.0" "setuptools==69.5.1" "wheel==0.43.0"
python -m pip install \
    torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

python -m pip install -c configs/umamba_official_constraints.txt \
    "numpy==1.26.4" \
    "scipy==1.11.4" \
    "scikit-image==0.22.0" \
    "scikit-learn==1.3.2" \
    "pandas==2.1.4" \
    "matplotlib==3.8.2" \
    "SimpleITK==2.3.1" \
    "imagecodecs==2024.1.1" \
    "dynamic-network-architectures==0.2" \
    "acvl-utils==0.2" \
    "batchgenerators==0.25" \
    "monai==1.3.0" \
    "transformers==4.38.2" \
    "opencv-python==4.9.0.80" \
    "dicom2nifti==2.4.9" \
    "medpy==0.5.1" \
    "nibabel==5.2.1" \
    "tifffile==2024.2.12" \
    "seaborn==0.13.2" \
    "graphviz==0.20.3" \
    "yacs==0.1.8" \
    "einops==0.7.0" \
    "packaging==23.2" \
    "ninja==1.11.1.1" \
    "tqdm==4.66.2" \
    "requests==2.31.0"

# These are the official March-2024 prebuilt CUDA 11.8 / torch 2.0 / CPython
# 3.10 / old-ABI wheels. They are installed from local assets so setup does not
# depend on GitHub access and does not compile CUDA code on the login node.
python -m pip install --no-deps "${CAUSAL_CONV1D_WHEEL}"
python -m pip install --no-deps "${MAMBA_SSM_WHEEL}"

python -m pip install --no-deps -e "${SOURCE_ROOT}/umamba"
python -m pip check

UMAMBA_RAW="${SOURCE_ROOT}/data/nnUNet_raw"
SOURCE_DATASET="${PROJECT_ROOT}/nnunet_caries/nnUNet_raw/Dataset701_CariXray"
TARGET_DATASET="${UMAMBA_RAW}/Dataset711_CariXray"
mkdir -p "${UMAMBA_RAW}" "${SOURCE_ROOT}/data/nnUNet_preprocessed" "${SOURCE_ROOT}/data/nnUNet_results"

if [[ ! -d "${SOURCE_DATASET}" ]]; then
    echo "Missing prepared nnU-Net raw dataset: ${SOURCE_DATASET}" >&2
    exit 2
fi
if [[ -L "${TARGET_DATASET}" ]]; then
    if [[ "$(readlink -f "${TARGET_DATASET}")" != "$(readlink -f "${SOURCE_DATASET}")" ]]; then
        echo "Dataset symlink points to an unexpected location: ${TARGET_DATASET}" >&2
        exit 2
    fi
elif [[ -e "${TARGET_DATASET}" ]]; then
    echo "Refusing to replace existing U-Mamba dataset path: ${TARGET_DATASET}" >&2
    exit 2
else
    ln -s "${SOURCE_DATASET}" "${TARGET_DATASET}"
fi

python - <<'PY'
import torch
import causal_conv1d
import mamba_ssm
from importlib.metadata import version
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available on this node:", torch.cuda.is_available())
print("causal-conv1d:", version("causal-conv1d"))
print("mamba-ssm:", version("mamba-ssm"))
print("trainer:", nnUNetTrainerUMambaBot.__name__)
print("Login-node CUDA=False is expected; the GPU job performs a real CUDA forward preflight.")
PY

echo "U-Mamba official environment is ready: ${ENV_NAME}"
echo "Source: ${SOURCE_ROOT}"
echo "Dataset link: ${TARGET_DATASET}"
