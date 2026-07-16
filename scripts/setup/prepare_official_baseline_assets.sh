#!/bin/bash

set -Eeuo pipefail

PROJECT_ROOT=/share/home/u2515283028/caries_project

cd "${PROJECT_ROOT}"
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

if ! conda env list | awk 'NF && $1 !~ /^#/ {print $1}' | grep -qx caries-baselines; then
  echo "Missing conda environment: caries-baselines" >&2
  echo "Run bash scripts/setup/setup_official_baselines_env.sh first." >&2
  exit 2
fi
conda activate caries-baselines
export PYTHONNOUSERSITE=1

module load http-proxy 2>/dev/null || true
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

mkdir -p .cache/huggingface .cache/torch .cache/official_baselines
export HF_HOME=${PROJECT_ROOT}/.cache/huggingface
export TORCH_HOME=${PROJECT_ROOT}/.cache/torch
export HF_HUB_DISABLE_TELEMETRY=1

python scripts/setup/prepare_official_baseline_weights.py --model all
python scripts/setup/prepare_official_baseline_weights.py \
  --model all \
  --verify-only

echo "Official baseline pretrained assets are ready."
