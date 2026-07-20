#!/bin/bash
#SBATCH -J umamba_bot2d
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/umamba_bot_2d_fold0_%j.out
#SBATCH -e /share/home/u2515283028/caries_project/runs/logs/umamba_bot_2d_fold0_%j.out

set -Eeuo pipefail

PROJECT_ROOT=/share/home/u2515283028/caries_project
SOURCE_ROOT="${PROJECT_ROOT}/external_models/U-Mamba-main"
DATA_ROOT="${SOURCE_ROOT}/data"
DATASET_ID=711
DATASET_NAME=Dataset711_CariXray
CONFIGURATION=2d
FOLD=0
TRAINER=nnUNetTrainerUMambaBot
TRAINING_DIR="${DATA_ROOT}/nnUNet_results/${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIGURATION}/fold_${FOLD}"
PRED_DIR="${PROJECT_ROOT}/umamba_caries/predictions/umamba_bot_official_2d_fold0"
RUN_DIR="${PROJECT_ROOT}/runs/umamba_bot_official_2d_fold0"

require_file() {
    local required_path="$1"
    if [[ ! -f "${required_path}" ]]; then
        echo "Missing required file: ${required_path}" >&2
        exit 2
    fi
}

require_dir() {
    local required_path="$1"
    if [[ ! -d "${required_path}" ]]; then
        echo "Missing required directory: ${required_path}" >&2
        exit 2
    fi
}

cd "${PROJECT_ROOT}"
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
conda activate umamba-caries
export PYTHONNOUSERSITE=1
export nnUNet_raw="${DATA_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${DATA_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${DATA_ROOT}/nnUNet_results"

mkdir -p runs/logs "${PRED_DIR}" "${RUN_DIR}"
require_file "${SOURCE_ROOT}/umamba/nnunetv2/nets/UMambaBot_2d.py"
require_dir "${DATA_ROOT}/nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans_2d"
require_file "${DATA_ROOT}/nnUNet_preprocessed/${DATASET_NAME}/splits_final.json"

echo "===== U-MAMBA-BOT OFFICIAL 2D / FIXED FOLD 0 ====="
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
which python
python --version
nvidia-smi

python - <<'PY'
import os
from pathlib import Path

import torch
from mamba_ssm import Mamba
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerUMambaBot import nnUNetTrainerUMambaBot

actual_paths = {
    "nnUNet_raw": nnUNet_raw,
    "nnUNet_preprocessed": nnUNet_preprocessed,
    "nnUNet_results": nnUNet_results,
}
for name, value in actual_paths.items():
    expected = os.environ[name]
    if Path(value).resolve() != Path(expected).resolve():
        raise RuntimeError(f"{name} mismatch: imported={value}, expected={expected}")
print("nnU-Net path preflight: OK")

print("torch:", torch.__version__)
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("The allocated compute node does not expose a CUDA GPU to PyTorch")
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))

model = Mamba(d_model=32, d_state=16, d_conv=4, expand=2).cuda().eval()
x = torch.randn(1, 64, 32, device="cuda")
with torch.no_grad():
    y = model(x)
torch.cuda.synchronize()
assert y.shape == x.shape, (x.shape, y.shape)
print("Mamba CUDA forward preflight: OK", tuple(y.shape))
print("trainer:", nnUNetTrainerUMambaBot.__name__)
PY

which nnUNetv2_train
which nnUNetv2_predict

nnUNetv2_train \
    "${DATASET_ID}" \
    "${CONFIGURATION}" \
    "${FOLD}" \
    -tr "${TRAINER}"

require_file "${TRAINING_DIR}/checkpoint_final.pth"

nnUNetv2_predict \
    -i "${DATA_ROOT}/nnUNet_raw/${DATASET_NAME}/imagesTs" \
    -o "${PRED_DIR}" \
    -d "${DATASET_ID}" \
    -c "${CONFIGURATION}" \
    -f "${FOLD}" \
    -tr "${TRAINER}" \
    --disable_tta

python src/eval_nnunet_caries.py --pred-dir "${PRED_DIR}"

python src/export_umamba_official_results.py \
    --run-name umamba_bot_official_2d_fold0 \
    --checkpoint "${TRAINING_DIR}/checkpoint_final.pth" \
    --validation-summary "${TRAINING_DIR}/validation/summary.json" \
    --pred-dir "${PRED_DIR}" \
    --output-dir "${RUN_DIR}"

cat "${RUN_DIR}/test_metrics.json"
cat "${RUN_DIR}/summary_metrics.json"
echo "U-Mamba-Bot 2D finished."
