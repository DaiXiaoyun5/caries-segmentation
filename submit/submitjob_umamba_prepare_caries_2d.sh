#!/bin/bash
#SBATCH -J umamba_prep
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/umamba_prepare_2d_%j.out
#SBATCH -e /share/home/u2515283028/caries_project/runs/logs/umamba_prepare_2d_%j.out

set -Eeuo pipefail

PROJECT_ROOT=/share/home/u2515283028/caries_project
SOURCE_ROOT="${PROJECT_ROOT}/external_models/U-Mamba-main"
DATASET_ID=711
DATASET_NAME=Dataset711_CariXray

cd "${PROJECT_ROOT}"
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
conda activate umamba-caries
export PYTHONNOUSERSITE=1

module load http-proxy 2>/dev/null || true
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

test -f "${SOURCE_ROOT}/umamba/nnunetv2/nets/UMambaBot_2d.py"
test -f "${SOURCE_ROOT}/data/nnUNet_raw/${DATASET_NAME}/dataset.json"

echo "===== U-MAMBA OFFICIAL 2D PREPARATION ====="
which python
python --version
which nnUNetv2_plan_and_preprocess

python - <<'PY'
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results
print("nnUNet_raw:", nnUNet_raw)
print("nnUNet_preprocessed:", nnUNet_preprocessed)
print("nnUNet_results:", nnUNet_results)
PY

nnUNetv2_plan_and_preprocess \
    -d "${DATASET_ID}" \
    -c 2d \
    --verify_dataset_integrity

python - <<'PY'
import json
from datetime import datetime
from pathlib import Path

project = Path("/share/home/u2515283028/caries_project")
source = project / "external_models/U-Mamba-main"
meta = json.loads(
    (project / "nnunet_caries/caries_nnunet_meta.json").read_text(encoding="utf-8")
)
dataset_name = "Dataset711_CariXray"
pre_dir = source / "data/nnUNet_preprocessed" / dataset_name
pre_dir.mkdir(parents=True, exist_ok=True)

split = [{"train": meta["train_ids"], "val": meta["val_ids"]}]
split_path = pre_dir / "splits_final.json"
split_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

record_dir = project / "umamba_caries"
record_dir.mkdir(parents=True, exist_ok=True)
record = {
    "method": "U-Mamba-Bot 2D",
    "dataset_id": 711,
    "dataset_name": dataset_name,
    "configuration": "2d",
    "split": {
        "train": len(meta["train_ids"]),
        "val": len(meta["val_ids"]),
        "test": len(meta["test_ids"]),
        "path": str(split_path),
    },
    "source": str(source),
    "prepared_at": datetime.now().astimezone().isoformat(timespec="seconds"),
}
(record_dir / "preparation.json").write_text(
    json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("Fixed split:", split_path)
print("train/val/test:", len(meta["train_ids"]), len(meta["val_ids"]), len(meta["test_ids"]))
PY

test -d "${SOURCE_ROOT}/data/nnUNet_preprocessed/${DATASET_NAME}/nnUNetPlans_2d"
test -f "${SOURCE_ROOT}/data/nnUNet_preprocessed/${DATASET_NAME}/splits_final.json"
echo "U-Mamba preprocessing and fixed split are ready."
