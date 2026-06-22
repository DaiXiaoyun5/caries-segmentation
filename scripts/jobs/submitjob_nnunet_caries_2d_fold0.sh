#!/bin/bash
set -e
#SBATCH -J nnunet2d
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/nnunet_caries_2d_fold0_%j.out

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet-caries

export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

cd /share/home/u2515283028/caries_project

export nnUNet_raw=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_raw
export nnUNet_preprocessed=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_preprocessed
export nnUNet_results=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_results

DATASET_ID=701
DATASET_NAME=Dataset701_CariXray
PRED_DIR=/share/home/u2515283028/caries_project/nnunet_caries/predictions/Dataset701_CariXray_2d_fold0

echo "============================================================"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Current dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "============================================================"

echo "GPU info:"
nvidia-smi

echo "============================================================"
echo "Check nnU-Net command"
echo "============================================================"
which nnUNetv2_plan_and_preprocess
which nnUNetv2_train
which nnUNetv2_predict

echo "============================================================"
echo "Step 1: plan and preprocess"
echo "============================================================"
nnUNetv2_plan_and_preprocess -d ${DATASET_ID} -c 2d --verify_dataset_integrity

echo "============================================================"
echo "Step 2: write fixed train/val split"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path

base = Path("/share/home/u2515283028/caries_project/nnunet_caries")
meta = json.loads((base / "caries_nnunet_meta.json").read_text(encoding="utf-8"))

dataset_folder = meta["dataset_folder_name"]
pre_dir = Path(meta["nnUNet_preprocessed"]) / dataset_folder
pre_dir.mkdir(parents=True, exist_ok=True)

splits = [{
    "train": meta["train_ids"],
    "val": meta["val_ids"],
}]

out = pre_dir / "splits_final.json"
out.write_text(json.dumps(splits, indent=2), encoding="utf-8")

print("Saved custom split:", out)
print("train:", len(meta["train_ids"]))
print("val:", len(meta["val_ids"]))
PY

echo "============================================================"
echo "Step 3: choose trainer"
echo "============================================================"

TRAINER_ARG=""
python - <<'PY'
import importlib.util
candidates = [
    "nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_50epochs",
    "nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_100epochs",
]
for c in candidates:
    if importlib.util.find_spec(c) is not None:
        print(c.split(".")[-1])
        raise SystemExit
print("DEFAULT")
PY

TRAINER_NAME=$(python - <<'PY'
import importlib.util
candidates = [
    "nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_50epochs",
    "nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_100epochs",
]
for c in candidates:
    if importlib.util.find_spec(c) is not None:
        print(c.split(".")[-1])
        raise SystemExit
print("DEFAULT")
PY
)

if [ "$TRAINER_NAME" = "DEFAULT" ]; then
  echo "No short trainer found. Using default nnUNetTrainer. This may run much longer."
  TRAINER_ARG=""
else
  echo "Using trainer: ${TRAINER_NAME}"
  TRAINER_ARG="-tr ${TRAINER_NAME}"
fi

echo "============================================================"
echo "Step 4: train nnU-Net 2d fold 0"
echo "============================================================"
nnUNetv2_train ${DATASET_ID} 2d 0 ${TRAINER_ARG}

echo "============================================================"
echo "Step 5: predict test set"
echo "============================================================"
mkdir -p "${PRED_DIR}"

nnUNetv2_predict \
  -i "${nnUNet_raw}/${DATASET_NAME}/imagesTs" \
  -o "${PRED_DIR}" \
  -d ${DATASET_ID} \
  -c 2d \
  -f 0 \
  ${TRAINER_ARG}

echo "============================================================"
echo "Step 6: evaluate test set"
echo "============================================================"
python src/eval_nnunet_caries.py --pred-dir "${PRED_DIR}"

echo "============================================================"
echo "Job finished at: $(date)"
echo "Prediction dir: ${PRED_DIR}"
echo "Metrics file: ${PRED_DIR}/nnunet_test_metrics.json"
echo "============================================================"
