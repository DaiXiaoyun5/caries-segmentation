#!/bin/bash
#SBATCH -J e0_nnunet500
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/e0_nnunet2d_500_fold0_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet-caries

module load http-proxy || true
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

export nnUNet_raw=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_raw
export nnUNet_preprocessed=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_preprocessed
export nnUNet_results=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_results

DATASET_ID=701
DATASET_NAME=Dataset701_CariXray
CONFIG=2d
FOLD=0
TRAINER=nnUNetTrainer_500epochs
PRED_DIR=/share/home/u2515283028/caries_project/nnunet_caries/predictions/E0_Dataset701_CariXray_2d_fold0_500epochs

echo "============================================================"
echo "E0 nnU-Net v2 baseline 2d fold0 500 epochs"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Current dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "============================================================"

echo "===== GPU info ====="
nvidia-smi

echo "===== commands ====="
which python
python --version
which nnUNetv2_plan_and_preprocess
which nnUNetv2_train
which nnUNetv2_predict

echo "===== import trainer ====="
python - <<'PY'
from nnunetv2.training.nnUNetTrainer.variants.training_length.nnUNetTrainer_500epochs import nnUNetTrainer_500epochs
print("OK trainer:", nnUNetTrainer_500epochs)
PY

echo "============================================================"
echo "Step 0: dataset existence check"
echo "============================================================"
ls -lh "${nnUNet_raw}/${DATASET_NAME}/dataset.json"

echo "imagesTr count:"
find "${nnUNet_raw}/${DATASET_NAME}/imagesTr" -name "*_0000.png" | wc -l

echo "labelsTr count:"
find "${nnUNet_raw}/${DATASET_NAME}/labelsTr" -name "*.png" | wc -l

echo "imagesTs count:"
find "${nnUNet_raw}/${DATASET_NAME}/imagesTs" -name "*_0000.png" | wc -l

echo "============================================================"
echo "Step 1: plan and preprocess"
echo "============================================================"
echo "Using existing preprocessed data. Do not run plan/preprocess inside training jobs."
test -d "${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_2d"
echo "b2nd count:"
find "${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_2d" -name "*.b2nd" | wc -l
echo "pkl count:"
find "${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_2d" -name "*.pkl" | wc -l

echo "============================================================"
echo "Step 2: write fixed train/val split"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path

base = Path("/share/home/u2515283028/caries_project/nnunet_caries")
meta_path = base / "caries_nnunet_meta.json"
meta = json.loads(meta_path.read_text(encoding="utf-8"))

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
print("test:", len(meta["test_ids"]))
PY

echo "============================================================"
echo "Step 3: train E0"
echo "============================================================"
nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TRAINER}

echo "============================================================"
echo "Step 4: predict test set"
echo "============================================================"
rm -rf "${PRED_DIR}"
mkdir -p "${PRED_DIR}"

nnUNetv2_predict \
  -i "${nnUNet_raw}/${DATASET_NAME}/imagesTs" \
  -o "${PRED_DIR}" \
  -d ${DATASET_ID} \
  -c ${CONFIG} \
  -f ${FOLD} \
  -tr ${TRAINER}

echo "============================================================"
echo "Step 5: evaluate test set"
echo "============================================================"
python src/eval_nnunet_caries.py --pred-dir "${PRED_DIR}"

echo "============================================================"
echo "Step 6: save E0 run config"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime

out_dir = Path("/share/home/u2515283028/caries_project/nnunet_caries/predictions/E0_Dataset701_CariXray_2d_fold0_500epochs")

config = {
    "experiment_id": "E0",
    "experiment_name": "nnUNet_v2_2d_baseline_fold0_500epochs",
    "dataset_id": 701,
    "dataset_name": "Dataset701_CariXray",
    "configuration": "2d",
    "fold": 0,
    "trainer": "nnUNetTrainer_500epochs",
    "plans": "nnUNetPlans",
    "num_epochs": 500,
    "input_channels": 1,
    "input_modality": "grayscale",
    "labels": {
        "background": 0,
        "caries": 1
    },
    "data_format": "PNG",
    "nnunet_paths": {
        "nnUNet_raw": "/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_raw",
        "nnUNet_preprocessed": "/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_preprocessed",
        "nnUNet_results": "/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_results"
    },
    "augmentation": {
        "type": "nnU-Net v2 default online augmentation",
        "profile": "default_2d",
        "train_only": True,
        "val_test_augmentation": "none_random_augmentation",
        "note": "E0 uses native nnU-Net v2 2D augmentation. Validation/test use deterministic preprocessing and nnU-Net inference only."
    },
    "custom_split": {
        "source": "nnunet_caries/caries_nnunet_meta.json",
        "splits_final": "nnunet_caries/nnUNet_preprocessed/Dataset701_CariXray/splits_final.json"
    },
    "created_at": datetime.now().isoformat(timespec="seconds")
}

out_dir.mkdir(parents=True, exist_ok=True)
(out_dir / "e0_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
print("Saved:", out_dir / "e0_config.json")
PY

echo "============================================================"
echo "E0 finished at: $(date)"
echo "Prediction dir: ${PRED_DIR}"
echo "============================================================"
