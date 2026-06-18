#!/bin/bash
#SBATCH -J e2_boundary500
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/e2_nnunet2d_boundary_500_fold0_%j.out

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
TRAINER=nnUNetTrainer_E2Boundary_500epochs
PRED_DIR=/share/home/u2515283028/caries_project/nnunet_caries/predictions/E2_Dataset701_CariXray_2d_fold0_Boundary_500epochs

echo "============================================================"
echo "E2 nnU-Net v2 2d fold0 + CI-region + Boundary auxiliary decoder, 500 epochs"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "nnUNet_raw=${nnUNet_raw}"
echo "nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "nnUNet_results=${nnUNet_results}"
echo "============================================================"

echo "===== GPU info ====="
nvidia-smi

echo "===== import checks ====="
python - <<'PY'
from nnunetv2.training.loss.cifs_boundary_loss import CIRegionBoundaryLoss
from nnunetv2.training.nnUNetTrainer.variants.boundary.nnUNetTrainer_E2Boundary_500epochs import nnUNetTrainer_E2Boundary_500epochs
print("OK loss:", CIRegionBoundaryLoss)
print("OK trainer:", nnUNetTrainer_E2Boundary_500epochs)
PY

echo "============================================================"
echo "Step 0: dataset check"
echo "============================================================"
ls -lh "${nnUNet_raw}/${DATASET_NAME}/dataset.json"
echo "imagesTr:" $(find "${nnUNet_raw}/${DATASET_NAME}/imagesTr" -name "*_0000.png" | wc -l)
echo "labelsTr:" $(find "${nnUNet_raw}/${DATASET_NAME}/labelsTr" -name "*.png" | wc -l)
echo "imagesTs:" $(find "${nnUNet_raw}/${DATASET_NAME}/imagesTs" -name "*_0000.png" | wc -l)

echo "============================================================"
echo "Step 1: plan/preprocess"
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
print("test:", len(meta["test_ids"]))
PY

echo "============================================================"
echo "Step 3: train E2"
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
echo "Step 6: save E2 config"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime

base = Path("/share/home/u2515283028/caries_project/nnunet_caries")
stats_path = base / "cifs_component_stats.json"
stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

out_dir = Path("/share/home/u2515283028/caries_project/nnunet_caries/predictions/E2_Dataset701_CariXray_2d_fold0_Boundary_500epochs")
out_dir.mkdir(parents=True, exist_ok=True)

config = {
    "experiment_id": "E2",
    "experiment_name": "nnUNet_v2_2d_CIRegion_Boundary_fold0_500epochs",
    "dataset_id": 701,
    "dataset_name": "Dataset701_CariXray",
    "configuration": "2d",
    "fold": 0,
    "trainer": "nnUNetTrainer_E2Boundary_500epochs",
    "base_trainer": "nnUNetTrainer_500epochs",
    "num_epochs": 500,
    "network": {
        "base": "PlainConvUNet",
        "modification": "BoundaryAuxDecoder wrapper",
        "seg_outputs": 7,
        "boundary_outputs": 7,
        "inference_output": "segmentation only"
    },
    "loss": {
        "name": "CI-region + Boundary auxiliary loss",
        "lambda_ft": 0.5,
        "lambda_topk": 0.2,
        "lambda_inst": 0.2,
        "lambda_boundary": 0.3,
        "tversky_alpha": 0.25,
        "tversky_beta": 0.75,
        "focal_gamma": 0.75,
        "topk_percent": 0.20,
        "instance_topq": 0.20,
        "component_weight_gamma": 0.30,
        "component_weight_max": 4.0,
        "boundary": {
            "target": "adaptive component boundary band",
            "radius_formula": "clip(round(0.15 * sqrt(component_area)), 2, 8)",
            "loss": "Boundary Dice + BCE"
        },
        "component_area_stats": stats
    },
    "augmentation": {
        "type": "nnU-Net v2 default online augmentation",
        "profile": "default_2d",
        "train_only": True,
        "val_test_augmentation": "none_random_augmentation",
        "note": "E2 uses native nnU-Net v2 2D augmentation. Validation/test use deterministic preprocessing and nnU-Net inference only."
    },
    "created_at": datetime.now().isoformat(timespec="seconds")
}

(out_dir / "e2_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
print("Saved:", out_dir / "e2_config.json")
PY

echo "============================================================"
echo "E2 finished at: $(date)"
echo "Prediction dir: ${PRED_DIR}"
echo "============================================================"
