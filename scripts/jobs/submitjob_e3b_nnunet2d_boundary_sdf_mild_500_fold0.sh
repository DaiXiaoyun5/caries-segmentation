#!/bin/bash
#SBATCH -J e3b_bsdfm500
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/e3b_nnunet2d_boundary_sdf_mild_500_fold0_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet-caries

export nnUNet_raw=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_raw
export nnUNet_preprocessed=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_preprocessed
export nnUNet_results=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_results

DATASET_ID=701
DATASET_NAME=Dataset701_CariXray
CONFIG=2d
FOLD=0
TRAINER=nnUNetTrainer_E3BoundarySDFMild_500epochs
PRED_DIR=/share/home/u2515283028/caries_project/nnunet_caries/predictions/E3b_Dataset701_CariXray_2d_fold0_BoundarySDFMild_500epochs

echo "============================================================"
echo "E3b nnU-Net v2 2d fold0 + mild CI-region + Boundary + light SDF, 500 epochs"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "============================================================"

nvidia-smi

echo "===== import checks ====="
python - <<'PY'
from nnunetv2.training.nnUNetTrainer.variants.boundary_sdf.nnUNetTrainer_E3BoundarySDFMild_500epochs import nnUNetTrainer_E3BoundarySDFMild_500epochs
from nnunetv2.training.loss.cifs_boundary_sdf_loss import CIRegionBoundarySDFLoss
print("OK trainer:", nnUNetTrainer_E3BoundarySDFMild_500epochs)
print("OK loss:", CIRegionBoundarySDFLoss)
PY

echo "===== use existing preprocessed data ====="
test -d "${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_2d"

echo "b2nd count:"
find "${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_2d" -name "*.b2nd" | wc -l

echo "pkl count:"
find "${nnUNet_preprocessed}/${DATASET_NAME}/nnUNetPlans_2d" -name "*.pkl" | wc -l

echo "===== check fixed split ====="
ls -lh "${nnUNet_preprocessed}/${DATASET_NAME}/splits_final.json"

echo "============================================================"
echo "Step 1: train E3b"
echo "============================================================"
nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLD} -tr ${TRAINER}

echo "============================================================"
echo "Step 2: predict test set"
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
echo "Step 3: evaluate test set"
echo "============================================================"
python src/eval_nnunet_caries.py --pred-dir "${PRED_DIR}"

echo "============================================================"
echo "Step 4: save E3b config"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path
from datetime import datetime

base = Path("/share/home/u2515283028/caries_project/nnunet_caries")
stats_path = base / "cifs_component_stats.json"
stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.exists() else {}

out_dir = Path("/share/home/u2515283028/caries_project/nnunet_caries/predictions/E3b_Dataset701_CariXray_2d_fold0_BoundarySDFMild_500epochs")
out_dir.mkdir(parents=True, exist_ok=True)

config = {
    "experiment_id": "E3b",
    "experiment_name": "nnUNet_v2_2d_CIRegionMild_Boundary_LightSDF_fold0_500epochs",
    "dataset_id": 701,
    "dataset_name": "Dataset701_CariXray",
    "configuration": "2d",
    "fold": 0,
    "trainer": "nnUNetTrainer_E3BoundarySDFMild_500epochs",
    "base_trainer": "nnUNetTrainer_500epochs",
    "num_epochs": 500,
    "network": {
        "base": "PlainConvUNet",
        "modification": "BoundarySDFDecoder wrapper",
        "seg_outputs": 7,
        "boundary_outputs": 7,
        "sdf_outputs": 7,
        "inference_output": "segmentation only"
    },
    "loss": {
        "name": "mild CI-region + Boundary + light SDF auxiliary loss",
        "lambda_ft": 0.30,
        "lambda_topk": 0.10,
        "lambda_inst": 0.10,
        "lambda_boundary": 0.30,
        "lambda_sdf": 0.05,
        "tversky_alpha": 0.30,
        "tversky_beta": 0.70,
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
        "sdf": {
            "target": "signed distance map, positive inside lesion and negative outside",
            "normalization": "clip(D_gt / 20, -1, 1)",
            "prediction_activation": "tanh",
            "loss": "weighted SmoothL1",
            "lambda_sdf": 0.05,
            "eta": 2.0,
            "tau": 4.0
        },
        "component_area_stats": stats
    },
    "augmentation": {
        "type": "nnU-Net v2 default online augmentation",
        "profile": "default_2d",
        "train_only": True,
        "val_test_augmentation": "none_random_augmentation"
    },
    "created_at": datetime.now().isoformat(timespec="seconds")
}

(out_dir / "e3b_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
print("Saved:", out_dir / "e3b_config.json")
PY

echo "============================================================"
echo "E3b finished at: $(date)"
echo "Prediction dir: ${PRED_DIR}"
echo "============================================================"
