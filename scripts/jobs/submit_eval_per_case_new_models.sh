#!/bin/bash
#SBATCH -J percase_new
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/per_case_new_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

run_eval () {
  RUN_NAME=$1
  MODULE_NAME=$2
  CLASS_NAME=$3

  echo "============================================================"
  echo "Evaluating ${RUN_NAME}"
  echo "Module: ${MODULE_NAME}"
  echo "Class : ${CLASS_NAME}"
  echo "============================================================"

  python src/eval_per_case_metrics.py \
    --run-name "${RUN_NAME}" \
    --module-name "${MODULE_NAME}" \
    --class-name "${CLASS_NAME}" \
    --batch-size 4 \
    --num-workers 2 \
    --save-preds 1 \
    --save-prob 0

  cat "runs/${RUN_NAME}/per_case_metrics_summary.json"
}

run_eval \
  "resnet34_unet3p_brm_e50_bs6" \
  "train_resnet34_unet3p_brm" \
  "ResNet34UNet3PlusBRM"

run_eval \
  "resnet34_unet3p_cdfe_e50_bs6" \
  "train_resnet34_unet3p_cdfe" \
  "ResNet34UNet3PlusCDFE"

run_eval \
  "resnet34_unet3p_edge_brm_e50_bs6" \
  "train_resnet34_unet3p_edge_brm" \
  "ResNet34UNet3PlusEdgeBRM"

echo "All per-case evaluations finished."
