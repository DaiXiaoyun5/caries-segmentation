#!/bin/bash
#SBATCH -J percase4
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/per_case_main4_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project

echo "============================================================"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Current dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Task: per-case metrics for 4 main models"
echo "============================================================"

echo "GPU info:"
nvidia-smi

echo "============================================================"
echo "Check evaluation script"
echo "============================================================"
ls -lh src/eval_per_case_metrics.py

run_eval () {
  RUN_NAME=$1
  MODULE_NAME=$2
  CLASS_NAME=$3

  echo ""
  echo "============================================================"
  echo "Evaluating: ${RUN_NAME}"
  echo "Module    : ${MODULE_NAME}"
  echo "Class     : ${CLASS_NAME}"
  echo "Time      : $(date)"
  echo "============================================================"

  if [ ! -d "runs/${RUN_NAME}" ]; then
    echo "ERROR: runs/${RUN_NAME} not found"
    exit 1
  fi

  if [ ! -f "runs/${RUN_NAME}/config.json" ]; then
    echo "ERROR: runs/${RUN_NAME}/config.json not found"
    exit 1
  fi

  if [ ! -f "runs/${RUN_NAME}/checkpoints/best.pth" ]; then
    echo "ERROR: runs/${RUN_NAME}/checkpoints/best.pth not found"
    exit 1
  fi

  python src/eval_per_case_metrics.py \
    --run-name "${RUN_NAME}" \
    --module-name "${MODULE_NAME}" \
    --class-name "${CLASS_NAME}" \
    --batch-size 4 \
    --num-workers 2 \
    --save-preds 1 \
    --save-prob 0

  echo "Finished: ${RUN_NAME}"
  echo "Output CSV:"
  ls -lh "runs/${RUN_NAME}/per_case_metrics.csv"
  echo "Output summary:"
  cat "runs/${RUN_NAME}/per_case_metrics_summary.json"
}

run_eval \
  "resnet34_unet3p_baseline_e50" \
  "train_resnet34_unet3p" \
  "ResNet34UNet3Plus"

run_eval \
  "resnet34_unet3p_edge_e50" \
  "train_resnet34_unet3p_edge" \
  "ResNet34UNet3PlusEdge"

run_eval \
  "resnet34_unet3p_edge_simam_e50_bs6" \
  "train_resnet34_unet3p_edge_simam" \
  "ResNet34UNet3PlusEdgeSimAM"

run_eval \
  "resnet34_unet3p_edge_simam_ra_e50_bs6" \
  "train_resnet34_unet3p_edge_simam_ra" \
  "ResNet34UNet3PlusEdgeSimAMRA"

echo ""
echo "============================================================"
echo "All four per-case evaluations finished at: $(date)"
echo "============================================================"

echo "Final check:"
for RUN_NAME in \
  resnet34_unet3p_baseline_e50 \
  resnet34_unet3p_edge_e50 \
  resnet34_unet3p_edge_simam_e50_bs6 \
  resnet34_unet3p_edge_simam_ra_e50_bs6
do
  echo "------------------------------------------------------------"
  echo "${RUN_NAME}"
  if [ -f "runs/${RUN_NAME}/per_case_metrics_summary.json" ]; then
    cat "runs/${RUN_NAME}/per_case_metrics_summary.json" | grep -E "num_cases|mean_dice|mean_iou|mean_precision|mean_recall"
  else
    echo "Missing summary."
  fi
done

echo "============================================================"
echo "Job finished at: $(date)"
echo "============================================================"
