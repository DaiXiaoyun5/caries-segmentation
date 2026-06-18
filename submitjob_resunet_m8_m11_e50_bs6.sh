#!/bin/bash
#SBATCH -J resunet_m8_m11
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/resunet_m8_m11_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

run_train_eval () {
  VARIANT="$1"
  RUN_NAME="$2"
  CLASS_NAME="$3"

  echo
  echo "================================================================"
  echo "Train ${RUN_NAME}"
  echo "variant=${VARIANT}"
  echo "class=${CLASS_NAME}"
  echo "================================================================"

  python src/train_resunet_skip_cbsa_blaspp_boundary.py \
    --variant "${VARIANT}" \
    --run-name "${RUN_NAME}" \
    --image-size 512 \
    --batch-size 6 \
    --epochs 50 \
    --lr 3e-4 \
    --weight-decay 1e-4 \
    --num-workers 2 \
    --encoder-name resnet34 \
    --encoder-weights imagenet \
    --cbsa-lambda 1e-4 \
    --cbsa-alpha 1.0 \
    --cbsa-beta 0.1 \
    --laspp-mid-channels 128 \
    --laspp-dropout 0.1 \
    --edge-weight 0.2

  echo
  echo "================================================================"
  echo "Eval per-case ${RUN_NAME}"
  echo "================================================================"

  python src/eval_per_case_metrics.py \
    --run-name "${RUN_NAME}" \
    --module-name train_resunet_skip_cbsa_blaspp_boundary \
    --class-name "${CLASS_NAME}" \
    --batch-size 4 \
    --num-workers 2 \
    --save-preds 0 \
    --save-prob 0

  echo
  echo "===== ${RUN_NAME} test_metrics.json ====="
  cat "runs/${RUN_NAME}/test_metrics.json"

  echo
  echo "===== ${RUN_NAME} per_case_metrics_summary.json ====="
  cat "runs/${RUN_NAME}/per_case_metrics_summary.json"
  echo
}

run_train_eval \
  "m8_skip_cbsa" \
  "resunet_skipcbsa_e50_bs6" \
  "ResNet34UNetSkipCBSA"

run_train_eval \
  "m9_bottleneck_laspp" \
  "resunet_blaspp_e50_bs6" \
  "ResNet34UNetBottleneckLiteASPP"

run_train_eval \
  "m10_skip_cbsa_bottleneck_laspp" \
  "resunet_skipcbsa_blaspp_e50_bs6" \
  "ResNet34UNetSkipCBSABottleneckLiteASPP"

run_train_eval \
  "m11_skip_cbsa_bottleneck_laspp_boundary" \
  "resunet_skipcbsa_blaspp_boundary_e50_bs6" \
  "ResNet34UNetSkipCBSABottleneckLiteASPPBoundary"

echo
echo "All M8-M11 experiments finished."
