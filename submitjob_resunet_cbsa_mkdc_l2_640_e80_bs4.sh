#!/bin/bash
#SBATCH -J resunet_l2_640
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/resunet_l2_640_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

RUN_NAME=resunet_cbsa_mkdc_l2_640_e80_bs4

if [ -d "runs/${RUN_NAME}" ]; then
  echo "ERROR: runs/${RUN_NAME} already exists. Rename or remove it first."
  exit 1
fi

echo "===== Train L2: R1-640 + BCE-Dice-FocalTversky ====="
python src/train_resunet_cbsa_mkdc_l2_640.py \
  --run-name ${RUN_NAME} \
  --image-size 640 \
  --batch-size 4 \
  --epochs 80 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --cbsa-lambda 1e-4 \
  --cbsa-alpha 1.0 \
  --cbsa-beta 0.1 \
  --mk-expansion-factor 2 \
  --mk-kernel-sizes 1,3,5 \
  --mk-depth 1 \
  --mk-add \
  --mk-activation relu6 \
  --bce-weight 0.3 \
  --dice-weight 0.3 \
  --ft-weight 0.4 \
  --ft-alpha 0.3 \
  --ft-beta 0.7 \
  --ft-gamma 0.75

echo "===== Eval per-case for L2 ====="
python src/eval_per_case_metrics.py \
  --run-name ${RUN_NAME} \
  --module-name train_resunet_cbsa_mkdc_l2_640 \
  --class-name ResNet34UNetCBSAMKDC \
  --batch-size 2 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== L2 test_metrics.json ====="
cat runs/${RUN_NAME}/test_metrics.json

echo
echo "===== L2 per_case_metrics_summary.json ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo
echo "L2 done. Run directory: runs/${RUN_NAME}"
