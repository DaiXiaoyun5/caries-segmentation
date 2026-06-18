#!/bin/bash
#SBATCH -J resunet_r1_640
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/resunet_r1_640_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

RUN_NAME=resunet_cbsa_mkdc_r1_640_e80_bs4

if [ -d "runs/${RUN_NAME}" ]; then
  echo "ERROR: runs/${RUN_NAME} already exists. Rename or remove it first."
  exit 1
fi

echo "===== Train R1: ResNet34-U-Net + CBSA + MKDC, 640 input, bs4, e80 ====="
python src/train_resunet_cbsa_mkdc_p2.py \
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
  --mk-activation relu6

echo "===== Eval per-case for R1 ====="
python src/eval_per_case_metrics.py \
  --run-name ${RUN_NAME} \
  --module-name train_resunet_cbsa_mkdc_p2 \
  --class-name ResNet34UNetCBSAMKDC \
  --batch-size 2 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== R1 test_metrics.json ====="
cat runs/${RUN_NAME}/test_metrics.json

echo
echo "===== R1 per_case_metrics_summary.json ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo
echo "R1 done. Run directory: runs/${RUN_NAME}"
