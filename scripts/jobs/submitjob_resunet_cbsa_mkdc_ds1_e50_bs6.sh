#!/bin/bash
#SBATCH -J p2_ds1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/p2_ds1_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

RUN_NAME=resunet_cbsa_mkdc_ds1_e50_bs6

if [ -d "runs/${RUN_NAME}" ]; then
  echo "ERROR: runs/${RUN_NAME} already exists. Rename or remove it first."
  exit 1
fi

echo "===== Train DS1: P2 + Deep Supervision ====="
python src/train_resunet_cbsa_mkdc_ds1.py \
  --run-name ${RUN_NAME} \
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
  --mk-expansion-factor 2 \
  --mk-kernel-sizes 1,3,5 \
  --mk-depth 1 \
  --mk-add \
  --mk-activation relu6 \
  --aux256-weight 0.5 \
  --aux128-weight 0.25

echo "===== Eval per-case for DS1 ====="
python src/eval_per_case_metrics.py \
  --run-name ${RUN_NAME} \
  --module-name train_resunet_cbsa_mkdc_ds1 \
  --class-name ResNet34UNetDS1CBSAMKDC \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== DS1 test_metrics.json ====="
cat runs/${RUN_NAME}/test_metrics.json

echo
echo "===== DS1 per_case_metrics_summary.json ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo
echo "DS1 done. Run directory: runs/${RUN_NAME}"
