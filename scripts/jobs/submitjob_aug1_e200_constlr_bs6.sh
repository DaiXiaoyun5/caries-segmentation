#!/bin/bash
#SBATCH -J aug1_e200_const
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/aug1_e200_const_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

RUN_NAME=resunet_cbsa_mkdc_auglite_e200_constlr_bs6

if [ -d "runs/${RUN_NAME}" ]; then
  echo "ERROR: runs/${RUN_NAME} already exists. Rename or remove it first."
  exit 1
fi

echo "===== Train AUG1-E200-ConstLR: AugLite + 200 epochs + constant lr ====="
python src/train_resunet_cbsa_mkdc_auglite_e200_constlr.py \
  --run-name ${RUN_NAME} \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 6 \
  --epochs 200 \
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

echo "===== Eval per-case for AUG1-E200-ConstLR ====="
python src/eval_per_case_metrics.py \
  --run-name ${RUN_NAME} \
  --module-name train_resunet_cbsa_mkdc_auglite_e200_constlr \
  --class-name ResNet34UNetCBSAMKDC \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== AUG1-E200-ConstLR test_metrics.json ====="
cat runs/${RUN_NAME}/test_metrics.json

echo
echo "===== AUG1-E200-ConstLR per_case_metrics_summary.json ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo
echo "AUG1-E200-ConstLR done. Run directory: runs/${RUN_NAME}"
