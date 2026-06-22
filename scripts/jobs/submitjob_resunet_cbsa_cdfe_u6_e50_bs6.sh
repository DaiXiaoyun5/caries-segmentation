#!/bin/bash
#SBATCH -J resunet_u6_cdfe
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/resunet_u6_cdfe_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

RUN_NAME=resunet_cbsa_cdfe_u6_e50_bs6

if [ -d "runs/${RUN_NAME}" ]; then
  echo "ERROR: runs/${RUN_NAME} already exists. Rename or remove it first."
  exit 1
fi

echo "===== Train U6: ResNet34-U-Net + CBSA + CDFE-lite ====="
python src/train_resunet_cbsa_cdfe_u6.py \
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
  --cdfe-reduction 4 \
  --cdfe-fg-thr 0.6 \
  --cdfe-bg-thr 0.3 \
  --cdfe-contrast-scale-init 0.1 \
  --cdfe-uncertain-scale-init 0.1 \
  --cdfe-detach-prob

echo "===== Eval per-case for U6 ====="
python src/eval_per_case_metrics.py \
  --run-name ${RUN_NAME} \
  --module-name train_resunet_cbsa_cdfe_u6 \
  --class-name ResNet34UNetCBSACDFE \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== U6 test_metrics.json ====="
cat runs/${RUN_NAME}/test_metrics.json

echo
echo "===== U6 per_case_metrics_summary.json ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo
echo "U6 done. Run directory: runs/${RUN_NAME}"
