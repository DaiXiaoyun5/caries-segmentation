#!/bin/bash
#SBATCH -J resunet_cbsa_m3
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/resunet_cbsa_m3_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

echo "===== Train M3: ResNet34-U-Net + CBSA ====="
python src/train_resunet_cbsa_m3.py \
  --run-name resunet_cbsa_e50_bs6 \
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
  --cbsa-beta 0.1

echo "===== Eval per-case for M3 ====="
python src/eval_per_case_metrics.py \
  --run-name resunet_cbsa_e50_bs6 \
  --module-name train_resunet_cbsa_m3 \
  --class-name ResNet34UNetCBSA \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== M3 test_metrics.json ====="
cat runs/resunet_cbsa_e50_bs6/test_metrics.json

echo
echo "===== M3 per_case_metrics_summary.json ====="
cat runs/resunet_cbsa_e50_bs6/per_case_metrics_summary.json

echo
echo "M3 done."
