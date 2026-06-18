#!/bin/bash
#SBATCH -J resnet34_resdecoder_b1
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/resnet34_resdecoder_b1_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

RUN_NAME=resnet34_resdecoder_b1_e50_bs6

if [ -d "runs/${RUN_NAME}" ]; then
  echo "ERROR: runs/${RUN_NAME} already exists. Rename or remove it first."
  exit 1
fi

echo "===== Train B1: pretrained ResNet34 encoder + residual decoder ====="
python src/train_resnet34_resdecoder_b1.py \
  --run-name ${RUN_NAME} \
  --image-size 512 \
  --batch-size 6 \
  --epochs 50 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --decoder-channels 256,128,64,64,32

echo "===== Eval per-case for B1 ====="
python src/eval_per_case_metrics.py \
  --run-name ${RUN_NAME} \
  --module-name train_resnet34_resdecoder_b1 \
  --class-name ResNet34ResDecoderUNet \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== B1 test_metrics.json ====="
cat runs/${RUN_NAME}/test_metrics.json

echo
echo "===== B1 per_case_metrics_summary.json ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo
echo "B1 done. Run directory: runs/${RUN_NAME}"
