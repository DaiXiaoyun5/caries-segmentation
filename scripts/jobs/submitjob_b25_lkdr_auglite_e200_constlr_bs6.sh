#!/bin/bash
#SBATCH -J b25_lkdr
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b25_lkdr_%j.out

set -e

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

mkdir -p runs/logs

echo "===== ENV ====="
which python
python --version
nvidia-smi

echo "===== PY COMPILE ====="
python -m py_compile src/train_resnet34_unet_brr_b25_lkdr_e200.py src/eval_per_case_metrics.py

echo "===== RUN B25: B2 + Large-Kernel Decoder Refiner ====="
python src/train_resnet34_unet_brr_b25_lkdr_e200.py \
  --run-name b25_lkdr_auglite_e200_constlr_bs6 \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 6 \
  --epochs 200 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --lkdr-hidden-channels 48 \
  --lkdr-expansion 3.0 \
  --lkdr-local-kernel 5 \
  --lkdr-context-kernel 7 \
  --lkdr-context-dilation 2 \
  --lkdr-init-scale 0.05 \
  --lkdr-max-scale 0.50

echo "===== PER-CASE EVAL B25 ====="
python src/eval_per_case_metrics.py \
  --run-name b25_lkdr_auglite_e200_constlr_bs6 \
  --module-name train_resnet34_unet_brr_b25_lkdr_e200 \
  --class-name ResNet34UNetBRRB25LKDR \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/b25_lkdr_auglite_e200_constlr_bs6/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/b25_lkdr_auglite_e200_constlr_bs6/per_case_metrics_summary.json

echo "===== DONE B25 LKDR ====="
