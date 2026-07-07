#!/bin/bash
#SBATCH -J b33_lglb
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b33_lglb_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b32_cglb_e200.py src/train_resnet34_unet_brr_b33_lglb_e200.py src/eval_per_case_metrics.py

echo "===== RUN B33: B2 + Local-Global-Local Bottleneck ====="
python src/train_resnet34_unet_brr_b33_lglb_e200.py \
  --run-name b33_lglb_auglite_e200_constlr_bs6 \
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
  --lglb-hidden-channels 128 \
  --lglb-local-kernel 5 \
  --lglb-global-kernel 7 \
  --lglb-global-dilation 3 \
  --lglb-dropout 0.0 \
  --lglb-scale-init 0.05 \
  --lglb-scale-max 0.30

echo "===== PER-CASE EVAL B33 ====="
python src/eval_per_case_metrics.py \
  --run-name b33_lglb_auglite_e200_constlr_bs6 \
  --module-name train_resnet34_unet_brr_b33_lglb_e200 \
  --class-name ResNet34UNetBRRB33LGLB \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/b33_lglb_auglite_e200_constlr_bs6/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/b33_lglb_auglite_e200_constlr_bs6/per_case_metrics_summary.json

echo "===== DONE B33 LGLB ====="
