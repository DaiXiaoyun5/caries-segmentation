#!/bin/bash
#SBATCH -J b24_wtconv
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b24_wtconv_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b24_wtconv_context_e200.py src/eval_per_case_metrics.py

echo "===== RUN B24: B2 + WTConv-lite Bottleneck Context ====="
python src/train_resnet34_unet_brr_b24_wtconv_context_e200.py \
  --run-name b24_wtconv_context_auglite_e200_constlr_bs6 \
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
  --wt-kernel-size 5 \
  --wt-levels 2 \
  --wt-init-scale 0.05 \
  --wt-wavelet-init-scale 0.10

echo "===== PER-CASE EVAL B24 ====="
python src/eval_per_case_metrics.py \
  --run-name b24_wtconv_context_auglite_e200_constlr_bs6 \
  --module-name train_resnet34_unet_brr_b24_wtconv_context_e200 \
  --class-name ResNet34UNetBRRB24WTConvContext \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/b24_wtconv_context_auglite_e200_constlr_bs6/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/b24_wtconv_context_auglite_e200_constlr_bs6/per_case_metrics_summary.json

echo "===== DONE B24 WTCONV CONTEXT ====="
