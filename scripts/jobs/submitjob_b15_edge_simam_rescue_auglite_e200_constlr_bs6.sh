#!/bin/bash
#SBATCH -J b15_esdr
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b15_esdr_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b15_edge_simam_rescue_e200.py src/eval_per_case_metrics.py

echo "===== RUN B15: B2 + Edge-SimAM Detail Rescue ====="
python src/train_resnet34_unet_brr_b15_edge_simam_rescue_e200.py \
  --run-name b15_edge_simam_rescue_auglite_e200_constlr_bs6 \
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
  --esdr-hidden-channels 16 \
  --esdr-max-delta 0.60 \
  --esdr-init-scale 0.10 \
  --esdr-bias-init -4.0 \
  --esdr-contrast-kernel 7 \
  --esdr-simam-lambda 1e-4

echo "===== PER-CASE EVAL B15 ====="
python src/eval_per_case_metrics.py \
  --run-name b15_edge_simam_rescue_auglite_e200_constlr_bs6 \
  --module-name train_resnet34_unet_brr_b15_edge_simam_rescue_e200 \
  --class-name ResNet34UNetBRRB15EdgeSimAMRescue \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/b15_edge_simam_rescue_auglite_e200_constlr_bs6/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/b15_edge_simam_rescue_auglite_e200_constlr_bs6/per_case_metrics_summary.json

echo "===== DONE B15 ESDR ====="
