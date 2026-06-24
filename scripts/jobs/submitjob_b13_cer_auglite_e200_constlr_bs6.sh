#!/bin/bash
#SBATCH -J b13_cer
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b13_cer_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b13_cer_e200.py src/eval_per_case_metrics.py

echo "===== RUN B13: B2 + Candidate Expansion Refiner ====="
python src/train_resnet34_unet_brr_b13_cer_e200.py \
  --run-name b13_cer_auglite_e200_constlr_bs6 \
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
  --cer-hidden-channels 24 \
  --cer-max-delta 0.80 \
  --cer-init-scale 0.12 \
  --cer-bias-init -4.0 \
  --cer-highpass-kernel 7 \
  --cer-candidate-tau 0.15 \
  --cer-candidate-sharpness 12.0

echo "===== PER-CASE EVAL B13 ====="
python src/eval_per_case_metrics.py \
  --run-name b13_cer_auglite_e200_constlr_bs6 \
  --module-name train_resnet34_unet_brr_b13_cer_e200 \
  --class-name ResNet34UNetBRRB13CER \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/b13_cer_auglite_e200_constlr_bs6/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/b13_cer_auglite_e200_constlr_bs6/per_case_metrics_summary.json

echo "===== DONE B13 CER ====="
