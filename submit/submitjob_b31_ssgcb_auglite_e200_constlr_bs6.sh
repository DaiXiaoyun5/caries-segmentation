#!/bin/bash
#SBATCH -J b31_ssgcb
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b31_ssgcb_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b30_ssgcb_light_e200.py src/train_resnet34_unet_brr_b31_ssgcb_e200.py src/eval_per_case_metrics.py

echo "===== RUN B31: B2 + stronger SSGCB dynamic bottleneck context ====="
python src/train_resnet34_unet_brr_b31_ssgcb_e200.py \
  --run-name b31_ssgcb_auglite_e200_constlr_bs6 \
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
  --ssgcb-branch-channels 128 \
  --ssgcb-dilations 1,3,5 \
  --ssgcb-dropout 0.0 \
  --ssgcb-scale-init 0.10 \
  --ssgcb-scale-max 0.50

echo "===== PER-CASE EVAL B31 ====="
python src/eval_per_case_metrics.py \
  --run-name b31_ssgcb_auglite_e200_constlr_bs6 \
  --module-name train_resnet34_unet_brr_b31_ssgcb_e200 \
  --class-name ResNet34UNetBRRB31SSGCB \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/b31_ssgcb_auglite_e200_constlr_bs6/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/b31_ssgcb_auglite_e200_constlr_bs6/per_case_metrics_summary.json

echo "===== DONE B31 SSGCB ====="
