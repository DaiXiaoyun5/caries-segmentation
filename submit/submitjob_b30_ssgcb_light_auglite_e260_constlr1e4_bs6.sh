#!/bin/bash
#SBATCH -J b30_e260_l1e4
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b30_ssgcb_light_e260_lr1e4_%j.out

set -e

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

mkdir -p runs/logs

RUN_NAME=b30_ssgcb_light_auglite_e260_constlr1e4_bs6

echo "===== ENV ====="
which python
python --version
nvidia-smi

echo "===== PY COMPILE ====="
python -m py_compile src/train_resnet34_unet_brr_b30_ssgcb_light_e200.py src/eval_per_case_metrics.py

echo "===== RUN B30-light E260: B2 + SSGCB, lr=1e-4 ====="
python src/train_resnet34_unet_brr_b30_ssgcb_light_e200.py \
  --run-name ${RUN_NAME} \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 6 \
  --epochs 260 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --ssgcb-branch-channels 96 \
  --ssgcb-dilations 1,3,5 \
  --ssgcb-dropout 0.0 \
  --ssgcb-scale-init 0.05 \
  --ssgcb-scale-max 0.30

echo "===== PER-CASE EVAL B30-light E260 LR1E-4 ====="
python src/eval_per_case_metrics.py \
  --run-name ${RUN_NAME} \
  --module-name train_resnet34_unet_brr_b30_ssgcb_light_e200 \
  --class-name ResNet34UNetBRRB30SSGCBLight \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/${RUN_NAME}/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo "===== DONE B30-light E260 LR1E-4 ====="
