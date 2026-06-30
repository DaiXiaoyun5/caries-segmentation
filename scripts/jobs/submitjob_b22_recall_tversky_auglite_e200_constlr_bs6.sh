#!/bin/bash
#SBATCH -J b22_tversky
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b22_tversky_%j.out

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
python -m py_compile src/train_resnet34_unet_brr_b22_recall_tversky_e200.py src/eval_per_case_metrics.py

echo "===== RUN B22: B2 + Recall-biased Tversky Main Loss ====="
python src/train_resnet34_unet_brr_b22_recall_tversky_e200.py \
  --run-name b22_recall_tversky_auglite_e200_constlr_bs6 \
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
  --tversky-warmup-epochs 20 \
  --tversky-alpha-warmup 0.45 \
  --tversky-beta-warmup 0.55 \
  --tversky-alpha 0.40 \
  --tversky-beta 0.60 \
  --tversky-gamma 1.0 \
  --dice-weight 0.60 \
  --tversky-weight 0.40

echo "===== PER-CASE EVAL B22 ====="
python src/eval_per_case_metrics.py \
  --run-name b22_recall_tversky_auglite_e200_constlr_bs6 \
  --module-name train_resnet34_unet_brr_b22_recall_tversky_e200 \
  --class-name ResNet34UNetBRRB22RecallTversky \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/b22_recall_tversky_auglite_e200_constlr_bs6/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/b22_recall_tversky_auglite_e200_constlr_bs6/per_case_metrics_summary.json

echo "===== DONE B22 RECALL-TVERSKY ====="
