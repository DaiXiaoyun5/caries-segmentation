#!/bin/bash
#SBATCH -J attunet_e260
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/attention_unet_r34_e260_%j.out

set -e

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

echo "ERROR: this ResNet34 attention-gate run is not canonical Attention U-Net."
echo "Use: sbatch submit/submitjob_attention_unet_official_e1000_bs2.sh"
exit 2

mkdir -p runs/logs

RUN_NAME=attention_unet_r34_auglite_e260_constlr_bs6

echo "===== ENV ====="
which python
python --version
nvidia-smi

echo "===== PY COMPILE ====="
python -m py_compile \
  src/train_comparison_baseline_common.py \
  src/train_attention_unet_r34_auglite_e260.py \
  src/eval_per_case_metrics.py

echo "===== RUN ATTENTION U-NET R34: E260, lr=3e-4 ====="
python src/train_attention_unet_r34_auglite_e260.py \
  --run-name ${RUN_NAME} \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 6 \
  --epochs 260 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 42 \
  --encoder-name resnet34 \
  --encoder-weights imagenet \
  --decoder-channels 256,128,64,32,16

echo "===== PER-CASE EVAL ATTENTION U-NET ====="
python src/eval_per_case_metrics.py \
  --run-name ${RUN_NAME} \
  --module-name train_attention_unet_r34_auglite_e260 \
  --class-name AttentionUNetResNet34 \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/${RUN_NAME}/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo "===== DONE ATTENTION U-NET R34 ====="
