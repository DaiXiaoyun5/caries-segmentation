#!/bin/bash
#SBATCH -J segformer_b2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/segformer_b2_e260_%j.out

set -e

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

mkdir -p runs/logs .cache/huggingface

export HF_HOME=/share/home/u2515283028/caries_project/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

RUN_NAME=segformer_b2_auglite_e260_constlr_bs6

echo "===== ENV ====="
which python
python --version
nvidia-smi

echo "===== DEPENDENCY CHECK ====="
python - <<'PY'
import transformers
print("transformers:", transformers.__version__)
from transformers import SegformerForSemanticSegmentation
print("SegFormer import: OK")
PY

echo "===== PY COMPILE ====="
python -m py_compile \
  src/train_comparison_baseline_common.py \
  src/train_segformer_b2_auglite_e260.py \
  src/eval_per_case_metrics.py

echo "===== RUN SEGFORMER-B2: E260, lr=3e-4 ====="
python src/train_segformer_b2_auglite_e260.py \
  --run-name ${RUN_NAME} \
  --image-size 512 \
  --augment-profile lite \
  --batch-size 6 \
  --epochs 260 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 42 \
  --encoder-weights imagenet \
  --pretrained-model-name nvidia/mit-b2 \
  --normalize-inputs 1

# The checkpoint is cached by the training process. Keep evaluation offline so
# it cannot silently fetch a different remote revision.
export HF_HUB_OFFLINE=1

echo "===== PER-CASE EVAL SEGFORMER-B2 ====="
python src/eval_per_case_metrics.py \
  --run-name ${RUN_NAME} \
  --module-name train_segformer_b2_auglite_e260 \
  --class-name SegFormerB2Binary \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0 \
  --save-prob 0

echo "===== TEST METRICS ====="
cat runs/${RUN_NAME}/test_metrics.json

echo "===== PER-CASE SUMMARY ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo "===== DONE SEGFORMER-B2 ====="
