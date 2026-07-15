#!/bin/bash
#SBATCH -J segformer_b2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/segformer_b2_official_%j.out
#SBATCH -e runs/logs/segformer_b2_official_%j.out

set -Eeuo pipefail

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
conda activate caries-baselines
export PYTHONNOUSERSITE=1

module load http-proxy 2>/dev/null || true
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

mkdir -p runs/logs .cache/huggingface
export HF_HOME=/share/home/u2515283028/caries_project/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

echo "===== SEGFORMER-B2 OFFICIAL RECIPE ====="
which python
python --version
nvidia-smi

python - <<'PY'
import transformers
print("transformers:", transformers.__version__)
from transformers import SegformerForSemanticSegmentation
print("SegFormer import: OK")
PY

python -m py_compile \
  src/official_baseline_common.py \
  src/train_segformer_b2_official.py

python src/train_segformer_b2_official.py \
  --run-name segformer_b2_official_40k_bs2 \
  --image-size 512 \
  --batch-size 2 \
  --val-batch-size 2 \
  --total-iters 40000 \
  --val-interval-iters 2000 \
  --lr 6e-5 \
  --head-lr-multiplier 10 \
  --weight-decay 0.01 \
  --warmup-iters 1500 \
  --warmup-ratio 1e-6 \
  --pretrained-model-name nvidia/mit-b2 \
  --num-workers 2 \
  --seed 42

cat runs/segformer_b2_official_40k_bs2/test_metrics.json
cat runs/segformer_b2_official_40k_bs2/summary_metrics.json
