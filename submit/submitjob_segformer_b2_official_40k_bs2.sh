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

mkdir -p runs/logs .cache/huggingface
export HF_HOME=/share/home/u2515283028/caries_project/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

SEGFORMER_ASSET_DIR=/share/home/u2515283028/caries_project/external_assets/nvidia_mit_b2
if [[ -f "${SEGFORMER_ASSET_DIR}/config.json" && \
      -f "${SEGFORMER_ASSET_DIR}/pytorch_model.bin" ]]; then
  SEGFORMER_PRETRAINED_SOURCE=${SEGFORMER_ASSET_DIR}
else
  SEGFORMER_PRETRAINED_SOURCE=nvidia/mit-b2
fi

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

echo "SegFormer pretrained source: ${SEGFORMER_PRETRAINED_SOURCE}"

python -m py_compile \
  scripts/setup/prepare_official_baseline_weights.py \
  src/official_baseline_common.py \
  src/train_segformer_b2_official.py

python scripts/setup/prepare_official_baseline_weights.py \
  --model segformer \
  --verify-only

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
  --pretrained-model-name "${SEGFORMER_PRETRAINED_SOURCE}" \
  --local-files-only \
  --num-workers 2 \
  --seed 42

cat runs/segformer_b2_official_40k_bs2/test_metrics.json
cat runs/segformer_b2_official_40k_bs2/summary_metrics.json
