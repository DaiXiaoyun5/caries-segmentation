#!/bin/bash
#SBATCH -J dlv3p_med260
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/deeplabv3plus_vainf_medical_%j.out
#SBATCH -e runs/logs/deeplabv3plus_vainf_medical_%j.out

set -Eeuo pipefail

cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
conda activate caries-baselines
export PYTHONNOUSERSITE=1

mkdir -p runs/logs .cache/torch
export TORCH_HOME=/share/home/u2515283028/caries_project/.cache/torch

echo "===== VAINF DEEPLABV3+ RESNET50 OS16 / MEDICAL E260 ====="
echo "Host: $(hostname)"
which python
python --version
nvidia-smi

python -m py_compile \
  scripts/setup/prepare_official_baseline_weights.py \
  scripts/validation/validate_deeplabv3plus_vainf.py \
  src/official_baseline_common.py \
  src/deeplabv3plus_vainf_common.py \
  src/train_deeplabv3plus_vainf_medical_e260.py

python scripts/setup/prepare_official_baseline_weights.py \
  --model deeplab \
  --verify-only

python scripts/validation/validate_deeplabv3plus_vainf.py --device cuda

python src/train_deeplabv3plus_vainf_medical_e260.py \
  --run-name deeplabv3plus_vainf_r50_os16_medical_auglite_e260_bs6 \
  --image-size 512 \
  --batch-size 6 \
  --val-batch-size 6 \
  --epochs 260 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --num-workers 2 \
  --seed 42

cat runs/deeplabv3plus_vainf_r50_os16_medical_auglite_e260_bs6/test_metrics.json
cat runs/deeplabv3plus_vainf_r50_os16_medical_auglite_e260_bs6/summary_metrics.json
