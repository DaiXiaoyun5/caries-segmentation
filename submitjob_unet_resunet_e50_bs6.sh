#!/bin/bash
#SBATCH -J unet_resunet
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/unet_resunet_e50_bs6_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

echo "============================================================"
echo "Job started at: $(date)"
echo "Host: $(hostname)"
echo "Current dir: $(pwd)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "Task: train Plain U-Net and Res-UNet baselines"
echo "============================================================"

nvidia-smi

echo "============================================================"
echo "Check Python / CUDA / key package"
echo "============================================================"
python - <<'PY'
import torch
import segmentation_models_pytorch as smp
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda:", torch.version.cuda)
print("gpu count:", torch.cuda.device_count())
print("smp:", smp.__version__)
if torch.cuda.is_available():
    print("gpu name:", torch.cuda.get_device_name(0))
PY

echo "============================================================"
echo "Train plain U-Net e50 bs6"
echo "============================================================"
srun python src/train_unet_resunet_baselines.py \
  --run-name unet_plain_e50_bs6 \
  --model-type plain_unet \
  --epochs 50 \
  --batch-size 6 \
  --num-workers 2 \
  --image-size 512 \
  --lr 3e-4 \
  --base-channels 64

echo "============================================================"
echo "Train Res-UNet ResNet34 e50 bs6"
echo "============================================================"
srun python src/train_unet_resunet_baselines.py \
  --run-name resunet_resnet34_e50_bs6 \
  --model-type resunet \
  --epochs 50 \
  --batch-size 6 \
  --num-workers 2 \
  --image-size 512 \
  --lr 3e-4 \
  --encoder-name resnet34 \
  --encoder-weights imagenet

echo "============================================================"
echo "Final metrics"
echo "============================================================"
python - <<'PY'
import json
from pathlib import Path

root = Path("/share/home/u2515283028/caries_project")

for rn in ["unet_plain_e50_bs6", "resunet_resnet34_e50_bs6"]:
    p = root / "runs" / rn / "per_case_metrics_summary.json"
    print("\n===== " + rn + " =====")
    if p.exists():
        print(json.dumps(json.load(open(p)), indent=2))
    else:
        print("MISSING:", p)
PY

echo "============================================================"
echo "Job finished at: $(date)"
echo "============================================================"
