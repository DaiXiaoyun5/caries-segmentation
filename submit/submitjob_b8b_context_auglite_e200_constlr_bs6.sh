#!/bin/bash
#SBATCH -J b8b_ctx
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b8b_context_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

echo "===== ENV ====="
which python
python --version
nvidia-smi

python -m py_compile src/train_resnet34_unet_brr_b8b_context_e200.py

echo "===== RUN B8B: B2 + Multi-scale Context Bottleneck ====="
python src/train_resnet34_unet_brr_b8b_context_e200.py \
  --run-name b8b_b2_context_bottleneck_auglite_e200_constlr_bs6 \
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
  --context-branch-channels 128 \
  --context-dropout 0.0 \
  --context-init-scale 0.10

echo "===== DONE B8B-context ====="
