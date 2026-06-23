#!/bin/bash
#SBATCH -J b9_dcf_lite
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o runs/logs/b9_dcf_lite_%j.out

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

python -m py_compile src/train_resnet34_unet_brr_b9_dcf_e200.py

echo "===== RUN B9-DCF-lite: B8B ctx64 + DCF dcf16 ====="
python src/train_resnet34_unet_brr_b9_dcf_e200.py \
  --run-name b9_dcf_lite_ctx64_auglite_e200_constlr_bs6 \
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
  --context-branch-channels 64 \
  --context-dropout 0.0 \
  --context-init-scale 0.10 \
  --dcf-channels 16 \
  --dcf-beta 0.20

echo "===== DONE B9-DCF-lite ====="
