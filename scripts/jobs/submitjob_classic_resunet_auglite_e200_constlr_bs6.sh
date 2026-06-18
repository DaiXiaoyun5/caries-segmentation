#!/bin/bash
#SBATCH -J resunet_aug_e200
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/classic_resunet_auglite_e200_const_%j.out

set -e
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

echo "===== Train Classic ResUNet + AugLite E200 ConstLR ====="
python src/train_aug_baseline_unet_resunet_e200.py \
  --model classic_resunet \
  --run-name classic_resunet_auglite_e200_constlr_bs6 \
  --image-size 512 \
  --epochs 200 \
  --batch-size 6 \
  --num-workers 2 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --seed 42 \
  --print-every 10

echo "Classic ResUNet + AugLite E200 done."
