#!/bin/bash
#SBATCH -J resenc_f0
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/nnunet_resenc_train_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet-caries

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

export nnUNet_raw=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_raw
export nnUNet_preprocessed=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_preprocessed
export nnUNet_results=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_results

echo "===== GPU check ====="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda:", torch.version.cuda)
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo
echo "===== Train nnU-Net ResEncUNetPlanner 2D fold0 ====="

nnUNetv2_train 701 2d 0 \
  -p nnUNetResEncUNetPlans \
  -tr nnUNetTrainer \
  -device cuda

echo
echo "ResEnc training done."
