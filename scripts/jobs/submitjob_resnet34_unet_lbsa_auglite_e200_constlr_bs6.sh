#!/bin/bash
#SBATCH -J r34u_lbsa_e200_c
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/r34u_lbsa_e200_const_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

RUN_NAME=resnet34_unet_lbsa_auglite_e200_constlr_bs6

if [ -d "runs/${RUN_NAME}" ]; then
  echo "ERROR: runs/${RUN_NAME} already exists. Rename or remove it first."
  exit 1
fi

echo "===== Train N2 ResNet34-U-Net + LBSA + AugLite E200 ConstLR ====="
python src/train_resnet34_unet_lbsa_auglite_e200_constlr.py   --run-name ${RUN_NAME}   --image-size 512   --augment-profile lite   --batch-size 6   --epochs 200   --lr 3e-4   --weight-decay 1e-4   --num-workers 2   --encoder-name resnet34   --encoder-weights imagenet   --lbsa-lambda 1e-4   --lbsa-alpha 1.0   --lbsa-beta 0.1

echo "===== Eval per-case for N2 ResNet34-U-Net + LBSA + AugLite E200 ConstLR ====="
python src/eval_per_case_metrics.py   --run-name ${RUN_NAME}   --module-name train_resnet34_unet_lbsa_auglite_e200_constlr   --class-name ResNet34UNetLBSA   --batch-size 4   --num-workers 2   --save-preds 0   --save-prob 0

echo "===== N2 ResNet34-U-Net LBSA AugLite E200 ConstLR test_metrics.json ====="
cat runs/${RUN_NAME}/test_metrics.json

echo
echo "===== N2 ResNet34-U-Net LBSA AugLite E200 ConstLR per_case_metrics_summary.json ====="
cat runs/${RUN_NAME}/per_case_metrics_summary.json

echo
echo "N2 ResNet34-U-Net LBSA AugLite E200 ConstLR done. Run directory: runs/${RUN_NAME}"
