#!/bin/bash
#SBATCH -J per_case_top_cbsa
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/per_case_top_cbsa_%j.out

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

echo "===== 1. CBSA + Lite-ASPP + BoundaryLoss ====="
srun python src/eval_per_case_metrics.py \
  --run-name resnet34_unet3p_cbsa_laspp_boundaryloss_e50_bs6 \
  --module-name train_resnet34_unet3p_cbsa_laspp \
  --class-name ResNet34UNet3PlusCBSALiteASPP \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0

echo "===== 2. CBSA + Lite-ASPP + CSFTL ====="
srun python src/eval_per_case_metrics.py \
  --run-name resnet34_unet3p_cbsa_laspp_csftl_e50_bs6 \
  --module-name train_resnet34_unet3p_cbsa_laspp \
  --class-name ResNet34UNet3PlusCBSALiteASPP \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0

echo "===== 3. CBSA + Lite-ASPP ====="
srun python src/eval_per_case_metrics.py \
  --run-name resnet34_unet3p_cbsa_laspp_e50_bs6 \
  --module-name train_resnet34_unet3p_cbsa_laspp \
  --class-name ResNet34UNet3PlusCBSALiteASPP \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0

echo "===== 4. CBSA + PPM ====="
srun python src/eval_per_case_metrics.py \
  --run-name resnet34_unet3p_cbsa_ppm_e50_bs6 \
  --module-name train_resnet34_unet3p_cbsa_ppm \
  --class-name ResNet34UNet3PlusCBSAPPMLite \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0

echo "===== 5. CBSA + OCR-Lite ====="
srun python src/eval_per_case_metrics.py \
  --run-name resnet34_unet3p_cbsa_ocr_lite_e50_bs6 \
  --module-name train_resnet34_unet3p_cbsa_ocr_lite \
  --class-name ResNet34UNet3PlusCBSAOCRLite \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0

echo "===== 6. CBSA ====="
srun python src/eval_per_case_metrics.py \
  --run-name resnet34_unet3p_cbsa_e50_bs6 \
  --module-name train_resnet34_unet3p_cbsa \
  --class-name ResNet34UNet3PlusCBSA \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0

echo "All top CBSA per-case evaluations finished."
