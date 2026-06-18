#!/bin/bash
#SBATCH -J resenc_plan
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/nnunet_resenc_plan_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet-caries

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

export nnUNet_raw=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_raw
export nnUNet_preprocessed=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_preprocessed
export nnUNet_results=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_results

echo "===== nnU-Net paths ====="
echo "nnUNet_raw=$nnUNet_raw"
echo "nnUNet_preprocessed=$nnUNet_preprocessed"
echo "nnUNet_results=$nnUNet_results"

echo
echo "===== Plan and preprocess ResEncUNetPlanner 2D for Dataset701_CariXray ====="

nnUNetv2_plan_and_preprocess \
  -d 701 \
  -pl ResEncUNetPlanner \
  -c 2d \
  -np 8 \
  -overwrite_plans_name nnUNetResEncUNetPlans

echo
echo "===== Generated plans ====="
find "$nnUNet_preprocessed/Dataset701_CariXray" -maxdepth 1 -type f -name "*Plans.json" -print -exec ls -lh {} \;

echo
echo "===== Generated preprocessed dirs ====="
find "$nnUNet_preprocessed/Dataset701_CariXray" -maxdepth 1 -type d -print

echo
echo "ResEnc planning/preprocessing done."
