#!/bin/bash
#SBATCH -J prep701
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/nnunet701_preprocess_only_%j.out

set -e

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate nnunet-caries

export nnUNet_raw=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_raw
export nnUNet_preprocessed=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_preprocessed
export nnUNet_results=/share/home/u2515283028/caries_project/nnunet_caries/nnUNet_results

echo "===== preprocess only Dataset701 2d ====="
echo "Start: $(date)"
echo "Host: $(hostname)"

rm -rf "${nnUNet_preprocessed}/Dataset701_CariXray/nnUNetPlans_2d"

nnUNetv2_plan_and_preprocess -d 701 -c 2d --verify_dataset_integrity

echo "===== check output ====="
find "${nnUNet_preprocessed}/Dataset701_CariXray/nnUNetPlans_2d" -name "*.b2nd" | wc -l
find "${nnUNet_preprocessed}/Dataset701_CariXray/nnUNetPlans_2d" -name "*.pkl" | wc -l

echo "Done: $(date)"
