#!/bin/bash
#SBATCH -J final_visual
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/final_visual_%j.out

set -e
PROJECT_ROOT="/share/home/u2515283028/caries_project"
cd "$PROJECT_ROOT"
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

# 先确保最终分组文件已经生成
if [ ! -f runs/group_analysis/top_cbsa_final/difficulty_groups_by_resnet34_unet3p.csv ]; then
  echo "[INFO] 未发现最终分组文件，先运行 analyze_final_paper_evidence.py"
  python src/analyze_final_paper_evidence.py
fi

srun python src/make_final_visual_cases.py --num-each 2 --threshold 0.5 --image-size 512
