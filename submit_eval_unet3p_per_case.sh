#!/bin/bash
#SBATCH -J eval_unet3p_pc
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/eval_unet3p_pc_%j.out

set -e

PROJECT_ROOT="/share/home/u2515283028/caries_project"
cd "$PROJECT_ROOT"
mkdir -p runs/logs

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

# 自动寻找你项目里已训练完成的原始 UNet3+ run。
# 如果你实际目录名不同，把它加到 CANDIDATES 第一行即可。
CANDIDATES=(
  "unet3p_baseline_e50_bs6_dp2_fair"
  "unet3p_baseline_e50_bs6_fair"
  "unet3p_baseline_e50"
  "unet3p_e50_bs6"
  "unet3p"
)

RUN_NAME=""
for r in "${CANDIDATES[@]}"; do
  if [ -f "runs/$r/config.json" ] && [ -f "runs/$r/checkpoints/best.pth" ]; then
    RUN_NAME="$r"
    break
  fi
done

if [ -z "$RUN_NAME" ]; then
  echo "[ERROR] 没找到 UNet3+ run 目录。请检查 runs/ 下实际名字。"
  echo "候选名如下："
  printf '  %s\n' "${CANDIDATES[@]}"
  echo
  echo "你可以执行：ls -d runs/*unet3p*"
  exit 1
fi

echo "Using UNet3+ run: $RUN_NAME"

if [ -f "runs/$RUN_NAME/per_case_metrics.csv" ]; then
  echo "[SKIP] runs/$RUN_NAME/per_case_metrics.csv 已存在，不重复评估。"
  exit 0
fi

srun python src/eval_per_case_metrics.py \
  --run-name "$RUN_NAME" \
  --module-name train_unet3p \
  --class-name UNet3Plus \
  --batch-size 4 \
  --num-workers 2 \
  --save-preds 0

echo "UNet3+ per-case evaluation finished: runs/$RUN_NAME/per_case_metrics.csv"
