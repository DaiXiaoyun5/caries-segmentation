#!/bin/bash
#SBATCH -J fg1_sw_eval
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/fg1_sw_eval_%j.out

set -e

module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cd /share/home/u2515283028/caries_project
mkdir -p runs/logs

RUN_NAME=resunet_cbsa_mkdc_fg1_crop_e80_bs6

if [ ! -d "runs/${RUN_NAME}" ]; then
  echo "ERROR: runs/${RUN_NAME} does not exist. Please train FG1 first."
  exit 1
fi

if [ ! -f "runs/${RUN_NAME}/checkpoints/best.pth" ]; then
  echo "ERROR: runs/${RUN_NAME}/checkpoints/best.pth does not exist."
  exit 1
fi

echo "===== Sliding-window eval for FG1 ====="
python src/eval_fg1_sliding_window_sw640.py \
  --run-name ${RUN_NAME} \
  --split test \
  --eval-size 640 \
  --patch-size 512 \
  --stride 128 \
  --batch-size 4 \
  --threshold 0.5

echo
echo "===== FG1 sliding-window summary ====="
cat runs/${RUN_NAME}/sliding_window_summary_sw640_p512_s128_test.json

echo
echo "FG1 sliding-window eval done."
