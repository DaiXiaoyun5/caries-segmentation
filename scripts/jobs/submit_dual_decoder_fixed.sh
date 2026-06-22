#!/bin/bash
#SBATCH -J u3p_dual_fix
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1                  # 【核心防坑】绝对不写4，防止4个进程互相抢夺显卡导致OOM
#SBATCH -c 4                  # 分配4个CPU线程用于加载图片
#SBATCH --gres=gpu:1          # 申请1张GPU
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/dual_fix_%j.out

# ================================
# 1. 加载平台环境
# ================================
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

# ================================
# 2. 配置武汉纺织大学内部代理 (极度关键)
# 解决预训练权重下载时的 Network Unreachable 报错
# ================================
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

# ================================
# 3. 切换工作目录并运行
# ================================
cd /share/home/u2515283028/caries_project

srun python src/train_resnet34_unet3p_edge_dual_decoder_fixed.py \
  --run-name unet3p_edge_dual_decoder_v2 \
  --epochs 50 \
  --batch-size 6 \
  --num-workers 4
