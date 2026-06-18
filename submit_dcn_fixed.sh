#!/bin/bash
#SBATCH -J u3p_dcn_fix
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1                  # 【防坑修复】必须为1！绝对不能填4，否则会导致4个代码副本同时抢显卡(OOM)
#SBATCH -c 4                  # 分配4个CPU核心给数据读取进程
#SBATCH --gres=gpu:1          # 申请1张GPU
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/dcn_fix_%j.out

# ================================
# 1. 加载平台预装的 Anaconda 模块
# ================================
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

# ================================
# 2. 注入内部网络代理 (极度关键)
# 防止预训练权重下载时报错 Network Unreachable
# ================================
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

# ================================
# 3. 切换到项目绝对路径
# ================================
cd /share/home/u2515283028/caries_project

# ================================
# 4. 运行训练代码 (使用 srun)
# ================================
srun python src/train_resnet34_unet3p_edge_dcn_fixed.py \
  --run-name unet3p_edge_dcn_v2 \
  --epochs 50 \
  --batch-size 6 \
  --num-workers 4
