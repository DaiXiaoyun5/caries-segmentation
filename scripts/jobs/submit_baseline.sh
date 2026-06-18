#!/bin/bash
#SBATCH -J caries_unet3p
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -n 1                  # 【防坑设计】严格限制为1个主任务，防止代码被执行4次导致显存爆炸
#SBATCH -c 4                  # 为这1个任务分配4个CPU核心，用于数据加载
#SBATCH --gres=gpu:1          # 申请1张GPU
#SBATCH -o /share/home/u2515283028/caries_project/runs/logs/baseline_%j.out

# ================================
# 1. 加载平台预装的 Anaconda 模块
# ================================
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

# ================================
# 2. 注入内部网络代理 (极度关键)
# 确保 PyTorch 下载预训练权重 (resnet34 等) 时不报错
# ================================
export http_proxy=http://211.67.63.75:3128
export https_proxy=http://211.67.63.75:3128

# ================================
# 3. 切换到项目绝对路径
# ================================
cd /share/home/u2515283028/caries_project

# ================================
# 4. 运行训练代码
# ================================
srun python src/train_baseline.py
