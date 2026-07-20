# U-Mamba 与 VainF DeepLabV3+ 后续实验操作说明

本文档包含两个 U-Mamba 离线 CUDA wheel 的安装方法，以及两套新
DeepLabV3+ 实验的运行方法。本次不需要 smoke test。执行下列命令前，请先在
超算上拉取并切换到本次实验分支，或者在 PR 合并后拉取最新的 `main` 分支。

## 1. 把 U-Mamba 文件放到指定目录

如果源码压缩包和两个 wheel 被上传到了项目根目录，请执行：

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

mkdir -p external_assets/umamba_wheels
mv -n U-Mamba-main.zip external_assets/U-Mamba-main.zip
mv -n causal_conv1d-1.2.0.post2+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl external_assets/umamba_wheels/
mv -n mamba_ssm-1.2.0.post1+cu118torch2.0cxx11abiFALSE-cp310-cp310-linux_x86_64.whl external_assets/umamba_wheels/
```

安装脚本要求使用上述精确文件名。这两个 wheel 对应 CPython 3.10、CUDA
11.8、torch 2.0 和旧版 CXX ABI。wheel、压缩包、解压后的第三方仓库、Conda
环境、数据集和 checkpoint 都会继续被 Git 忽略，不会上传到 GitHub。

## 2. 创建独立的 U-Mamba 环境

请在登录节点执行下面的命令。登录节点没有 GPU，因此此时出现
`torch.cuda.is_available() == False` 属于正常现象，不代表安装失败。正式 GPU
作业会在训练前执行真实的 CUDA 前向传播检查。

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

bash scripts/setup/setup_umamba_official_env.sh
```

只有安装脚本最后出现以下内容，才说明环境创建完成：

```text
U-Mamba official environment is ready: umamba-caries
```

## 3. 先预处理 U-Mamba 数据，再训练 fold 0

首先提交数据规划和预处理作业：

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_umamba_prepare_caries_2d.sh
```

等待该 Slurm 作业成功结束后，再提交正式训练：

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_umamba_bot_official_2d_fold0.sh
```

训练结束后，用于论文分析的小型结果文件会导出到
`runs/umamba_bot_official_2d_fold0/`。体积较大的官方 checkpoint 仍保存在独立的
U-Mamba results 目录中，不会复制到 Git。

## 4. 准备 VainF DeepLabV3+ 的 ResNet50 预训练权重缓存

下面的命令只需要在登录节点执行一次。它只处理 DeepLabV3+，因此即使
SegFormer 的本地 `config.json` 哈希仍有问题，也不会阻塞 DeepLabV3+ 权重准备。

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train
conda activate caries-baselines

export PYTHONNOUSERSITE=1
export TORCH_HOME=/share/home/u2515283028/caries_project/.cache/torch
python scripts/setup/prepare_official_baseline_weights.py \
  --model deeplab
python scripts/setup/prepare_official_baseline_weights.py \
  --model deeplab \
  --verify-only
```

两个命令都应该输出：

```text
DeepLabV3+ ResNet50 cache: OK
```

生成的资产清单必须记录 VainF 网络、output stride 16，以及 ASPP dilation
`6/12/18`。

## 5. 提交两套 DeepLabV3+ 实验

第一套是严格遵循 VainF 参考训练策略的版本：

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_deeplabv3plus_vainf_r50_os16_official_30k_bs4.sh
```

第二套是针对 CariXray 医学分割任务适配的版本：

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_deeplabv3plus_vainf_r50_os16_medical_e260_bs6.sh
```

两套实验使用完全相同的 VainF ResNet50、OS16、ASPP `6/12/18` 网络结构。
严格版用于复现公认参考训练方案；医学适配版用于判断此前 DeepLabV3+ 表现较差
是否主要由训练策略不适合小病灶医学分割造成。

预期结果目录如下：

```text
runs/deeplabv3plus_vainf_r50_os16_official_30k_bs4/
runs/deeplabv3plus_vainf_r50_os16_medical_auglite_e260_bs6/
```

每个目录都会包含 `config.json`、`history.csv`、`best_train_metrics.json`、
`best_val_metrics.json`、`test_metrics.json`、`summary_metrics.json`、逐病例指标、
运行状态记录和预测可视化。

历史实验 `deeplabv3plus_r50_os16_30k_bs4` 使用的是 SMP 实现和 ASPP
`12/24/36`，现已被新实验替代。旧结果保留用于审计和问题追踪，但不能再作为
论文最终主对比表中的 DeepLabV3+ 结果。
