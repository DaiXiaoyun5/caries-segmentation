# Caries Segmentation

龋齿 X 光图像分割实验仓库。项目以二值语义分割为主，包含经典基线、ResNet34 编码器模型、UNet3+ 系列、边界感知与注意力模块消融，以及 nnU-Net、MedSAM 和 YOLO 对照实验。

本仓库最初用于 Slurm GPU 集群，现有训练脚本默认项目目录为：

```text
/share/home/u2515283028/caries_project
```

如果部署到其他位置，请先替换 Python 和 Shell 脚本中的该路径。Shell 脚本已按用途收纳到 `scripts/`，但文件名保持不变。

## 项目结构

```text
.
├── src/                     # 数据处理、训练、评估、分析脚本
├── scripts/
│   ├── jobs/                # Slurm 训练、评估和冒烟任务
│   ├── tools/               # 检查、补丁生成和结果打包工具
│   └── setup/               # 依赖安装、下载和外部模型准备
├── requirements.txt         # 核心 Python 依赖
└── docs/PROJECT_GUIDE.md    # 模型族、数据流和脚本导航
```

更完整的项目导航见 [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md)。

## 数据约定

主要训练脚本读取制表符分隔的 split 文件：

```text
/absolute/path/to/image.jpg<TAB>/absolute/path/to/mask.png
```

默认目录布局：

```text
data/
├── unzip/
│   ├── images/              # 原始 JPG 图像
│   └── labels/              # 原始标注（部分准备脚本使用）
├── masks/
│   └── all/                 # 二值 PNG mask
└── splits/
    ├── train.txt
    ├── val.txt
    └── test.txt
```

Mask 在训练时按 `> 0` 转为前景。`src/make_splits.py` 根据文件名前缀生成 train/val/test 列表。

## 环境

推荐使用 Linux、CUDA GPU 和 Python 3.9+：

```bash
conda create -n caries-train python=3.10 -y
conda activate caries-train

# 根据集群 CUDA 版本安装匹配的 PyTorch 后：
pip install -r requirements.txt
```

PyTorch 应从其官方渠道选择与 CUDA 驱动匹配的版本，因此未在 `requirements.txt` 中固定版本。nnU-Net、MedSAM、Ultralytics 和 RWKV 相关实验属于可选扩展，需要按对应脚本单独安装。

## 快速检查

在运行训练前，建议依次检查数据：

```bash
python src/check_pairs.py
python src/make_all_masks.py
python src/make_splits.py
python src/dataset_smoke_test.py
```

注意：这些脚本同样使用固定的 `PROJECT_ROOT`，运行前应确认路径。

## 训练示例

### 直接运行

```bash
python src/train_resnet34_unet3p.py \
  --run-name smoke_resnet34_unet3p \
  --image-size 512 \
  --batch-size 2 \
  --epochs 1 \
  --num-workers 0
```

### Slurm 集群

```bash
sbatch scripts/jobs/submitjob_resnet34_unet_auglite_e200_constlr_bs6.sh
```

任务脚本通常完成训练、逐病例评估以及结果摘要输出。请先检查分区名、Conda 环境、GPU 配额和日志路径是否适合当前集群。

## 输出

大多数实验写入：

```text
runs/<run-name>/
├── checkpoints/
│   ├── best.pth
│   └── last.pth
├── config.json
├── history.csv
├── test_metrics.json
├── per_case_metrics.csv
├── per_case_metrics_summary.json
└── preds*/
```

模型权重、数据集、日志和大体积结果已在 `.gitignore` 中排除，不应提交到 Git。

## 评估与分析

常用入口：

```bash
# 通用逐病例评估
python src/eval_per_case_metrics.py \
  --run-name <run-name> \
  --module-name <training-module> \
  --class-name <model-class>

# 最终模型复杂度
python src/measure_final_model_complexity.py

# 最终可视化病例
python src/make_final_visual_cases.py

# 论文证据汇总
python src/analyze_final_paper_evidence.py
```

## 当前注意事项

- 多个脚本存在硬编码的集群绝对路径，迁移环境时必须统一修改。
- `src/` 中部分模型通过同目录模块互相导入，应从项目根目录运行脚本。
- 少量早期中文注释存在历史编码乱码，不影响 Python 语法，但后续维护时建议逐步修复。
- 仓库包含大量实验快照和消融版本；不要仅凭文件名删除“重复”脚本，它们可能对应已完成实验。
- 本项目用于研究实验，不应直接用于临床诊断。
