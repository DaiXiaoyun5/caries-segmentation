# 项目导航

## 1. 实验主线

项目围绕龋齿小目标、边界模糊和类别不平衡问题，逐步扩展以下模型族：

| 模型族 | 代表脚本 | 用途 |
| --- | --- | --- |
| 经典基线 | `train_unet_resunet_baselines.py`、`train_classic_resunet_b0.py` | U-Net、ResUNet 基础对照 |
| SMP 基线 | `train_smp_segmentation.py` | DeepLabV3+、PSPNet 等对照 |
| UNet3+ | `train_unet3p.py`、`train_unet3p_dp2.py` | 全尺度跳跃连接与深监督 |
| ResNet34-UNet3+ | `train_resnet34_unet3p.py` | 主要编码器基线 |
| 边界分支 | `train_resnet34_unet3p_edge*.py` | 边界监督、细化和双解码器实验 |
| CBSA/LASPP 等模块 | `train_resnet34_unet3p_cbsa*.py`、`train_resunet_cbsa*.py` | 注意力、多尺度上下文和跳跃连接消融 |
| BRR 系列 | `train_resnet34_unet_brr_*.py` | boundary/rim 辅助监督与后续改进 |
| 外部基线 | `prepare_nnunet_caries.py`、`eval_medsam_gtbox_caries.py` | nnU-Net、MedSAM 对照 |
| 检测对照 | `prepare_yolo_caries_detection.py` | YOLO 检测格式和召回评估 |

## 2. 数据流

```text
原始图像与标注
      │
      ├─ check_pairs.py
      ├─ make_all_masks.py
      └─ make_splits.py
              │
              ▼
 data/splits/{train,val,test}.txt
              │
              ├─ train_*.py
              │      └─ runs/<run-name>/checkpoints/best.pth
              │
              ├─ eval_per_case_metrics.py
              │      └─ per_case_metrics*.{csv,json}
              │
              └─ analyze_*.py / make_final_visual_cases.py
```

Split 文件每行包含绝对图像路径和绝对 mask 路径，以 Tab 分隔。

## 3. 文件命名规则

### Python

- `train_*.py`：模型定义与训练。
- `eval_*.py`：测试集或逐病例评估。
- `prepare_*.py`：第三方框架数据准备。
- `make_*.py`：mask、split、可视化等产物生成。
- `analyze_*.py`：难度分组、病灶尺寸和论文证据分析。
- `measure_*.py`：参数量、FLOPs 和速度统计。
- `smoke_*.py`：最小前向或依赖检查。

### Shell

- `scripts/jobs/submitjob_*.sh`：常规 Slurm 训练任务。
- `scripts/jobs/submitjob_smoke_*.sh`：短周期冒烟测试。
- `scripts/jobs/submit_*.sh`：评估和分析任务。
- `scripts/tools/make_*_patch.sh`：根据既有脚本构建后续实验版本。
- `scripts/tools/pack_*.sh`：收集代码、指标和实验产物。
- `scripts/setup/*.sh`：依赖安装、模型下载和外部基线准备。

## 4. BRR 消融命名

仓库中的 B1/B2/B3/B4/B6/B7 与 MS 脚本构成一条连续消融线：

- B1：boundary/rim 辅助头。
- B2：RBSG、pre-aux、半径和 warm-up 等变体。
- B3：MBC 变体。
- B4：LSHNW 变体。
- B6：component recall 变体。
- B7：ULRB A/B 变体。
- MS：组合式后续模型。

对应 `scripts/jobs/submitjob_smoke_*.sh` 可先做短周期验证，再运行 200 epoch 正式任务。

## 5. 外部实验

### nnU-Net v2

1. `prepare_nnunet_caries.py` 转换数据。
2. `scripts/jobs/submitjob_nnunet*_plan*.sh` 规划与预处理。
3. `scripts/jobs/submitjob_nnunet*_fold0*.sh` 训练。
4. `eval_nnunet_caries.py` 统一评估。

### MedSAM

1. `scripts/setup/setup_medsam_caries.sh` 准备仓库与权重。
2. `scripts/jobs/submitjob_medsam_gtbox_caries.sh` 使用 GT box 评估。
3. `scripts/tools/pack_medsam_gtbox_result.sh` 打包结果。

### YOLO

1. `prepare_yolo_caries_detection.py` 生成检测数据。
2. `scripts/jobs/submitjob_yolo11n_caries_detect.sh` 训练。
3. `eval_yolo_caries_recall_conf.py` 分析置信度与召回率。

## 6. 迁移到其他机器

迁移时至少检查以下内容：

1. 将 `/share/home/u2515283028/caries_project` 替换为实际项目路径。
2. 更新 Slurm 的分区、CPU、GPU 和日志配置。
3. 创建 `caries-train` 环境并安装匹配 CUDA 的 PyTorch。
4. 重新生成 split，避免其中保留旧机器的绝对路径。
5. 确认 ImageNet 预训练权重可下载，或提前放入缓存。
6. 先运行数据 smoke test 和 1 epoch 任务，再提交长任务。

## 7. 维护建议

- 新实验使用独立的 `--run-name`，不要覆盖已有 `runs/<run-name>`。
- 新增模型时同步提供 smoke job 和逐病例评估配置。
- 记录 Python、PyTorch、CUDA、GPU 与随机种子。
- 保留最终 `config.json`、`history.csv` 和指标 JSON；权重与预测图使用外部存储。
- 后续若要重构目录，先批量更新 Shell 脚本和 Python 模块导入，再移动文件。
