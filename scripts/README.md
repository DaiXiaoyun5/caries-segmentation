# Shell 脚本

Shell 脚本按用途分为三个目录：

- `jobs/`：Slurm 训练、评估、可视化、复杂度统计与冒烟任务。
- `tools/`：数据检查、实验脚本生成、结果收集与打包工具。
- `setup/`：依赖安装、资源下载和外部模型准备。

请从项目根目录运行脚本，以保持脚本中的 `src/`、`runs/` 和其他相对路径可用。

```bash
# 提交训练任务
sbatch scripts/jobs/submitjob_resnet34_unet_auglite_e200_constlr_bs6.sh

# 生成后续实验脚本
bash scripts/tools/make_b2_preaux_patch.sh

# 准备外部模型
bash scripts/setup/setup_medsam_caries.sh
```

脚本仍默认集群项目根目录为 `/share/home/u2515283028/caries_project`。迁移到其他环境时，需要同步修改绝对路径、Conda 环境、Slurm 分区和 GPU 配置。
