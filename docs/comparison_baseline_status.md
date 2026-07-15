# CariXray external baseline reproduction status

This document separates faithful method reproductions from older exploratory
or B30-controlled variants. The main comparison table must use only the rows
marked **paper-ready after rerun**.

## Status

| Method | Architecture and recipe | Job | Status |
|---|---|---|---|
| nnU-Net v2 2D | Official self-configuring 2D pipeline and fixed CariXray split | `scripts/jobs/submitjob_nnunet_caries_2d_fold0_fixed.sh` | Already implemented; do not regenerate |
| Attention U-Net | Native five-level U-Net, official feature scale 4, additive grid gates, first low-level skip ungated, deep supervision; Adam/Dice official CT recipe adapted only from 3D to 2D | `submit/submitjob_attention_unet_official_e1000_bs2.sh` | Paper-ready after rerun |
| SegFormer-B2 | ImageNet MiT-B2 + standard all-MLP decoder; AdamW `6e-5`, head `10x`, weight decay `0.01`, warmup + poly schedule, two-class CE | `submit/submitjob_segformer_b2_official_40k_bs2.sh` | Paper-ready after rerun |
| DeepLabV3+ | ResNet50 ImageNet, output stride 16, ASPP `12/24/36`, standard decoder; SGD, encoder `0.1x` LR, poly schedule, CE | `submit/submitjob_deeplabv3plus_r50_os16_30k_bs4.sh` | Paper-ready after rerun |
| UNet++ | Native `32/64/128/256/512` encoder, nested dense skip pathways, four deep-supervision heads and accurate-mode averaging; Adam `3e-4`, BCE+Dice | `submit/submitjob_unetpp_official_ds_e200_bs4.sh` | Paper-ready after rerun |
| Swin-Unet | User-supplied official Swin-T 224 source/config/checkpoint; official mirrored encoder/decoder initialization, SGD/poly and CE+Dice | `submit/submitjob_swin_unet_official_224_e150_bs6.sh` | Paper-ready after assets are uploaded |
| U-Mamba-Bot 2D | User-supplied official repository, embedded nnU-Net v2.1.1 trainer, official 1000-epoch recipe and fixed CariXray fold 0 | `submit/submitjob_umamba_prepare_caries_2d.sh`, then `submit/submitjob_umamba_bot_official_2d_fold0.sh` | Ready after isolated environment setup and preprocessing |

## Superseded results that must not enter the paper table

- `train_attention_unet_r34_auglite_e260.py` is a project-specific
  ResNet34 U-Net with additive gates. It is not the native Attention U-Net.
- `train_segformer_b2_auglite_e260.py` uses the B30 constant-AdamW/BCE
  protocol instead of SegFormer's official optimizer groups, CE, warmup, and
  polynomial schedule.
- `runs/deeplabv3plus_resnet34_e50_bs6/` is a short ResNet34 exploratory run.
- `runs/unetpp_baseline_v3_bs6/` is a short generic SMP run without the
  original four-head deep-supervision setup.

The historical Python and submit files remain in Git for auditability, but the
two obsolete submit scripts now stop with a clear error to prevent accidental
use.

## Swin-Unet asset placement

`external_assets/` is intentionally ignored by Git and therefore is not created
by `git pull`. Create it once on the cluster:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

mkdir -p /share/home/u2515283028/caries_project/external_assets
```

Do not commit the ZIP or checkpoint. Upload the supplied local files to exactly:

```text
/share/home/u2515283028/caries_project/external_assets/Swin-Unet-main.zip
/share/home/u2515283028/caries_project/external_assets/swin_tiny_patch4_window7_224.pth
```

Verified local SHA256 values:

```text
Swin-Unet-main.zip
0dff0004b71514fb3509b4fabc6b5ffd76d33aee283ba9fca8e0b1ceeb6215dc

swin_tiny_patch4_window7_224.pth
9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3
```

The submit job verifies both hashes, extracts the source to the ignored
`external_models/` directory, and imports the official network directly. It
does not copy or reimplement the Swin-Unet architecture.

## Isolated external-baseline environment

The existing `caries-train` environment has PyTorch `1.10.0+cu111`, while
Transformers 4.50 requires PyTorch 2.0 or newer. Do not upgrade
`caries-train`, because that environment must remain stable for B2/B30. Run
this once to create the separate `caries-baselines` environment:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

bash scripts/setup/setup_official_baselines_env.sh
```

The five external-baseline submit jobs first activate `caries-train` to obey
the project convention, then switch to `caries-baselines`. The B2/B30 jobs are
unchanged and continue to use only `caries-train`.

The setup script downloads `nvidia/mit-b2` and the ImageNet ResNet50 weights on
the login node, stores them under the ignored project `.cache/` directory, and
immediately verifies offline loading. SegFormer and DeepLabV3+ jobs never rely
on compute-node Internet access; each performs an offline cache check before
starting training. This changes neither architecture nor initialization.

If the environment already exists, rerunning the same setup command is safe and
also prepares any missing pretrained assets. A successful setup ends with both
of these lines:

```text
SegFormer MiT-B2 cache: OK
DeepLabV3+ ResNet50 cache: OK
```

## U-Mamba official environment and data

The login-node output `nvidia-smi: command not found` and
`cuda_available=False` is normal because the login node has no GPU. The
important incompatibility is that `caries-train` uses torch 1.10/cu111, while
official U-Mamba requires Python 3.10, torch 2.0.1 and cu118. U-Mamba therefore
uses a second isolated environment named `umamba-caries` and the official
prebuilt CUDA extension wheels; it does not compile with login-node `nvcc`.

Upload the supplied source archive to:

```text
/share/home/u2515283028/caries_project/external_assets/U-Mamba-main.zip
```

Verified SHA256:

```text
7d03c731ddb30061d46f1f40d42885ffce868c8581dffcce38794f20c4f8decc
```

Then install, preprocess, and train in this order:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

bash scripts/setup/setup_umamba_official_env.sh
sbatch submit/submitjob_umamba_prepare_caries_2d.sh
```

Wait until the preparation job finishes successfully, then run:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_umamba_bot_official_2d_fold0.sh
```

The training job performs a real Mamba CUDA forward preflight on its allocated
GPU before starting the official trainer. It reuses the existing raw
`Dataset701_CariXray` through a symlink named `Dataset711_CariXray`, but creates
independent U-Mamba preprocessed data and results so it cannot corrupt the
existing nnU-Net run. Dataset ID 711 is used only to avoid colliding with the
official archive's example `Dataset701_AbdomenCT`; the images and fixed split
are unchanged.

## Diagnosing an early baseline failure

Jobs `94572` and `94573` were submitted before PR #21 was pulled. Their logs
prove two independent startup failures:

- SegFormer `94572` used `caries-train` and stopped because `transformers` was
  not installed.
- DeepLabV3+ `94573` used `caries-train` and stopped when its compute node could
  not download `resnet50-19c8e357.pth`.

Small structured records are retained in
`diagnostics/official_baselines/*_failure.json`. The setup and submit scripts
now address both failures with an isolated environment and login-node weight
prefetching.

The latest Git commit contains only `config.json` for Attention U-Net and
UNet++. No `*.out`, `*.err`, or `*.log` file was committed because logs and
all of `runs/` are intentionally ignored. A config without `history.csv`
narrows the failure to first-batch loading or the initial forward check, but it
does not reveal whether the cause was a bad sample, CUDA incompatibility, or
out-of-memory.

After this update, every official Python runner writes `run_status.json`,
`failure_report.json`, and `failure_traceback.txt` beside `config.json` when it
fails. It also writes a small, non-ignored
`diagnostics/official_baselines/*_failure.json`, so future tracebacks can be
uploaded with a normal Git commit. The Slurm scripts merge stderr into stdout.
Inspect the existing cluster logs with:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

ls -lt runs/logs | head -20
tail -n 160 runs/logs/attention_unet_official_*.out
tail -n 160 runs/logs/unetpp_official_*.out
```

If a rerun fails, inspect the preserved reports directly:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

cat runs/attention_unet_official_2d_e1000_bs2/failure_traceback.txt
cat runs/unetpp_official_ds_e200_bs4/failure_traceback.txt
```

## Submit commands

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

sbatch submit/submitjob_attention_unet_official_e1000_bs2.sh
sbatch submit/submitjob_segformer_b2_official_40k_bs2.sh
sbatch submit/submitjob_deeplabv3plus_r50_os16_30k_bs4.sh
sbatch submit/submitjob_unetpp_official_ds_e200_bs4.sh
sbatch submit/submitjob_swin_unet_official_224_e150_bs6.sh
```

## Output contract

Every new runner saves:

```text
runs/{run_name}/config.json
runs/{run_name}/history.csv
runs/{run_name}/best_val_metrics.json
runs/{run_name}/best_train_metrics.json
runs/{run_name}/test_metrics.json
runs/{run_name}/summary_metrics.json
runs/{run_name}/per_case_metrics.csv
runs/{run_name}/per_case_metrics_summary.json
runs/{run_name}/checkpoints/{best,last}.pth
runs/{run_name}/preds/test_preview.png
```

U-Mamba retains its official nnU-Net checkpoints in the isolated official
results directory. `src/export_umamba_official_results.py` exports its
history, configuration, fixed-split validation information, global-pixel test
metrics, per-case metrics and summary files into
`runs/umamba_bot_official_2d_fold0/` without copying the large checkpoint.

The architecture, optimizer, loss, scheduler, budget, input size, and
checkpoint rule are recorded in each `config.json`. Only the CariXray split and
final metric implementation are shared across methods. Swin-Unet still trains
and selects its checkpoint at the official 224 resolution, but its final
prediction is bilinearly restored to 512 and compared with the source GT mask
resized directly to 512. This avoids scoring a 224-round-tripped mask against
the 512-resolution results of the other methods.

## Primary references

- Attention U-Net: <https://arxiv.org/abs/1804.03999> and
  <https://github.com/ozan-oktay/Attention-Gated-Networks>
- SegFormer: <https://arxiv.org/abs/2105.15203> and
  <https://github.com/NVlabs/SegFormer>
- DeepLabV3+: <https://arxiv.org/abs/1802.02611> and
  <https://github.com/VainF/DeepLabV3Plus-Pytorch>
- UNet++: <https://arxiv.org/abs/1807.10165> and
  <https://github.com/MrGiovanni/UNetPlusPlus>
- Swin-Unet: <https://arxiv.org/abs/2105.05537> and
  <https://github.com/HuCaoFighting/Swin-Unet>
- U-Mamba: <https://arxiv.org/abs/2401.04722> and
  <https://github.com/bowang-lab/U-Mamba>
