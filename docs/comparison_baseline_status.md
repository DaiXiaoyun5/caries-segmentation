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
| U-Mamba-Bot 2D | User-supplied official repository and embedded nnU-Net trainer | Not generated yet | Waiting only for cluster CUDA/PyTorch compatibility information |

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

## Install/check dependencies

Run once after pulling this branch:

```bash
cd /share/home/u2515283028/caries_project
module load anaconda3/4.12.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate caries-train

pip install -r requirements.txt
python -c "import torch, transformers, segmentation_models_pytorch, timm, yacs, einops; print(torch.__version__)"
```

SegFormer may download `nvidia/mit-b2` on its first run. DeepLabV3+ may
download ImageNet ResNet50 weights on its first run. Their caches are kept in
the project `.cache/` directory and are excluded from Git.

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
