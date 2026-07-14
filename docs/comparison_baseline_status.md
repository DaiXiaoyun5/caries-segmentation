# CariXray external comparison baseline status

This inventory was checked against the repository after pulling `origin/main`.
It prevents duplicate experiments and separates exploratory runs from final,
controlled comparisons.

## Existing experiments (not regenerated)

| Model | Repository evidence | Current protocol/result | Paper-table status |
|---|---|---|---|
| nnU-Net v2 2D | `src/prepare_nnunet_caries.py`, `src/eval_nnunet_caries.py`, and `scripts/jobs/submitjob_nnunet_caries_2d_fold0_fixed.sh` | Official self-configuring pipeline and fixed project split are already implemented. Prediction metrics live under the ignored `nnunet_caries/` directory, so the numeric result is not present in Git. | Keep the official recipe. Copy its final metric JSON into the paper evidence package before writing the table. |
| DeepLabV3+ (ResNet34) | `src/train_smp_segmentation.py` and `runs/deeplabv3plus_resnet34_e50_bs6/` | 512, batch 6, 50 epochs, lr 3e-4; test Dice 0.8335. No AugLite and no AdamW weight decay in this old run. | Existing exploratory result only; it is not an E260-controlled result. |
| U-Net++ (ResNet34) | `src/train_unetpp.py` and `runs/unetpp_baseline_v3_bs6/` | 512, batch 6, 30 epochs, lr 3e-4; test Dice 0.8367. No AugLite and no E260 training. | Existing exploratory result only; it is not an E260-controlled result. |

The two short-run CNN results were not regenerated because the user explicitly
requested that already completed models should only be reported. If they enter
the final main table, their different protocol must be disclosed or they should
later be rerun under the controlled E260 protocol.

## New controlled experiments

Both new experiments use the same split and controlled protocol as B30 E260:
512 x 512, AugLite, batch size 6, 260 epochs, constant AdamW lr 3e-4,
weight decay 1e-4, BCE + Dice, and seed 42.

| Model | Training script | Submit script |
|---|---|---|
| Attention U-Net (ResNet34) | `src/train_attention_unet_r34_auglite_e260.py` | `submit/submitjob_attention_unet_r34_auglite_e260_constlr_bs6.sh` |
| SegFormer-B2 | `src/train_segformer_b2_auglite_e260.py` | `submit/submitjob_segformer_b2_auglite_e260_constlr_bs6.sh` |

Attention U-Net uses ordinary additive attention gates on four skip features.
It has no boundary/rim supervision and no RBSG, so it directly tests whether
B2's task-prior mechanism is more useful than generic attention filtering.

SegFormer-B2 uses the official NVIDIA `nvidia/mit-b2` ImageNet checkpoint and
the standard SegFormer all-MLP decoder. It tests whether generic hierarchical
Transformer context is sufficient for the weak-boundary, small-lesion task.

## Deferred official implementations (no substitute model generated)

### Swin-Unet

Required assets and decisions:

1. Official source: <https://github.com/HUCAOFIGHTING/SWIN-UNET>
2. Official Swin-T pretrained checkpoint linked from that repository:
   <https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY>
3. Confirm the fair input policy. The official configuration is 224 x 224 with
   window size 7, whereas this project uses 512 x 512. A 512 adaptation changes
   the fixed-resolution/window configuration and pretrained decoder loading.

Please provide the official repository ZIP and pretrained checkpoint if the
supercomputer cannot access GitHub/Google Drive. They should remain outside Git
because checkpoints and third-party repositories are ignored.

### U-Mamba-Bot 2D

Required assets and environment:

1. Official source: <https://github.com/bowang-lab/U-Mamba>
2. A separate official-compatible environment (the repository specifies Ubuntu
   20.04, CUDA 11.8, Python 3.10, PyTorch 2.0.1, `causal-conv1d`, and
   `mamba-ssm`).
3. Install `U-Mamba/umamba` in editable mode and point its nnU-Net paths to the
   existing Dataset701 conversion.

The official 2D command uses `nnUNetTrainerUMambaBot`. Reimplementing only a
home-made bottleneck block inside the current SMP U-Net would not be a valid
U-Mamba comparison, so no such substitute was generated.
