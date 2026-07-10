#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
对指定 run_name 的 best.pth 做逐图测试集评估，生成：
runs/{run_name}/per_case_metrics.csv
runs/{run_name}/per_case_metrics_summary.json
runs/{run_name}/preds_per_case/*.png  可选

支持常见模型：
1. train_unet3p.py                       UNet3Plus
2. train_resnet34_unet3p.py              ResNet34UNet3Plus
3. train_resnet34_unet3p_edge.py         ResNet34UNet3PlusEdge
4. train_resnet34_unet3p_edge_simam.py   ResNet34UNet3PlusEdgeSimAM
5. train_resnet34_unet3p_edge_ra.py      ResNet34UNet3PlusEdgeRA
6. train_resnet34_unet3p_edge_simam_ra.py
7. train_resnet34_unet3p_edge_simam_scse_hd1.py

如果某个模型类名不一致，运行时报错后再按具体文件补适配。
"""

import os
import sys
import csv
import json
import inspect
import argparse
import importlib
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class TestMaskDataset(Dataset):
    def __init__(self, split_file, target_size=(512, 512)):
        self.split_file = Path(split_file)
        self.target_size = target_size
        self.items = []

        with open(self.split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, mask_path = line.split("\t")
                self.items.append((Path(img_path), Path(mask_path)))

        if len(self.items) == 0:
            raise RuntimeError(f"No samples found in {self.split_file}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = (np.array(mask).astype(np.float32) > 0).astype(np.float32)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()

        return image_tensor, mask_tensor, str(img_path), str(mask_path)


def infer_model_info(run_name: str):
    """
    根据 run_name 粗略推断训练脚本和模型类。
    如果你的 run_name 比较特殊，可以用命令行显式传：
    --module-name xxx --class-name xxx
    """
    rn = run_name.lower()

    # 注意顺序：更具体的放前面
    if "simam_scse_hd1" in rn:
        return "train_resnet34_unet3p_edge_simam_scse_hd1", "ResNet34UNet3PlusEdgeSimAMSCSEHD1"
    if "simam_ra" in rn:
        return "train_resnet34_unet3p_edge_simam_ra", "ResNet34UNet3PlusEdgeSimAMRA"
    if "edge_simam" in rn:
        return "train_resnet34_unet3p_edge_simam", "ResNet34UNet3PlusEdgeSimAM"
    if "edge_ra" in rn:
        return "train_resnet34_unet3p_edge_ra", "ResNet34UNet3PlusEdgeRA"
    if "edge_ag_scse" in rn:
        return "train_resnet34_unet3p_edge_ag_scse", "ResNet34UNet3PlusEdgeAGSCSE"
    if "edge_ag" in rn:
        return "train_resnet34_unet3p_edge_ag_scse", "ResNet34UNet3PlusEdgeAGSCSE"
    if "edge_scse" in rn:
        return "train_resnet34_unet3p_edge_ag_scse", "ResNet34UNet3PlusEdgeAGSCSE"
    if "edge_wavelet" in rn:
        return "train_resnet34_unet3p_edge_wavelet", "ResNet34UNet3PlusEdgeWavelet"
    if "edge_dcn" in rn:
        return "train_resnet34_unet3p_edge_dcn", "ResNet34UNet3PlusEdgeDCN"
    if "edge" in rn:
        return "train_resnet34_unet3p_edge", "ResNet34UNet3PlusEdge"

    if "b34_dmrc" in rn:
        return "train_resnet34_unet_brr_b34_dmrc_e260", "ResNet34UNetBRRB34DMRC"
    if "b33_lglb" in rn:
        return "train_resnet34_unet_brr_b33_lglb_e200", "ResNet34UNetBRRB33LGLB"
    if "b32_cglb" in rn:
        return "train_resnet34_unet_brr_b32_cglb_e200", "ResNet34UNetBRRB32CGLB"
    if "b30_ssgcb_only" in rn or "b30_ssgcbonly" in rn:
        return "train_resnet34_unet_brr_b30_ssgcb_only_e200", "ResNet34UNetB30SSGCBOnly"
    if "b31_ssgcb" in rn:
        return "train_resnet34_unet_brr_b31_ssgcb_e200", "ResNet34UNetBRRB31SSGCB"
    if "b30_ssgcb" in rn:
        return "train_resnet34_unet_brr_b30_ssgcb_light_e200", "ResNet34UNetBRRB30SSGCBLight"
    if "b29_context" in rn or "b29_ctx" in rn:
        return "train_resnet34_unet_brr_b29_context_only_e200", "ResNet34UNetBRRB29ContextOnly"
    if "b28_rbsg" in rn or "b28_rbsgo" in rn:
        return "train_resnet34_unet_brr_b28_rbsg_only_e200", "ResNet34UNetBRRB28RBSGOnly"
    if "b27_comp" in rn or "b27_topk" in rn:
        return "train_resnet34_unet_brr_b27_comp_topk_e200", "ResNet34UNetBRRB27CompTopK"
    if "b26_cemia" in rn:
        return "train_resnet34_unet_brr_b26_cemia_lite_e200", "ResNet34UNetBRRB26CEMIALite"
    if "b25_lkdr" in rn:
        return "train_resnet34_unet_brr_b25_lkdr_e200", "ResNet34UNetBRRB25LKDR"
    if "b24_wtconv" in rn or "b24_wt" in rn:
        return "train_resnet34_unet_brr_b24_wtconv_context_e200", "ResNet34UNetBRRB24WTConvContext"
    if "b23_ram" in rn:
        return "train_resnet34_unet_brr_b23_ram_lite_e200", "ResNet34UNetBRRB23RAMLite"
    if "b22_recall" in rn or "b22_tversky" in rn:
        return "train_resnet34_unet_brr_b22_recall_tversky_e200", "ResNet34UNetBRRB22RecallTversky"
    if "b21_ife" in rn or "b21_ifelite" in rn:
        return "train_resnet34_unet_brr_b21_ife_lite_e200", "ResNet34UNetBRRB21IFELite"
    if "b20_dmr" in rn:
        return "train_resnet34_unet_brr_b20_dmr_e200", "ResNet34UNetBRRB20DMR"
    if "b19_flip" in rn or "b19_consistency" in rn:
        return "train_resnet34_unet_brr_b19_flip_consistency_e200", "ResNet34UNetBRRB19FlipConsistency"
    if "b18_ema" in rn:
        return "train_resnet34_unet_brr_b18_ema_e200", "ResNet34UNetBRRB18EMA"
    if "b17_wavelet" in rn or "b17_wfpr" in rn:
        return "train_resnet34_unet_brr_b17_wavelet_rescue_e200", "ResNet34UNetBRRB17WaveletRescue"
    if "b16_sgdf" in rn:
        return "train_resnet34_unet_brr_b16_sgdf_e200", "ResNet34UNetBRRB16SGDF"
    if "b15_edge_simam" in rn or "b15_esdr" in rn:
        return "train_resnet34_unet_brr_b15_edge_simam_rescue_e200", "ResNet34UNetBRRB15EdgeSimAMRescue"
    if "b14_ppm" in rn:
        return "train_resnet34_unet_brr_b14_ppm_e200", "ResNet34UNetBRRB14PPM"
    if "b13_cer" in rn:
        return "train_resnet34_unet_brr_b13_cer_e200", "ResNet34UNetBRRB13CER"
    if "b12_tdr" in rn:
        return "train_resnet34_unet_brr_b12_tdr_e200", "ResNet34UNetBRRB12TDR"
    if "b11_ema" in rn:
        return "train_resnet34_unet_brr_b11_ema_e200", "ResNet34UNetBRRB11EMA"
    if "b10_dmsk" in rn:
        return "train_resnet34_unet_brr_b10_dmsk_skip_e200", "ResNet34UNetBRRB10DMSKSkip"

    if "resnet34_unet3p" in rn:
        return "train_resnet34_unet3p", "ResNet34UNet3Plus"

    if "unet3p" in rn:
        return "train_unet3p", "UNet3Plus"

    raise RuntimeError(
        f"Cannot infer model class from run_name={run_name}. "
        f"Please pass --module-name and --class-name manually."
    )


def build_model(model_cls, cfg):
    """
    尽量根据模型 __init__ 参数名自动传参。
    """
    sig = inspect.signature(model_cls.__init__)
    params = set(sig.parameters.keys())
    params.discard("self")

    deep_supervision = cfg.get("deep_supervision", 1)
    if isinstance(deep_supervision, str):
        deep_supervision = int(deep_supervision)
    deep_supervision = bool(deep_supervision)

    cat_channels = cfg.get("cat_channels", cfg.get("cat_ch", 64))
    base_channels = cfg.get("base_channels", cfg.get("base_ch", 64))

    candidates = {
        "encoder_name": cfg.get("encoder_name", "resnet34"),
        "encoder_weights": cfg.get("encoder_weights", "imagenet"),
        "in_channels": 3,
        "num_classes": 1,
        "classes": 1,
        "cat_channels": cat_channels,
        "cat_ch": cat_channels,
        "base_channels": base_channels,
        "base_ch": base_channels,
        "deep_supervision": deep_supervision,

        # SimAM / RA / SCSE / AG / Wavelet 等可选参数
        "simam_lambda": cfg.get("simam_lambda", 1e-4),
        "simam_position": cfg.get("simam_position", "all"),
        "ra_scale": cfg.get("ra_scale", 1.0),
        "scse_reduction": cfg.get("scse_reduction", 16),
        "use_ag": cfg.get("use_ag", 0),
        "use_scse": cfg.get("use_scse", 0),
        "wavelet_position": cfg.get("wavelet_position", "hd1"),
        "wavelet_scale": cfg.get("wavelet_scale", 0.1),

        # B10 DMSK-Skip-lite optional parameters. Signature filtering below
        # keeps these isolated from all existing model classes.
        "dmsk_hidden_channels": cfg.get("dmsk_hidden_channels", 32),
        "dmsk_context_dilation": cfg.get("dmsk_context_dilation", 3),
        "dmsk_init_scale": cfg.get("dmsk_init_scale", 0.10),
        "dmsk_hard_selection": bool(cfg.get("dmsk_hard_selection", 1)),

        # B11 EMA optional parameter.
        "ema_reduction": cfg.get("ema_reduction", 4),

        # B12 Texture Detail Rescue optional parameters.
        "tdr_hidden_channels": cfg.get("tdr_hidden_channels", 16),
        "tdr_max_delta": cfg.get("tdr_max_delta", 0.75),
        "tdr_init_scale": cfg.get("tdr_init_scale", 0.15),
        "tdr_bias_init": cfg.get("tdr_bias_init", -3.0),
        "tdr_highpass_kernel": cfg.get("tdr_highpass_kernel", 7),

        # B13 Candidate Expansion Refiner optional parameters.
        "cer_hidden_channels": cfg.get("cer_hidden_channels", 24),
        "cer_max_delta": cfg.get("cer_max_delta", 0.80),
        "cer_init_scale": cfg.get("cer_init_scale", 0.12),
        "cer_bias_init": cfg.get("cer_bias_init", -4.0),
        "cer_highpass_kernel": cfg.get("cer_highpass_kernel", 7),
        "cer_candidate_tau": cfg.get("cer_candidate_tau", 0.15),
        "cer_candidate_sharpness": cfg.get("cer_candidate_sharpness", 12.0),

        # B14 PPM-lite bottleneck optional parameters.
        "ppm_hidden_channels": cfg.get("ppm_hidden_channels", 64),
        "ppm_pool_sizes": cfg.get("ppm_pool_sizes", [1, 2, 3, 6]),
        "ppm_dropout": cfg.get("ppm_dropout", 0.0),
        "ppm_init_scale": cfg.get("ppm_init_scale", 0.08),

        # B15 Edge-SimAM Detail Rescue optional parameters.
        "esdr_hidden_channels": cfg.get("esdr_hidden_channels", 16),
        "esdr_max_delta": cfg.get("esdr_max_delta", 0.60),
        "esdr_init_scale": cfg.get("esdr_init_scale", 0.10),
        "esdr_bias_init": cfg.get("esdr_bias_init", -4.0),
        "esdr_contrast_kernel": cfg.get("esdr_contrast_kernel", 7),
        "esdr_simam_lambda": cfg.get("esdr_simam_lambda", 1e-4),

        # B16 Semantic-Guided Detail Fusion optional parameters.
        "sgdf_hidden_channels": cfg.get("sgdf_hidden_channels", 32),
        "sgdf_init_scale": cfg.get("sgdf_init_scale", 0.05),
        "sgdf_use_skip1": bool(cfg.get("sgdf_use_skip1", 1)),
        "sgdf_use_skip2": bool(cfg.get("sgdf_use_skip2", 1)),

        # B17 Wavelet Frequency Prompt Rescue optional parameters.
        "wfpr_hidden_channels": cfg.get("wfpr_hidden_channels", 16),
        "wfpr_max_delta": cfg.get("wfpr_max_delta", 0.55),
        "wfpr_init_scale": cfg.get("wfpr_init_scale", 0.08),
        "wfpr_bias_init": cfg.get("wfpr_bias_init", -4.0),
        "wfpr_candidate_tau": cfg.get("wfpr_candidate_tau", 0.12),
        "wfpr_candidate_sharpness": cfg.get("wfpr_candidate_sharpness", 12.0),

        # B21 IFE-lite optional parameters.
        "ife_init_scale": cfg.get("ife_init_scale", 0.0),
        "ife_max_scale": cfg.get("ife_max_scale", 0.30),
        "ife_curvature_weight": cfg.get("ife_curvature_weight", 0.60),

        # B24 WTConv context optional parameters.
        "wt_kernel_size": cfg.get("wt_kernel_size", 5),
        "wt_levels": cfg.get("wt_levels", 2),
        "wt_init_scale": cfg.get("wt_init_scale", 0.05),
        "wt_wavelet_init_scale": cfg.get("wt_wavelet_init_scale", 0.10),

        # B25 Large-Kernel Decoder Refiner optional parameters.
        "lkdr_hidden_channels": cfg.get("lkdr_hidden_channels", 48),
        "lkdr_expansion": cfg.get("lkdr_expansion", 3.0),
        "lkdr_local_kernel": cfg.get("lkdr_local_kernel", 5),
        "lkdr_context_kernel": cfg.get("lkdr_context_kernel", 7),
        "lkdr_context_dilation": cfg.get("lkdr_context_dilation", 2),
        "lkdr_init_scale": cfg.get("lkdr_init_scale", 0.05),
        "lkdr_max_scale": cfg.get("lkdr_max_scale", 0.50),

        # B26 CEMIA-lite optional parameters.
        "cemia_alpha": cfg.get("cemia_alpha", 0.10),
        "cemia_beta": cfg.get("cemia_beta", 0.10),

        # B29 / B8B context bottleneck optional parameters.
        "context_branch_channels": cfg.get("context_branch_channels", 128),
        "context_dropout": cfg.get("context_dropout", 0.0),
        "context_init_scale": cfg.get("context_init_scale", 0.10),

        # B30/B31 SSGCB optional parameters.
        "ssgcb_branch_channels": cfg.get("ssgcb_branch_channels", 96),
        "ssgcb_dilations": cfg.get("ssgcb_dilations", [1, 3, 5]),
        "ssgcb_dropout": cfg.get("ssgcb_dropout", 0.0),
        "ssgcb_scale_init": cfg.get("ssgcb_scale_init", 0.05),
        "ssgcb_scale_max": cfg.get("ssgcb_scale_max", 0.30),

        # B32 CGLB optional parameters.
        "cglb_hidden_channels": cfg.get("cglb_hidden_channels", 128),
        "cglb_kernel_size": cfg.get("cglb_kernel_size", 7),
        "cglb_expansion": cfg.get("cglb_expansion", 2.0),
        "cglb_dropout": cfg.get("cglb_dropout", 0.0),
        "cglb_scale_init": cfg.get("cglb_scale_init", 0.05),
        "cglb_scale_max": cfg.get("cglb_scale_max", 0.30),

        # B33 LGLB optional parameters.
        "lglb_hidden_channels": cfg.get("lglb_hidden_channels", 128),
        "lglb_local_kernel": cfg.get("lglb_local_kernel", 5),
        "lglb_global_kernel": cfg.get("lglb_global_kernel", 7),
        "lglb_global_dilation": cfg.get("lglb_global_dilation", 3),
        "lglb_dropout": cfg.get("lglb_dropout", 0.0),
        "lglb_scale_init": cfg.get("lglb_scale_init", 0.05),
        "lglb_scale_max": cfg.get("lglb_scale_max", 0.30),

        # B34 DMRC-lite optional parameters.
        "dmrc_mid_channels": cfg.get("dmrc_mid_channels", 96),
        "dmrc_dropout": cfg.get("dmrc_dropout", 0.0),
        "dmrc_scale_init": cfg.get("dmrc_scale_init", 0.05),
        "dmrc_scale_max": cfg.get("dmrc_scale_max", 0.30),
    }

    kwargs = {k: v for k, v in candidates.items() if k in params}

    print("=" * 80)
    print("Instantiate model:")
    print("class:", model_cls)
    print("kwargs:", kwargs)
    print("=" * 80)

    return model_cls(**kwargs)


def load_checkpoint_state(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    # 兼容 DataParallel 的 module. 前缀
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        new_state[k] = v

    return new_state


def extract_main_logits(outputs):
    """
    兼容：
    plain: d1 或 [d1,d2,d3,d4,d5]
    edge: ([d1,d2,d3,d4,d5], edge_logits)
    dual: ([d1,...], edge_logits, center_logits)
    bgfe: ([d1,...], [b1,b2,b3,b4])
    """
    if isinstance(outputs, tuple):
        seg_outputs = outputs[0]
    else:
        seg_outputs = outputs

    if isinstance(seg_outputs, (list, tuple)):
        return seg_outputs[0]
    return seg_outputs


def compute_per_case_metrics(logits, masks, img_paths, mask_paths, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    B = preds.shape[0]
    rows = []

    for i in range(B):
        pred = preds[i, 0].detach().cpu().numpy().astype(np.uint8)
        prob = probs[i, 0].detach().cpu().numpy().astype(np.float32)
        gt = masks[i, 0].detach().cpu().numpy().astype(np.uint8)

        tp = int(np.logical_and(pred == 1, gt == 1).sum())
        fp = int(np.logical_and(pred == 1, gt == 0).sum())
        fn = int(np.logical_and(pred == 0, gt == 1).sum())
        tn = int(np.logical_and(pred == 0, gt == 0).sum())

        eps = 1e-7
        dice = float((2 * tp + eps) / (2 * tp + fp + fn + eps))
        iou = float((tp + eps) / (tp + fp + fn + eps))
        precision = float((tp + eps) / (tp + fp + eps))
        recall = float((tp + eps) / (tp + fn + eps))

        gt_area = int(gt.sum())
        pred_area = int(pred.sum())
        h, w = gt.shape
        gt_area_ratio = float(gt_area / (h * w))
        pred_area_ratio = float(pred_area / (h * w))

        img_path = Path(img_paths[i])
        mask_path = Path(mask_paths[i])
        case_id = img_path.stem

        rows.append({
            "case_id": case_id,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "gt_area": gt_area,
            "gt_area_ratio": gt_area_ratio,
            "pred_area": pred_area,
            "pred_area_ratio": pred_area_ratio,
            "dice": dice,
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "_pred": pred,
            "_prob": prob,
        })

    return rows


def save_mask_png(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((arr * 255).astype(np.uint8)).save(path)


def main(args):
    run_dir = PROJECT_ROOT / "runs" / args.run_name
    cfg_path = run_dir / "config.json"
    ckpt_path = run_dir / "checkpoints" / "best.pth"

    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found: {cfg_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"best.pth not found: {ckpt_path}")

    cfg = load_json(cfg_path)

    image_size = int(cfg.get("image_size", args.image_size))
    target_size = (image_size, image_size)

    module_name = args.module_name
    class_name = args.class_name

    if module_name is None or class_name is None:
        module_name, class_name = infer_model_info(args.run_name)

    print("=" * 80)
    print("Run name:", args.run_name)
    print("Run dir :", run_dir)
    print("Module  :", module_name)
    print("Class   :", class_name)
    print("Ckpt    :", ckpt_path)
    print("Image size:", image_size)
    print("=" * 80)

    module = importlib.import_module(module_name)

    if not hasattr(module, class_name):
        available = [x for x in dir(module) if "UNet" in x or "ResNet" in x]
        raise AttributeError(
            f"{module_name} has no class {class_name}. "
            f"Available model-like names: {available}"
        )

    model_cls = getattr(module, class_name)
    model = build_model(model_cls, cfg)

    state = load_checkpoint_state(ckpt_path)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("=" * 80)
    print("Checkpoint loaded.")
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))
    if len(missing) > 0:
        print("First missing keys:", missing[:10])
    if len(unexpected) > 0:
        print("First unexpected keys:", unexpected[:10])
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    test_split = PROJECT_ROOT / "data/splits/test.txt"
    ds = TestMaskDataset(test_split, target_size=target_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    pred_dir = run_dir / "preds_per_case"
    if args.save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            images, masks, img_paths, mask_paths = batch
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            outputs = model(images)
            logits = extract_main_logits(outputs)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=True,
                )

            rows = compute_per_case_metrics(
                logits,
                masks,
                img_paths,
                mask_paths,
                threshold=args.threshold,
            )

            for row in rows:
                pred = row.pop("_pred")
                prob = row.pop("_prob")

                if args.save_preds:
                    case_id = row["case_id"]
                    save_mask_png(pred, pred_dir / f"{case_id}_pred.png")
                    # 概率图可选保存，文件会比较多；默认不保存
                    if args.save_prob:
                        prob_u8 = np.clip(prob * 255.0, 0, 255).astype(np.uint8)
                        Image.fromarray(prob_u8).save(pred_dir / f"{case_id}_prob.png")

                all_rows.append(row)

            if (batch_idx + 1) % 10 == 0:
                print(f"Evaluated batches: {batch_idx + 1}/{len(loader)}")

    out_csv = run_dir / "per_case_metrics.csv"
    fieldnames = [
        "case_id",
        "image_path",
        "mask_path",
        "gt_area",
        "gt_area_ratio",
        "pred_area",
        "pred_area_ratio",
        "dice",
        "iou",
        "precision",
        "recall",
        "tp",
        "fp",
        "fn",
        "tn",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    summary = {
        "run_name": args.run_name,
        "num_cases": len(all_rows),
        "threshold": args.threshold,
        "mean_dice": float(np.mean([r["dice"] for r in all_rows])),
        "mean_iou": float(np.mean([r["iou"] for r in all_rows])),
        "mean_precision": float(np.mean([r["precision"] for r in all_rows])),
        "mean_recall": float(np.mean([r["recall"] for r in all_rows])),
        "median_dice": float(np.median([r["dice"] for r in all_rows])),
        "median_iou": float(np.median([r["iou"] for r in all_rows])),
        "median_precision": float(np.median([r["precision"] for r in all_rows])),
        "median_recall": float(np.median([r["recall"] for r in all_rows])),
        "module_name": module_name,
        "class_name": class_name,
        "checkpoint": str(ckpt_path),
        "per_case_csv": str(out_csv),
    }

    out_json = run_dir / "per_case_metrics_summary.json"
    save_json(summary, out_json)

    print("=" * 80)
    print("Per-case evaluation finished.")
    print("Saved CSV :", out_csv)
    print("Saved JSON:", out_json)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--module-name", type=str, default=None)
    parser.add_argument("--class-name", type=str, default=None)

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--save-preds", type=int, default=1)
    parser.add_argument("--save-prob", type=int, default=0)

    args = parser.parse_args()
    main(args)
