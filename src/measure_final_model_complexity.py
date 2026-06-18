#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
最终论文模型复杂度统计：Params、GFLOPs（可选 thop）、推理时间、FPS。

输出目录：
runs/model_complexity_final/

输出：
- final_model_complexity.csv
- final_model_complexity.json

说明：
1. BoundaryLoss / CSFTL 只改变训练损失，不改变网络结构，所以复杂度与 CBSA+Lite-ASPP 相同。
2. 默认 encoder_weights=None，避免在超算上联网下载 ImageNet 权重；复杂度与是否加载预训练权重无关。
3. 若环境没有 thop，则 GFLOPs 留空，但 Params / 推理速度仍会输出。

用法：
cd /share/home/u2515283028/caries_project
python src/measure_final_model_complexity.py --image-size 512 --warmup 10 --repeat 50
"""

import os
import sys
import csv
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
SRC_DIR = PROJECT_ROOT / "src"
OUT_DIR = PROJECT_ROOT / "runs/model_complexity_final"
sys.path.insert(0, str(SRC_DIR))


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


class MainOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out


def profile_gflops(model, image_size, device):
    try:
        from thop import profile
    except Exception as e:
        return "", f"thop unavailable: {repr(e)}"

    model = model.to(device).eval()
    wrapper = MainOutputWrapper(model).to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        flops, _ = profile(wrapper, inputs=(dummy,), verbose=False)
    return float(flops / 1e9), ""


def measure_speed(model, image_size, device, warmup=10, repeat=50):
    model = model.to(device).eval()
    wrapper = MainOutputWrapper(model).to(device).eval()
    dummy = torch.randn(1, 3, image_size, image_size, device=device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = wrapper(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(repeat):
            _ = wrapper(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()

    mean_ms = (t1 - t0) * 1000.0 / repeat
    fps = 1000.0 / mean_ms if mean_ms > 0 else 0.0
    return float(mean_ms), float(fps)


def get_metric_from_run(run_name):
    out = {}
    tm = load_json(PROJECT_ROOT / "runs" / run_name / "test_metrics.json") or {}
    ps = load_json(PROJECT_ROOT / "runs" / run_name / "per_case_metrics_summary.json") or {}
    for m in ["dice", "iou", "precision", "recall"]:
        out[f"test_{m}"] = tm.get(m, "")
        out[f"per_case_mean_{m}"] = ps.get(f"mean_{m}", "")
    return out


def build_unet3p():
    from train_unet3p import UNet3Plus
    return UNet3Plus(in_channels=3, num_classes=1, base_ch=64, deep_supervision=True)


def build_resnet34_unet3p():
    from train_resnet34_unet3p import ResNet34UNet3Plus
    return ResNet34UNet3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
    )


def build_cbsa():
    from train_resnet34_unet3p_cbsa import ResNet34UNet3PlusCBSA
    return ResNet34UNet3PlusCBSA(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
        cbsa_position="all",
        cbsa_lambda=1e-4,
        cbsa_alpha=0.2,
        cbsa_beta=0.1,
    )


def build_cbsa_laspp():
    from train_resnet34_unet3p_cbsa_laspp import ResNet34UNet3PlusCBSALiteASPP
    return ResNet34UNet3PlusCBSALiteASPP(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
        cbsa_position="all",
        cbsa_lambda=1e-4,
        cbsa_alpha=0.2,
        cbsa_beta=0.1,
        aspp_hidden=128,
        aspp_dilations=(1, 3, 5),
        aspp_dropout=0.0,
        aspp_res_scale=0.1,
    )


def build_cbsa_ppm():
    from train_resnet34_unet3p_cbsa_ppm import ResNet34UNet3PlusCBSAPPMLite
    return ResNet34UNet3PlusCBSAPPMLite(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
        cbsa_position="all",
        cbsa_lambda=1e-4,
        cbsa_alpha=0.2,
        cbsa_beta=0.1,
    )


def build_cbsa_ocr():
    from train_resnet34_unet3p_cbsa_ocr_lite import ResNet34UNet3PlusCBSAOCRLite
    return ResNet34UNet3PlusCBSAOCRLite(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
        cbsa_position="all",
        cbsa_lambda=1e-4,
        cbsa_alpha=0.2,
        cbsa_beta=0.1,
    )


def build_edge_simam():
    from train_resnet34_unet3p_edge_simam import ResNet34UNet3PlusEdgeSimAM
    return ResNet34UNet3PlusEdgeSimAM(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
        simam_lambda=1e-4,
        simam_position="all",
    )


def resolve_run(candidates):
    for rn in candidates:
        if (PROJECT_ROOT / "runs" / rn).exists():
            return rn
    return candidates[0]


MODEL_SPECS = [
    {
        "model_name": "UNet3+",
        "run_candidates": ["unet3p_baseline_e50_bs6_dp2_fair", "unet3p_baseline_e50_bs6_fair", "unet3p_baseline_e50", "unet3p_e50_bs6", "unet3p"],
        "builder": build_unet3p,
        "arch_note": "basic baseline",
    },
    {
        "model_name": "ResNet34-UNet3+",
        "run_candidates": ["resnet34_unet3p_baseline_e50"],
        "builder": build_resnet34_unet3p,
        "arch_note": "strong baseline",
    },
    {
        "model_name": "ResNet34-UNet3+ + CBSA",
        "run_candidates": ["resnet34_unet3p_cbsa_e50_bs6"],
        "builder": build_cbsa,
        "arch_note": "CBSA module",
    },
    {
        "model_name": "ResNet34-UNet3+ + CBSA + Lite-ASPP",
        "run_candidates": ["resnet34_unet3p_cbsa_laspp_e50_bs6"],
        "builder": build_cbsa_laspp,
        "arch_note": "main model",
    },
    {
        "model_name": "CBSA + Lite-ASPP + BoundaryLoss",
        "run_candidates": ["resnet34_unet3p_cbsa_laspp_boundaryloss_e50_bs6"],
        "builder": build_cbsa_laspp,
        "arch_note": "same architecture as CBSA+Lite-ASPP; loss-only change",
    },
    {
        "model_name": "CBSA + Lite-ASPP + CSFTL",
        "run_candidates": ["resnet34_unet3p_cbsa_laspp_csftl_e50_bs6"],
        "builder": build_cbsa_laspp,
        "arch_note": "same architecture as CBSA+Lite-ASPP; loss-only change",
    },
    {
        "model_name": "ResNet34-UNet3+ + CBSA + PPM",
        "run_candidates": ["resnet34_unet3p_cbsa_ppm_e50_bs6"],
        "builder": build_cbsa_ppm,
        "arch_note": "alternative context module",
    },
    {
        "model_name": "ResNet34-UNet3+ + CBSA + OCR-Lite",
        "run_candidates": ["resnet34_unet3p_cbsa_ocr_lite_e50_bs6"],
        "builder": build_cbsa_ocr,
        "arch_note": "alternative context module",
    },
    {
        "model_name": "Edge + SimAM",
        "run_candidates": ["resnet34_unet3p_edge_simam_e50_bs6"],
        "builder": build_edge_simam,
        "arch_note": "old strong reference",
    },
]


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    rows = []
    for idx, spec in enumerate(MODEL_SPECS, start=1):
        run_name = resolve_run(spec["run_candidates"])
        row = {
            "model_order": idx,
            "model_name": spec["model_name"],
            "run_name": run_name,
            "arch_note": spec["arch_note"],
            "device": str(device),
            "image_size": args.image_size,
            "params_m": "",
            "gflops_512": "",
            "inference_ms_mean": "",
            "fps": "",
            "test_dice": "",
            "test_iou": "",
            "test_precision": "",
            "test_recall": "",
            "per_case_mean_dice": "",
            "per_case_mean_iou": "",
            "per_case_mean_precision": "",
            "per_case_mean_recall": "",
            "error": "",
        }

        print("=" * 80)
        print(f"[{idx}/{len(MODEL_SPECS)}] {spec['model_name']} ({run_name})")
        print("=" * 80)

        try:
            model = spec["builder"]()
            row["params_m"] = params_m(model)

            try:
                gflops, err = profile_gflops(model, args.image_size, device)
                row["gflops_512"] = gflops
                if err:
                    row["error"] += err
            except Exception as e:
                row["error"] += f" GFLOPs failed: {repr(e)}"

            # 重新创建，避免 thop hook 干扰速度测试
            model = spec["builder"]()
            try:
                ms, fps = measure_speed(model, args.image_size, device, args.warmup, args.repeat)
                row["inference_ms_mean"] = ms
                row["fps"] = fps
            except Exception as e:
                row["error"] += f" Speed failed: {repr(e)}"

            row.update(get_metric_from_run(run_name))

        except Exception as e:
            row["error"] += f" Build failed: {repr(e)}"

        print(json.dumps(row, ensure_ascii=False, indent=2))
        rows.append(row)

        if device.type == "cuda":
            torch.cuda.empty_cache()

    fields = [
        "model_order", "model_name", "run_name", "arch_note", "device", "image_size",
        "params_m", "gflops_512", "inference_ms_mean", "fps",
        "test_dice", "test_iou", "test_precision", "test_recall",
        "per_case_mean_dice", "per_case_mean_iou", "per_case_mean_precision", "per_case_mean_recall",
        "error",
    ]

    out_csv = OUT_DIR / "final_model_complexity.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fields})

    out_json = OUT_DIR / "final_model_complexity.json"
    save_json(rows, out_json)

    print("=" * 80)
    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
