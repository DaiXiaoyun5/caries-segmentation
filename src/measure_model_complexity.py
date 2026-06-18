#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Measure Params and GFLOPs for main models.

Outputs:
runs/model_complexity/model_complexity_params_gflops.csv
runs/model_complexity/model_complexity_params_gflops.json

FLOPs measured with thop on input: [1, 3, 512, 512]
"""

import os
import sys
import json
import inspect
import subprocess
from pathlib import Path

import torch
import torch.nn as nn

import segmentation_models_pytorch as smp


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
SRC_DIR = PROJECT_ROOT / "src"
OUT_DIR = PROJECT_ROOT / "runs/model_complexity"
sys.path.insert(0, str(SRC_DIR))


def ensure_thop():
    try:
        import thop  # noqa
        return
    except Exception:
        print("thop not found, installing...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "thop",
            "-i", "https://pypi.tuna.tsinghua.edu.cn/simple",
            "--proxy", "http://211.67.63.75:3128"
        ])


def params_m(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


class MainOutputWrapper(nn.Module):
    """
    Make all custom outputs become a single tensor for FLOPs profiling.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)

        if isinstance(outputs, tuple):
            seg_outputs = outputs[0]
        else:
            seg_outputs = outputs

        if isinstance(seg_outputs, (list, tuple)):
            return seg_outputs[0]

        return seg_outputs


def profile_gflops(model, image_size=512):
    ensure_thop()
    from thop import profile

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    wrapped = MainOutputWrapper(model).to(device)
    dummy = torch.randn(1, 3, image_size, image_size).to(device)

    with torch.no_grad():
        flops, params = profile(wrapped, inputs=(dummy,), verbose=False)

    return flops / 1e9


def load_json(path):
    if not Path(path).exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_test_metrics(run_name):
    path = PROJECT_ROOT / "runs" / run_name / "test_metrics.json"
    if not path.exists():
        return {}
    return load_json(path)


def get_summary_metrics(run_name):
    path = PROJECT_ROOT / "runs" / run_name / "summary_metrics.json"
    if not path.exists():
        return {}
    return load_json(path)


def build_resnet34_unet3p():
    from train_resnet34_unet3p import ResNet34UNet3Plus
    return ResNet34UNet3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
    )


def build_resnet34_unet3p_edge():
    from train_resnet34_unet3p_edge import ResNet34UNet3PlusEdge
    return ResNet34UNet3PlusEdge(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
    )


def build_resnet34_unet3p_edge_simam():
    from train_resnet34_unet3p_edge_simam import ResNet34UNet3PlusEdgeSimAM
    return ResNet34UNet3PlusEdgeSimAM(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cat_ch=64,
        deep_supervision=True,
        simam_lambda=1e-4,
        simam_position="all",
    )


def build_resnet34_unet3p_edge_simam_ra():
    from train_resnet34_unet3p_edge_simam_ra import ResNet34UNet3PlusEdgeSimAMRA
    return ResNet34UNet3PlusEdgeSimAMRA(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cat_channels=64,
        deep_supervision=True,
        simam_lambda=1e-4,
        simam_position="all",
        ra_scale=1.0,
    )


def build_unet3p():
    from train_unet3p import UNet3Plus
    return UNet3Plus(
        in_channels=3,
        num_classes=1,
        base_ch=64,
        deep_supervision=True,
    )


def build_unetpp():
    return smp.UnetPlusPlus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


def build_deeplabv3plus():
    return smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


def build_pspnet():
    return smp.PSPNet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


MODELS = [
    {
        "model_name": "UNet3+ baseline",
        "run_name": "unet3p_baseline_e50_bs6_dp2_fair",
        "builder": build_unet3p,
    },
    {
        "model_name": "UNet++ baseline",
        "run_name": "unetpp_baseline_v3_bs6",
        "builder": build_unetpp,
    },
    {
        "model_name": "ResNet34-UNet3+",
        "run_name": "resnet34_unet3p_baseline_e50",
        "builder": build_resnet34_unet3p,
    },
    {
        "model_name": "Edge",
        "run_name": "resnet34_unet3p_edge_e50",
        "builder": build_resnet34_unet3p_edge,
    },
    {
        "model_name": "Edge + SimAM",
        "run_name": "resnet34_unet3p_edge_simam_e50_bs6",
        "builder": build_resnet34_unet3p_edge_simam,
    },
    {
        "model_name": "Edge + SimAM + RA",
        "run_name": "resnet34_unet3p_edge_simam_ra_e50_bs6",
        "builder": build_resnet34_unet3p_edge_simam_ra,
    },
    {
        "model_name": "DeepLabV3+",
        "run_name": "deeplabv3plus_resnet34_e50_bs6",
        "builder": build_deeplabv3plus,
    },
    {
        "model_name": "PSPNet",
        "run_name": "pspnet_resnet34_e50_bs6",
        "builder": build_pspnet,
    },
]


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for idx, item in enumerate(MODELS, start=1):
        model_name = item["model_name"]
        run_name = item["run_name"]

        print("=" * 80)
        print(f"[{idx}/{len(MODELS)}] Measuring: {model_name} ({run_name})")
        print("=" * 80)

        row = {
            "model_name": model_name,
            "run_name": run_name,
            "params_m": "",
            "gflops_512": "",
            "dice": "",
            "iou": "",
            "precision": "",
            "recall": "",
            "error": "",
        }

        try:
            model = item["builder"]()
            row["params_m"] = params_m(model)

            try:
                row["gflops_512"] = profile_gflops(model, image_size=512)
            except Exception as e:
                row["gflops_512"] = ""
                row["error"] += f" FLOPs failed: {repr(e)}"

            metrics = get_test_metrics(run_name)
            if metrics:
                row["dice"] = metrics.get("dice", "")
                row["iou"] = metrics.get("iou", "")
                row["precision"] = metrics.get("precision", "")
                row["recall"] = metrics.get("recall", "")

        except Exception as e:
            row["error"] = repr(e)

        print(json.dumps(row, indent=2, ensure_ascii=False))
        rows.append(row)

    out_csv = OUT_DIR / "model_complexity_params_gflops.csv"
    fields = ["model_name", "run_name", "params_m", "gflops_512", "dice", "iou", "precision", "recall", "error"]

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            values = []
            for k in fields:
                v = str(r.get(k, ""))
                v = v.replace(",", ";")
                values.append(v)
            f.write(",".join(values) + "\n")

    out_json = OUT_DIR / "model_complexity_params_gflops.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Saved:", out_csv)
    print("Saved:", out_json)
    print("=" * 80)


if __name__ == "__main__":
    main()
