#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
自动挑选论文可视化案例，并生成对比图。

输出目录：
runs/final_visual_cases/

每个案例输出一张 panel：
原图 / GT / UNet3+ / ResNet34-UNet3+ / CBSA / CBSA+Lite-ASPP / CBSA+PPM

挑选逻辑：
1. small_improved：小病灶中 CBSA+Lite-ASPP 相对 ResNet34-UNet3+ 提升较大的样本
2. hard_improved：困难样本中 CBSA+Lite-ASPP 提升较大的样本
3. ppm_small_better：小病灶中 PPM 优于 Lite-ASPP 的样本
4. failure_or_degraded：CBSA+Lite-ASPP 失败或退化样本

用法：
cd /share/home/u2515283028/caries_project
python src/make_final_visual_cases.py --num-each 2 --threshold 0.5
"""

import os
import sys
import csv
import json
import argparse
import importlib
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
SRC_DIR = PROJECT_ROOT / "src"
OUT_DIR = PROJECT_ROOT / "runs/final_visual_cases"
sys.path.insert(0, str(SRC_DIR))

from eval_per_case_metrics import build_model, load_checkpoint_state, extract_main_logits  # noqa

UNET3P_CANDIDATES = [
    "unet3p_baseline_e50_bs6_dp2_fair",
    "unet3p_baseline_e50_bs6_fair",
    "unet3p_baseline_e50",
    "unet3p_e50_bs6",
    "unet3p",
]

MODEL_SPECS_RAW = [
    {
        "key": "unet3p",
        "title": "UNet3+",
        "run_candidates": UNET3P_CANDIDATES,
        "module": "train_unet3p",
        "cls": "UNet3Plus",
    },
    {
        "key": "resnet34_unet3p",
        "title": "ResNet34-UNet3+",
        "run_candidates": ["resnet34_unet3p_baseline_e50"],
        "module": "train_resnet34_unet3p",
        "cls": "ResNet34UNet3Plus",
    },
    {
        "key": "cbsa",
        "title": "CBSA",
        "run_candidates": ["resnet34_unet3p_cbsa_e50_bs6"],
        "module": "train_resnet34_unet3p_cbsa",
        "cls": "ResNet34UNet3PlusCBSA",
    },
    {
        "key": "cbsa_laspp",
        "title": "CBSA+Lite-ASPP",
        "run_candidates": ["resnet34_unet3p_cbsa_laspp_e50_bs6"],
        "module": "train_resnet34_unet3p_cbsa_laspp",
        "cls": "ResNet34UNet3PlusCBSALiteASPP",
    },
    {
        "key": "cbsa_ppm",
        "title": "CBSA+PPM",
        "run_candidates": ["resnet34_unet3p_cbsa_ppm_e50_bs6"],
        "module": "train_resnet34_unet3p_cbsa_ppm",
        "cls": "ResNet34UNet3PlusCBSAPPMLite",
    },
]

LESION_GROUP_CSV = PROJECT_ROOT / "runs/group_analysis/lesion_size_512/test_lesion_size_groups_512.csv"
DIFF_GROUP_CSV = PROJECT_ROOT / "runs/group_analysis/top_cbsa_final/difficulty_groups_by_resnet34_unet3p.csv"


def read_csv_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def resolve_run(candidates, require_ckpt=True, require_per_case=True):
    for rn in candidates:
        rd = PROJECT_ROOT / "runs" / rn
        if not rd.exists():
            continue
        if require_ckpt and not (rd / "checkpoints/best.pth").exists():
            continue
        if require_per_case and not (rd / "per_case_metrics.csv").exists():
            continue
        return rn
    return None


def get_model_specs():
    specs = []
    missing = []
    for raw in MODEL_SPECS_RAW:
        rn = resolve_run(raw["run_candidates"], require_ckpt=True, require_per_case=True)
        if rn is None:
            missing.append(raw)
            continue
        spec = dict(raw)
        spec["run_name"] = rn
        specs.append(spec)
    return specs, missing


def per_case_map(run_name):
    path = PROJECT_ROOT / "runs" / run_name / "per_case_metrics.csv"
    return {r["case_id"]: r for r in read_csv_dict(path)}


def load_group_maps():
    lesion = {}
    if LESION_GROUP_CSV.exists():
        for r in read_csv_dict(LESION_GROUP_CSV):
            lesion[r["case_id"]] = r.get("lesion_size_group", "")

    diff = {}
    if DIFF_GROUP_CSV.exists():
        for r in read_csv_dict(DIFF_GROUP_CSV):
            diff[r["case_id"]] = r.get("difficulty_group", "")

    return lesion, diff


def select_cases(specs, num_each=2):
    keys = {s["key"]: s for s in specs}
    required = ["resnet34_unet3p", "cbsa_laspp"]
    for k in required:
        if k not in keys:
            raise RuntimeError(f"缺少 {k}，无法选案例。")

    base = per_case_map(keys["resnet34_unet3p"]["run_name"])
    main = per_case_map(keys["cbsa_laspp"]["run_name"])
    ppm = per_case_map(keys["cbsa_ppm"]["run_name"]) if "cbsa_ppm" in keys else {}
    lesion_map, diff_map = load_group_maps()

    rows = []
    for cid, b in base.items():
        if cid not in main:
            continue
        m = main[cid]
        p = ppm.get(cid)
        row = {
            "case_id": cid,
            "image_path": b.get("image_path", ""),
            "mask_path": b.get("mask_path", ""),
            "lesion_size_group": lesion_map.get(cid, ""),
            "difficulty_group": diff_map.get(cid, ""),
            "resnet_dice": to_float(b["dice"]),
            "main_dice": to_float(m["dice"]),
            "delta_main_vs_resnet": to_float(m["dice"]) - to_float(b["dice"]),
            "main_recall": to_float(m["recall"]),
            "main_precision": to_float(m["precision"]),
            "gt_area": to_float(b.get("gt_area", "")),
        }
        if p is not None:
            row["ppm_dice"] = to_float(p["dice"])
            row["delta_ppm_vs_main"] = to_float(p["dice"]) - to_float(m["dice"])
        else:
            row["ppm_dice"] = np.nan
            row["delta_ppm_vs_main"] = np.nan
        rows.append(row)

    selected = []
    used = set()

    def add(category, candidates):
        count = 0
        for r in candidates:
            cid = r["case_id"]
            if cid in used:
                continue
            item = dict(r)
            item["category"] = category
            selected.append(item)
            used.add(cid)
            count += 1
            if count >= num_each:
                break

    small_improved = sorted(
        [r for r in rows if r["lesion_size_group"] == "small"],
        key=lambda x: x["delta_main_vs_resnet"],
        reverse=True,
    )
    add("small_improved", small_improved)

    hard_improved = sorted(
        [r for r in rows if r["difficulty_group"] == "hard"],
        key=lambda x: x["delta_main_vs_resnet"],
        reverse=True,
    )
    add("hard_improved", hard_improved)

    ppm_small_better = sorted(
        [r for r in rows if r["lesion_size_group"] == "small" and not np.isnan(r["delta_ppm_vs_main"])],
        key=lambda x: x["delta_ppm_vs_main"],
        reverse=True,
    )
    add("ppm_small_better", ppm_small_better)

    failure = sorted(
        rows,
        key=lambda x: (x["main_dice"], x["delta_main_vs_resnet"]),
    )
    add("failure_or_degraded", failure)

    return selected


def load_model(spec, device):
    run_dir = PROJECT_ROOT / "runs" / spec["run_name"]
    cfg = load_json(run_dir / "config.json")
    module = importlib.import_module(spec["module"])
    model_cls = getattr(module, spec["cls"])
    model = build_model(model_cls, cfg)
    state = load_checkpoint_state(run_dir / "checkpoints/best.pth")
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {spec['title']} from {spec['run_name']} | missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device).eval()
    return model


def resize_image_and_mask(image_path, mask_path, image_size=512):
    img = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BILINEAR)
    mask = Image.open(mask_path).convert("L").resize((image_size, image_size), Image.NEAREST)
    img_np = np.array(img).astype(np.float32) / 255.0
    mask_np = (np.array(mask).astype(np.float32) > 0).astype(np.uint8)
    x = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float()
    return img, mask_np, x


def predict_mask(model, x, device, threshold=0.5):
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
        logits = extract_main_logits(out)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=True)
        prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    pred = (prob > threshold).astype(np.uint8)
    return pred, prob


def overlay_mask(img, mask, color=(255, 0, 0), alpha=0.45):
    arr = np.array(img).astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    m = mask.astype(bool)
    arr[m] = arr[m] * (1 - alpha) + color_arr * alpha
    return arr.astype(np.uint8)


def save_panel(case, specs, models, device, image_size=512, threshold=0.5):
    img, gt, x = resize_image_and_mask(case["image_path"], case["mask_path"], image_size=image_size)

    cols = []
    titles = []

    cols.append(np.array(img))
    titles.append("Image")

    cols.append(overlay_mask(img, gt, color=(255, 0, 0)))
    titles.append("GT")

    pred_records = {}
    for spec in specs:
        model = models[spec["key"]]
        pred, prob = predict_mask(model, x, device, threshold=threshold)
        pred_records[spec["key"]] = {
            "pred_area": int(pred.sum()),
            "prob_mean": float(prob.mean()),
        }
        cols.append(overlay_mask(img, pred, color=(255, 255, 0)))
        titles.append(spec["title"])

    n = len(cols)
    fig_w = max(12, n * 2.5)
    plt.figure(figsize=(fig_w, 3.2))
    for i, (arr, title) in enumerate(zip(cols, titles), start=1):
        ax = plt.subplot(1, n, i)
        ax.imshow(arr)
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    subtitle = (
        f"{case['category']} | {case['case_id']} | "
        f"lesion={case.get('lesion_size_group','')} | diff={case.get('difficulty_group','')} | "
        f"ΔDice={case.get('delta_main_vs_resnet', 0):.4f}"
    )
    plt.suptitle(subtitle, fontsize=10)
    plt.tight_layout()

    out_dir = OUT_DIR / case["category"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{case['case_id']}_{case['category']}.png"
    plt.savefig(out_png, dpi=180)
    plt.close()

    meta = dict(case)
    meta["panel_png"] = str(out_png)
    meta["pred_records"] = pred_records
    return meta


def main(args):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    specs, missing = get_model_specs()
    print("Available visual models:")
    for s in specs:
        print(f"- {s['title']} -> {s['run_name']}")
    if missing:
        print("Missing visual models:")
        for m in missing:
            print(m["title"], m["run_candidates"])

    selected = select_cases(specs, num_each=args.num_each)
    save_json(selected, OUT_DIR / "selected_cases_before_inference.json")

    # 为避免图太挤，最多保留前 max_models 个模型；默认所有可用模型都会展示。
    if args.max_models > 0:
        specs = specs[:args.max_models]

    models = {}
    for spec in specs:
        models[spec["key"]] = load_model(spec, device)

    results = []
    for i, case in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] Making panel for {case['case_id']} ({case['category']})")
        try:
            results.append(save_panel(case, specs, models, device, image_size=args.image_size, threshold=args.threshold))
        except Exception as e:
            bad = dict(case)
            bad["error"] = repr(e)
            results.append(bad)
            print("[WARN] failed:", repr(e))

    save_json(results, OUT_DIR / "visual_case_summary.json")
    print("=" * 80)
    print("Visual cases finished.")
    print("Output dir:", OUT_DIR)
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-each", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-models", type=int, default=0, help="0 means use all available visual models")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    main(args)
