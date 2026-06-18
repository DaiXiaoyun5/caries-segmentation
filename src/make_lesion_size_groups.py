#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
根据 test.txt 中每张图的 GT mask 前景面积，划分 small / medium / large lesion groups。

输出：
runs/group_analysis/lesion_size/test_lesion_size_groups.csv
runs/group_analysis/lesion_size/test_lesion_size_summary.json
runs/group_analysis/lesion_size/test_lesion_area_hist.png
"""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
TEST_SPLIT = PROJECT_ROOT / "data/splits/test.txt"
OUT_DIR = PROJECT_ROOT / "runs/group_analysis/lesion_size"


def read_split(split_file):
    items = []
    with open(split_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, mask_path = line.split("\t")
            items.append((Path(img_path), Path(mask_path)))
    return items


def mask_area(mask_path):
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    fg = arr > 0
    area = int(fg.sum())
    h, w = arr.shape
    ratio = float(area / (h * w))
    return area, ratio, h, w


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    items = read_split(TEST_SPLIT)
    print("=" * 80)
    print("Test split:", TEST_SPLIT)
    print("Num test samples:", len(items))
    print("=" * 80)

    rows = []
    for idx, (img_path, mask_path) in enumerate(items):
        area, ratio, h, w = mask_area(mask_path)
        case_id = img_path.stem

        rows.append({
            "index": idx,
            "case_id": case_id,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
            "gt_area": area,
            "gt_area_ratio": ratio,
            "height": h,
            "width": w,
        })

    # 按 gt_area 从小到大排序
    rows_sorted = sorted(rows, key=lambda x: x["gt_area"])
    n = len(rows_sorted)

    n_small = n // 3
    n_medium = n // 3
    # 剩下的给 large
    for rank, row in enumerate(rows_sorted):
        row["area_rank"] = rank + 1
        if rank < n_small:
            row["lesion_size_group"] = "small"
        elif rank < n_small + n_medium:
            row["lesion_size_group"] = "medium"
        else:
            row["lesion_size_group"] = "large"

    # 恢复原 test 顺序输出
    rows_out = sorted(rows_sorted, key=lambda x: x["index"])

    out_csv = OUT_DIR / "test_lesion_size_groups.csv"
    fieldnames = [
        "index",
        "case_id",
        "lesion_size_group",
        "area_rank",
        "gt_area",
        "gt_area_ratio",
        "height",
        "width",
        "image_path",
        "mask_path",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_out:
            writer.writerow(row)

    # 汇总统计
    summary = {
        "num_samples": n,
        "groups": {},
        "area_min": int(min(r["gt_area"] for r in rows)),
        "area_max": int(max(r["gt_area"] for r in rows)),
        "area_mean": float(np.mean([r["gt_area"] for r in rows])),
        "area_median": float(np.median([r["gt_area"] for r in rows])),
        "ratio_min": float(min(r["gt_area_ratio"] for r in rows)),
        "ratio_max": float(max(r["gt_area_ratio"] for r in rows)),
        "ratio_mean": float(np.mean([r["gt_area_ratio"] for r in rows])),
        "ratio_median": float(np.median([r["gt_area_ratio"] for r in rows])),
    }

    for group in ["small", "medium", "large"]:
        gs = [r for r in rows_sorted if r["lesion_size_group"] == group]
        areas = [r["gt_area"] for r in gs]
        ratios = [r["gt_area_ratio"] for r in gs]
        summary["groups"][group] = {
            "num_samples": len(gs),
            "area_min": int(min(areas)),
            "area_max": int(max(areas)),
            "area_mean": float(np.mean(areas)),
            "area_median": float(np.median(areas)),
            "ratio_min": float(min(ratios)),
            "ratio_max": float(max(ratios)),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_median": float(np.median(ratios)),
        }

    out_json = OUT_DIR / "test_lesion_size_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 画面积分布图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        areas = [r["gt_area"] for r in rows]
        plt.figure(figsize=(8, 5))
        plt.hist(areas, bins=30)
        plt.xlabel("GT lesion area (pixels)")
        plt.ylabel("Number of test images")
        plt.title("Distribution of GT lesion area in test set")
        plt.tight_layout()
        out_png = OUT_DIR / "test_lesion_area_hist.png"
        plt.savefig(out_png, dpi=150)
        plt.close()
        print("Saved histogram:", out_png)
    except Exception as e:
        print("Warning: failed to save histogram:", repr(e))

    print("=" * 80)
    print("Lesion size grouping finished.")
    print("Saved CSV :", out_csv)
    print("Saved JSON:", out_json)
    print("=" * 80)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
