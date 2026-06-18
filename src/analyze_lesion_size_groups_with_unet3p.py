#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并 512x512 小/中/大病灶分组文件和多个模型的 per_case_metrics.csv，
生成小病灶分组性能表。

输入：
runs/group_analysis/lesion_size_512/test_lesion_size_groups_512.csv
runs/{run_name}/per_case_metrics.csv

输出：
runs/group_analysis/lesion_size_512/lesion_size_group_metrics_with_unet3p.csv
runs/group_analysis/lesion_size_512/lesion_size_group_metrics_with_unet3p_wide.csv
runs/group_analysis/lesion_size_512/lesion_size_group_metrics_with_unet3p.xlsx
runs/group_analysis/lesion_size_512/lesion_size_group_metrics_with_unet3p.json
runs/group_analysis/lesion_size_512/lesion_size_dice_bar_with_unet3p.png
runs/group_analysis/lesion_size_512/lesion_size_recall_bar_with_unet3p.png
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
GROUP_DIR = PROJECT_ROOT / "runs/group_analysis/lesion_size_512"
GROUP_CSV = GROUP_DIR / "test_lesion_size_groups_512.csv"

MODELS = [
    {
        "run_name": "unet3p_baseline_e50_bs6_dp2_fair",
        "model_name": "UNet3+ baseline",
        "order": 1,
    },
    {
        "run_name": "resnet34_unet3p_baseline_e50",
        "model_name": "ResNet34-UNet3+",
        "order": 2,
    },
    {
        "run_name": "resnet34_unet3p_edge_e50",
        "model_name": "Edge",
        "order": 3,
    },
    {
        "run_name": "resnet34_unet3p_edge_simam_e50_bs6",
        "model_name": "Edge + SimAM",
        "order": 4,
    },
    {
        "run_name": "resnet34_unet3p_edge_simam_ra_e50_bs6",
        "model_name": "Edge + SimAM + RA",
        "order": 5,
    },
]

GROUP_ORDER = ["small", "medium", "large"]
METRICS = ["dice", "iou", "precision", "recall"]


def read_csv_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_dict(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def mean(xs):
    xs = [x for x in xs if not np.isnan(x)]
    if len(xs) == 0:
        return float("nan")
    return float(np.mean(xs))


def std(xs):
    xs = [x for x in xs if not np.isnan(x)]
    if len(xs) <= 1:
        return 0.0
    return float(np.std(xs, ddof=1))


def main():
    if not GROUP_CSV.exists():
        raise FileNotFoundError(f"分组文件不存在：{GROUP_CSV}")

    GROUP_DIR.mkdir(parents=True, exist_ok=True)

    group_rows = read_csv_dict(GROUP_CSV)
    group_map = {}

    for r in group_rows:
        case_id = r["case_id"]
        group_map[case_id] = {
            "lesion_size_group": r["lesion_size_group"],
            "gt_area_512": int(float(r["gt_area_512"])),
            "gt_area_ratio_512": float(r["gt_area_ratio_512"]),
            "area_rank": int(float(r["area_rank"])),
        }

    print("=" * 80)
    print("Loaded lesion size groups:", GROUP_CSV)
    print("num group cases:", len(group_map))
    print("=" * 80)

    long_rows = []
    merged_case_rows = []

    missing_files = []

    for m in MODELS:
        run_name = m["run_name"]
        model_name = m["model_name"]
        order = m["order"]

        per_case_path = PROJECT_ROOT / "runs" / run_name / "per_case_metrics.csv"

        if not per_case_path.exists():
            print(f"[MISSING] {per_case_path}")
            missing_files.append(str(per_case_path))
            continue

        rows = read_csv_dict(per_case_path)
        print(f"Loaded {model_name}: {len(rows)} cases from {per_case_path}")

        buckets = {g: [] for g in GROUP_ORDER}
        unmatched = []

        for r in rows:
            case_id = r["case_id"]

            if case_id not in group_map:
                unmatched.append(case_id)
                continue

            ginfo = group_map[case_id]
            group = ginfo["lesion_size_group"]

            item = {
                "case_id": case_id,
                "run_name": run_name,
                "model_name": model_name,
                "model_order": order,
                "lesion_size_group": group,
                "gt_area_512": ginfo["gt_area_512"],
                "gt_area_ratio_512": ginfo["gt_area_ratio_512"],
                "area_rank": ginfo["area_rank"],
                "dice": to_float(r["dice"]),
                "iou": to_float(r["iou"]),
                "precision": to_float(r["precision"]),
                "recall": to_float(r["recall"]),
                "pred_area": int(float(r["pred_area"])),
                "pred_area_ratio": to_float(r["pred_area_ratio"]),
                "image_path": r["image_path"],
                "mask_path": r["mask_path"],
            }

            buckets[group].append(item)
            merged_case_rows.append(item)

        if unmatched:
            print(f"[WARN] {model_name}: unmatched cases = {len(unmatched)}")
            print("first unmatched:", unmatched[:10])

        for group in GROUP_ORDER:
            items = buckets[group]
            out = {
                "model_order": order,
                "model_name": model_name,
                "run_name": run_name,
                "lesion_size_group": group,
                "num_cases": len(items),
            }

            for metric in METRICS:
                vals = [x[metric] for x in items]
                out[f"mean_{metric}"] = mean(vals)
                out[f"std_{metric}"] = std(vals)
                out[f"median_{metric}"] = float(np.median(vals)) if len(vals) else float("nan")

            # 面积统计也保留，方便论文写小/中/大病灶范围
            areas = [x["gt_area_512"] for x in items]
            ratios = [x["gt_area_ratio_512"] for x in items]
            out["gt_area_512_min"] = int(min(areas)) if areas else -1
            out["gt_area_512_max"] = int(max(areas)) if areas else -1
            out["gt_area_512_mean"] = mean(areas)
            out["gt_area_ratio_512_mean"] = mean(ratios)

            long_rows.append(out)

    if missing_files:
        print("=" * 80)
        print("[WARNING] 有模型缺少 per_case_metrics.csv：")
        for x in missing_files:
            print(x)
        print("缺失模型不会进入本次汇总。")
        print("=" * 80)

    # 排序
    long_rows = sorted(
        long_rows,
        key=lambda x: (int(x["model_order"]), GROUP_ORDER.index(x["lesion_size_group"]))
    )

    merged_case_rows = sorted(
        merged_case_rows,
        key=lambda x: (int(x["model_order"]), x["area_rank"])
    )

    out_long = GROUP_DIR / "lesion_size_group_metrics_with_unet3p.csv"
    long_fields = [
        "model_order", "model_name", "run_name", "lesion_size_group", "num_cases",
        "mean_dice", "std_dice", "median_dice",
        "mean_iou", "std_iou", "median_iou",
        "mean_precision", "std_precision", "median_precision",
        "mean_recall", "std_recall", "median_recall",
        "gt_area_512_min", "gt_area_512_max", "gt_area_512_mean",
        "gt_area_ratio_512_mean",
    ]
    write_csv_dict(out_long, long_rows, long_fields)

    # wide 表：每个模型一行
    wide_rows = []
    by_model = defaultdict(dict)

    for r in long_rows:
        key = (r["model_order"], r["model_name"], r["run_name"])
        g = r["lesion_size_group"]
        by_model[key][g] = r

    for key in sorted(by_model.keys(), key=lambda x: x[0]):
        order, model_name, run_name = key
        out = {
            "model_order": order,
            "model_name": model_name,
            "run_name": run_name,
        }

        for g in GROUP_ORDER:
            gr = by_model[key].get(g, {})
            out[f"{g}_num_cases"] = gr.get("num_cases", "")
            for metric in METRICS:
                out[f"{g}_{metric}"] = gr.get(f"mean_{metric}", "")

        # 重点增量：small 组相对两个 baseline
        wide_rows.append(out)

    # 计算 delta
    def find_model_row(model_name):
        for r in wide_rows:
            if r["model_name"] == model_name:
                return r
        return None

    unet3p_row = find_model_row("UNet3+ baseline")
    resnet_row = find_model_row("ResNet34-UNet3+")

    for r in wide_rows:
        for g in GROUP_ORDER:
            for metric in ["dice", "iou", "recall"]:
                val = r.get(f"{g}_{metric}", "")
                if val == "":
                    continue
                val = float(val)

                if unet3p_row is not None and unet3p_row.get(f"{g}_{metric}", "") != "":
                    r[f"{g}_{metric}_delta_vs_unet3p"] = val - float(unet3p_row[f"{g}_{metric}"])
                else:
                    r[f"{g}_{metric}_delta_vs_unet3p"] = ""

                if resnet_row is not None and resnet_row.get(f"{g}_{metric}", "") != "":
                    r[f"{g}_{metric}_delta_vs_resnet34_unet3p"] = val - float(resnet_row[f"{g}_{metric}"])
                else:
                    r[f"{g}_{metric}_delta_vs_resnet34_unet3p"] = ""

    out_wide = GROUP_DIR / "lesion_size_group_metrics_with_unet3p_wide.csv"
    wide_fields = [
        "model_order", "model_name", "run_name",
    ]
    for g in GROUP_ORDER:
        wide_fields += [
            f"{g}_num_cases",
            f"{g}_dice", f"{g}_iou", f"{g}_precision", f"{g}_recall",
            f"{g}_dice_delta_vs_unet3p",
            f"{g}_iou_delta_vs_unet3p",
            f"{g}_recall_delta_vs_unet3p",
            f"{g}_dice_delta_vs_resnet34_unet3p",
            f"{g}_iou_delta_vs_resnet34_unet3p",
            f"{g}_recall_delta_vs_resnet34_unet3p",
        ]

    write_csv_dict(out_wide, wide_rows, wide_fields)

    out_case = GROUP_DIR / "lesion_size_merged_per_case_with_unet3p.csv"
    case_fields = [
        "case_id", "model_name", "run_name", "model_order",
        "lesion_size_group", "gt_area_512", "gt_area_ratio_512", "area_rank",
        "dice", "iou", "precision", "recall",
        "pred_area", "pred_area_ratio",
        "image_path", "mask_path",
    ]
    write_csv_dict(out_case, merged_case_rows, case_fields)

    # json summary
    out_json = GROUP_DIR / "lesion_size_group_metrics_with_unet3p.json"
    summary = {
        "group_csv": str(GROUP_CSV),
        "models": MODELS,
        "missing_files": missing_files,
        "long_csv": str(out_long),
        "wide_csv": str(out_wide),
        "merged_per_case_csv": str(out_case),
        "long_rows": long_rows,
        "wide_rows": wide_rows,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 尝试保存 Excel
    try:
        import pandas as pd
        out_xlsx = GROUP_DIR / "lesion_size_group_metrics_with_unet3p.xlsx"
        with pd.ExcelWriter(out_xlsx) as writer:
            pd.DataFrame(long_rows).to_excel(writer, index=False, sheet_name="group_long")
            pd.DataFrame(wide_rows).to_excel(writer, index=False, sheet_name="group_wide")
            pd.DataFrame(merged_case_rows).to_excel(writer, index=False, sheet_name="per_case_merged")
        print("Saved Excel:", out_xlsx)
    except Exception as e:
        print("[WARN] failed to save Excel:", repr(e))

    # 画 Dice 和 Recall 柱状图
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot_metric(metric, out_png):
            models = [m["model_name"] for m in MODELS if (PROJECT_ROOT / "runs" / m["run_name"] / "per_case_metrics.csv").exists()]
            x = np.arange(len(GROUP_ORDER))
            width = 0.15

            plt.figure(figsize=(11, 6))

            for i, model_name in enumerate(models):
                vals = []
                for g in GROUP_ORDER:
                    matched = [
                        r for r in long_rows
                        if r["model_name"] == model_name and r["lesion_size_group"] == g
                    ]
                    if matched:
                        vals.append(float(matched[0][f"mean_{metric}"]))
                    else:
                        vals.append(np.nan)

                offset = (i - (len(models) - 1) / 2) * width
                plt.bar(x + offset, vals, width, label=model_name)

            plt.xticks(x, ["Small", "Medium", "Large"])
            plt.ylabel(metric.capitalize())
            plt.title(f"Lesion size group analysis: {metric}")
            plt.ylim(0, 1.0)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()

        out_dice_png = GROUP_DIR / "lesion_size_dice_bar_with_unet3p.png"
        out_recall_png = GROUP_DIR / "lesion_size_recall_bar_with_unet3p.png"

        plot_metric("dice", out_dice_png)
        plot_metric("recall", out_recall_png)

        print("Saved figure:", out_dice_png)
        print("Saved figure:", out_recall_png)

    except Exception as e:
        print("[WARN] failed to save figures:", repr(e))

    print("=" * 80)
    print("Lesion size group analysis finished.")
    print("Saved long CSV :", out_long)
    print("Saved wide CSV :", out_wide)
    print("Saved case CSV :", out_case)
    print("Saved JSON     :", out_json)
    print("=" * 80)

    print("Preview wide rows:")
    for r in wide_rows:
        print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
