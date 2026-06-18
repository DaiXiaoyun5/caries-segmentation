#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
困难样本分组 + 逐图提升分析 + 错误类型统计。

分组依据：
- 使用 ResNet34-UNet3+ baseline 的逐图 Dice
- Dice 最低 1/3: hard
- 中间 1/3: medium
- 最高 1/3: easy

参与模型：
1. UNet3+ baseline
2. ResNet34-UNet3+
3. Edge
4. Edge + SimAM
5. Edge + SimAM + RA

输出：
runs/group_analysis/difficulty/difficulty_groups_by_resnet34_unet3p.csv
runs/group_analysis/difficulty/difficulty_group_metrics_with_unet3p.csv
runs/group_analysis/difficulty/difficulty_group_metrics_with_unet3p_wide.csv
runs/group_analysis/difficulty/difficulty_group_metrics_with_unet3p.xlsx
runs/group_analysis/difficulty/difficulty_dice_bar_with_unet3p.png
runs/group_analysis/difficulty/difficulty_recall_bar_with_unet3p.png

runs/group_analysis/delta/delta_edge_simam_vs_resnet34_unet3p.csv
runs/group_analysis/delta/delta_summary_edge_simam_vs_resnet34_unet3p.json
runs/group_analysis/delta/delta_dice_hist_edge_simam_vs_resnet34_unet3p.png
runs/group_analysis/delta/baseline_dice_vs_delta_dice.png

runs/group_analysis/error_analysis/error_type_counts_with_unet3p.csv
runs/group_analysis/error_analysis/error_type_counts_with_unet3p.xlsx
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")

DIFF_DIR = PROJECT_ROOT / "runs/group_analysis/difficulty"
DELTA_DIR = PROJECT_ROOT / "runs/group_analysis/delta"
ERROR_DIR = PROJECT_ROOT / "runs/group_analysis/error_analysis"

BASELINE_RUN = "resnet34_unet3p_baseline_e50"

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

GROUP_ORDER = ["hard", "medium", "easy"]
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


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def mean(xs):
    xs = [float(x) for x in xs if not np.isnan(float(x))]
    if len(xs) == 0:
        return float("nan")
    return float(np.mean(xs))


def std(xs):
    xs = [float(x) for x in xs if not np.isnan(float(x))]
    if len(xs) <= 1:
        return 0.0
    return float(np.std(xs, ddof=1))


def median(xs):
    xs = [float(x) for x in xs if not np.isnan(float(x))]
    if len(xs) == 0:
        return float("nan")
    return float(np.median(xs))


def get_per_case_path(run_name):
    return PROJECT_ROOT / "runs" / run_name / "per_case_metrics.csv"


def make_difficulty_groups():
    baseline_csv = get_per_case_path(BASELINE_RUN)
    if not baseline_csv.exists():
        raise FileNotFoundError(f"缺少 baseline per_case_metrics.csv: {baseline_csv}")

    rows = read_csv_dict(baseline_csv)
    rows_sorted = sorted(rows, key=lambda r: to_float(r["dice"]))

    n = len(rows_sorted)
    n_hard = n // 3
    n_medium = n // 3

    group_rows = []
    group_map = {}

    for rank, r in enumerate(rows_sorted):
        case_id = r["case_id"]
        baseline_dice = to_float(r["dice"])

        if rank < n_hard:
            group = "hard"
        elif rank < n_hard + n_medium:
            group = "medium"
        else:
            group = "easy"

        out = {
            "case_id": case_id,
            "difficulty_group": group,
            "difficulty_rank": rank + 1,
            "baseline_run": BASELINE_RUN,
            "baseline_dice": baseline_dice,
            "baseline_iou": to_float(r["iou"]),
            "baseline_precision": to_float(r["precision"]),
            "baseline_recall": to_float(r["recall"]),
            "gt_area": int(float(r["gt_area"])),
            "gt_area_ratio": to_float(r["gt_area_ratio"]),
            "image_path": r["image_path"],
            "mask_path": r["mask_path"],
        }

        group_rows.append(out)
        group_map[case_id] = out

    out_csv = DIFF_DIR / "difficulty_groups_by_resnet34_unet3p.csv"
    fields = [
        "case_id",
        "difficulty_group",
        "difficulty_rank",
        "baseline_run",
        "baseline_dice",
        "baseline_iou",
        "baseline_precision",
        "baseline_recall",
        "gt_area",
        "gt_area_ratio",
        "image_path",
        "mask_path",
    ]
    write_csv_dict(out_csv, group_rows, fields)

    # 摘要
    summary = {
        "baseline_run": BASELINE_RUN,
        "num_cases": n,
        "groups": {},
    }
    for g in GROUP_ORDER:
        gs = [r for r in group_rows if r["difficulty_group"] == g]
        summary["groups"][g] = {
            "num_cases": len(gs),
            "baseline_dice_min": mean([min([x["baseline_dice"] for x in gs])] if gs else [np.nan]),
            "baseline_dice_max": mean([max([x["baseline_dice"] for x in gs])] if gs else [np.nan]),
            "baseline_dice_mean": mean([x["baseline_dice"] for x in gs]),
            "baseline_dice_median": median([x["baseline_dice"] for x in gs]),
        }

    save_json(summary, DIFF_DIR / "difficulty_groups_summary_by_resnet34_unet3p.json")

    print("=" * 80)
    print("Difficulty groups generated.")
    print("Saved:", out_csv)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)

    return group_map, group_rows


def analyze_group_metrics(group_map):
    long_rows = []
    wide_rows = []
    merged_case_rows = []
    missing = []

    for m in MODELS:
        run_name = m["run_name"]
        model_name = m["model_name"]
        order = m["order"]
        per_case = get_per_case_path(run_name)

        if not per_case.exists():
            print(f"[MISSING] {per_case}")
            missing.append(str(per_case))
            continue

        rows = read_csv_dict(per_case)
        buckets = {g: [] for g in GROUP_ORDER}
        unmatched = []

        for r in rows:
            case_id = r["case_id"]
            if case_id not in group_map:
                unmatched.append(case_id)
                continue

            ginfo = group_map[case_id]
            group = ginfo["difficulty_group"]

            item = {
                "case_id": case_id,
                "model_order": order,
                "model_name": model_name,
                "run_name": run_name,
                "difficulty_group": group,
                "difficulty_rank": ginfo["difficulty_rank"],
                "baseline_dice_for_grouping": ginfo["baseline_dice"],
                "gt_area": int(float(r["gt_area"])),
                "gt_area_ratio": to_float(r["gt_area_ratio"]),
                "pred_area": int(float(r["pred_area"])),
                "pred_area_ratio": to_float(r["pred_area_ratio"]),
                "dice": to_float(r["dice"]),
                "iou": to_float(r["iou"]),
                "precision": to_float(r["precision"]),
                "recall": to_float(r["recall"]),
                "image_path": r["image_path"],
                "mask_path": r["mask_path"],
            }

            buckets[group].append(item)
            merged_case_rows.append(item)

        if unmatched:
            print(f"[WARN] {model_name}: unmatched cases {len(unmatched)}, first={unmatched[:5]}")

        for g in GROUP_ORDER:
            items = buckets[g]
            out = {
                "model_order": order,
                "model_name": model_name,
                "run_name": run_name,
                "difficulty_group": g,
                "num_cases": len(items),
            }

            for metric in METRICS:
                vals = [x[metric] for x in items]
                out[f"mean_{metric}"] = mean(vals)
                out[f"std_{metric}"] = std(vals)
                out[f"median_{metric}"] = median(vals)

            out["baseline_dice_for_grouping_mean"] = mean([x["baseline_dice_for_grouping"] for x in items])
            out["gt_area_mean"] = mean([x["gt_area"] for x in items])
            out["gt_area_ratio_mean"] = mean([x["gt_area_ratio"] for x in items])

            long_rows.append(out)

    long_rows = sorted(long_rows, key=lambda x: (int(x["model_order"]), GROUP_ORDER.index(x["difficulty_group"])))

    # wide
    by_model = defaultdict(dict)
    for r in long_rows:
        key = (r["model_order"], r["model_name"], r["run_name"])
        by_model[key][r["difficulty_group"]] = r

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

        wide_rows.append(out)

    # delta vs UNet3+ 和 ResNet34 baseline
    def find_row(model_name):
        for r in wide_rows:
            if r["model_name"] == model_name:
                return r
        return None

    unet3p_row = find_row("UNet3+ baseline")
    resnet_row = find_row("ResNet34-UNet3+")

    for r in wide_rows:
        for g in GROUP_ORDER:
            for metric in ["dice", "iou", "recall", "precision"]:
                val = r.get(f"{g}_{metric}", "")
                if val == "":
                    r[f"{g}_{metric}_delta_vs_unet3p"] = ""
                    r[f"{g}_{metric}_delta_vs_resnet34_unet3p"] = ""
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

    out_long = DIFF_DIR / "difficulty_group_metrics_with_unet3p.csv"
    long_fields = [
        "model_order", "model_name", "run_name", "difficulty_group", "num_cases",
        "mean_dice", "std_dice", "median_dice",
        "mean_iou", "std_iou", "median_iou",
        "mean_precision", "std_precision", "median_precision",
        "mean_recall", "std_recall", "median_recall",
        "baseline_dice_for_grouping_mean",
        "gt_area_mean",
        "gt_area_ratio_mean",
    ]
    write_csv_dict(out_long, long_rows, long_fields)

    out_wide = DIFF_DIR / "difficulty_group_metrics_with_unet3p_wide.csv"
    wide_fields = ["model_order", "model_name", "run_name"]
    for g in GROUP_ORDER:
        wide_fields += [
            f"{g}_num_cases",
            f"{g}_dice", f"{g}_iou", f"{g}_precision", f"{g}_recall",
            f"{g}_dice_delta_vs_unet3p",
            f"{g}_iou_delta_vs_unet3p",
            f"{g}_recall_delta_vs_unet3p",
            f"{g}_precision_delta_vs_unet3p",
            f"{g}_dice_delta_vs_resnet34_unet3p",
            f"{g}_iou_delta_vs_resnet34_unet3p",
            f"{g}_recall_delta_vs_resnet34_unet3p",
            f"{g}_precision_delta_vs_resnet34_unet3p",
        ]
    write_csv_dict(out_wide, wide_rows, wide_fields)

    out_case = DIFF_DIR / "difficulty_merged_per_case_with_unet3p.csv"
    case_fields = [
        "case_id", "model_order", "model_name", "run_name",
        "difficulty_group", "difficulty_rank", "baseline_dice_for_grouping",
        "gt_area", "gt_area_ratio", "pred_area", "pred_area_ratio",
        "dice", "iou", "precision", "recall",
        "image_path", "mask_path",
    ]
    merged_case_rows = sorted(merged_case_rows, key=lambda x: (x["model_order"], x["difficulty_rank"]))
    write_csv_dict(out_case, merged_case_rows, case_fields)

    # json
    summary = {
        "baseline_run_for_grouping": BASELINE_RUN,
        "models": MODELS,
        "missing_files": missing,
        "long_csv": str(out_long),
        "wide_csv": str(out_wide),
        "merged_per_case_csv": str(out_case),
        "long_rows": long_rows,
        "wide_rows": wide_rows,
    }
    save_json(summary, DIFF_DIR / "difficulty_group_metrics_with_unet3p.json")

    # Excel
    try:
        import pandas as pd
        out_xlsx = DIFF_DIR / "difficulty_group_metrics_with_unet3p.xlsx"
        with pd.ExcelWriter(out_xlsx) as writer:
            pd.DataFrame(long_rows).to_excel(writer, index=False, sheet_name="group_long")
            pd.DataFrame(wide_rows).to_excel(writer, index=False, sheet_name="group_wide")
            pd.DataFrame(merged_case_rows).to_excel(writer, index=False, sheet_name="per_case_merged")
        print("Saved Excel:", out_xlsx)
    except Exception as e:
        print("[WARN] failed to save Excel:", repr(e))

    # plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def plot_metric(metric, out_png):
            existing_models = []
            for m in MODELS:
                if get_per_case_path(m["run_name"]).exists():
                    existing_models.append(m["model_name"])

            x = np.arange(len(GROUP_ORDER))
            width = 0.15

            plt.figure(figsize=(11, 6))
            for i, model_name in enumerate(existing_models):
                vals = []
                for g in GROUP_ORDER:
                    matched = [r for r in long_rows if r["model_name"] == model_name and r["difficulty_group"] == g]
                    vals.append(float(matched[0][f"mean_{metric}"]) if matched else np.nan)
                offset = (i - (len(existing_models) - 1) / 2) * width
                plt.bar(x + offset, vals, width, label=model_name)

            plt.xticks(x, ["Hard", "Medium", "Easy"])
            plt.ylabel(metric.capitalize())
            plt.title(f"Difficulty group analysis: {metric}")
            plt.ylim(0, 1.0)
            plt.legend(fontsize=8)
            plt.tight_layout()
            plt.savefig(out_png, dpi=150)
            plt.close()

        plot_metric("dice", DIFF_DIR / "difficulty_dice_bar_with_unet3p.png")
        plot_metric("recall", DIFF_DIR / "difficulty_recall_bar_with_unet3p.png")
        print("Saved difficulty figures.")
    except Exception as e:
        print("[WARN] failed to save plots:", repr(e))

    print("=" * 80)
    print("Difficulty group analysis finished.")
    print("Saved:", out_long)
    print("Saved:", out_wide)
    print("=" * 80)
    print("Preview wide rows:")
    for r in wide_rows:
        print(json.dumps(r, ensure_ascii=False, indent=2))

    return wide_rows, merged_case_rows


def analyze_delta():
    base_csv = get_per_case_path("resnet34_unet3p_baseline_e50")
    ours_csv = get_per_case_path("resnet34_unet3p_edge_simam_e50_bs6")

    if not base_csv.exists() or not ours_csv.exists():
        print("[WARN] cannot run delta analysis, missing baseline or Edge+SimAM per-case csv.")
        return

    base_rows = {r["case_id"]: r for r in read_csv_dict(base_csv)}
    ours_rows = {r["case_id"]: r for r in read_csv_dict(ours_csv)}

    rows = []
    for case_id, b in base_rows.items():
        if case_id not in ours_rows:
            continue
        o = ours_rows[case_id]

        out = {
            "case_id": case_id,
            "baseline_dice": to_float(b["dice"]),
            "edge_simam_dice": to_float(o["dice"]),
            "delta_dice": to_float(o["dice"]) - to_float(b["dice"]),
            "baseline_iou": to_float(b["iou"]),
            "edge_simam_iou": to_float(o["iou"]),
            "delta_iou": to_float(o["iou"]) - to_float(b["iou"]),
            "baseline_precision": to_float(b["precision"]),
            "edge_simam_precision": to_float(o["precision"]),
            "delta_precision": to_float(o["precision"]) - to_float(b["precision"]),
            "baseline_recall": to_float(b["recall"]),
            "edge_simam_recall": to_float(o["recall"]),
            "delta_recall": to_float(o["recall"]) - to_float(b["recall"]),
            "gt_area": int(float(b["gt_area"])),
            "gt_area_ratio": to_float(b["gt_area_ratio"]),
            "image_path": b["image_path"],
            "mask_path": b["mask_path"],
        }
        rows.append(out)

    rows = sorted(rows, key=lambda r: r["delta_dice"], reverse=True)

    out_csv = DELTA_DIR / "delta_edge_simam_vs_resnet34_unet3p.csv"
    fields = [
        "case_id",
        "baseline_dice", "edge_simam_dice", "delta_dice",
        "baseline_iou", "edge_simam_iou", "delta_iou",
        "baseline_precision", "edge_simam_precision", "delta_precision",
        "baseline_recall", "edge_simam_recall", "delta_recall",
        "gt_area", "gt_area_ratio",
        "image_path", "mask_path",
    ]
    write_csv_dict(out_csv, rows, fields)

    deltas = [r["delta_dice"] for r in rows]
    delta_recalls = [r["delta_recall"] for r in rows]
    delta_precisions = [r["delta_precision"] for r in rows]

    summary = {
        "comparison": "Edge + SimAM vs ResNet34-UNet3+",
        "num_cases": len(rows),
        "mean_delta_dice": mean(deltas),
        "median_delta_dice": median(deltas),
        "mean_delta_recall": mean(delta_recalls),
        "median_delta_recall": median(delta_recalls),
        "mean_delta_precision": mean(delta_precisions),
        "median_delta_precision": median(delta_precisions),
        "num_delta_dice_positive": int(sum(1 for x in deltas if x > 0)),
        "num_delta_dice_negative": int(sum(1 for x in deltas if x < 0)),
        "num_delta_dice_gt_0_01": int(sum(1 for x in deltas if x > 0.01)),
        "num_delta_dice_gt_0_03": int(sum(1 for x in deltas if x > 0.03)),
        "num_delta_dice_gt_0_05": int(sum(1 for x in deltas if x > 0.05)),
        "top20_improved_cases": rows[:20],
        "top20_degraded_cases": sorted(rows, key=lambda r: r["delta_dice"])[:20],
        "csv": str(out_csv),
    }

    save_json(summary, DELTA_DIR / "delta_summary_edge_simam_vs_resnet34_unet3p.json")

    try:
        import pandas as pd
        out_xlsx = DELTA_DIR / "delta_edge_simam_vs_resnet34_unet3p.xlsx"
        with pd.ExcelWriter(out_xlsx) as writer:
            pd.DataFrame(rows).to_excel(writer, index=False, sheet_name="delta_all")
            pd.DataFrame(rows[:20]).to_excel(writer, index=False, sheet_name="top20_improved")
            pd.DataFrame(sorted(rows, key=lambda r: r["delta_dice"])[:20]).to_excel(writer, index=False, sheet_name="top20_degraded")
        print("Saved delta Excel:", out_xlsx)
    except Exception as e:
        print("[WARN] failed to save delta Excel:", repr(e))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        plt.hist(deltas, bins=30)
        plt.xlabel("Delta Dice (Edge+SimAM - ResNet34-UNet3+)")
        plt.ylabel("Number of test images")
        plt.title("Distribution of per-case Dice improvement")
        plt.tight_layout()
        plt.savefig(DELTA_DIR / "delta_dice_hist_edge_simam_vs_resnet34_unet3p.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.scatter([r["baseline_dice"] for r in rows], [r["delta_dice"] for r in rows], s=12)
        plt.axhline(0, linestyle="--")
        plt.xlabel("Baseline Dice")
        plt.ylabel("Delta Dice")
        plt.title("Baseline Dice vs Delta Dice")
        plt.tight_layout()
        plt.savefig(DELTA_DIR / "baseline_dice_vs_delta_dice.png", dpi=150)
        plt.close()

        print("Saved delta plots.")
    except Exception as e:
        print("[WARN] failed to save delta plots:", repr(e))

    print("=" * 80)
    print("Delta analysis finished.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=" * 80)


def analyze_error_counts(group_map):
    rows = []

    for m in MODELS:
        run_name = m["run_name"]
        model_name = m["model_name"]
        order = m["order"]
        per_case = get_per_case_path(run_name)

        if not per_case.exists():
            continue

        pcs = read_csv_dict(per_case)

        for scope in ["all", "hard", "medium", "easy"]:
            if scope == "all":
                items = pcs
            else:
                items = [r for r in pcs if r["case_id"] in group_map and group_map[r["case_id"]]["difficulty_group"] == scope]

            out = {
                "model_order": order,
                "model_name": model_name,
                "run_name": run_name,
                "scope": scope,
                "num_cases": len(items),
                "num_dice_lt_0_60": sum(1 for r in items if to_float(r["dice"]) < 0.60),
                "num_dice_lt_0_70": sum(1 for r in items if to_float(r["dice"]) < 0.70),
                "num_recall_lt_0_70": sum(1 for r in items if to_float(r["recall"]) < 0.70),
                "num_precision_lt_0_70": sum(1 for r in items if to_float(r["precision"]) < 0.70),
                "mean_dice": mean([to_float(r["dice"]) for r in items]),
                "mean_iou": mean([to_float(r["iou"]) for r in items]),
                "mean_precision": mean([to_float(r["precision"]) for r in items]),
                "mean_recall": mean([to_float(r["recall"]) for r in items]),
            }
            rows.append(out)

    out_csv = ERROR_DIR / "error_type_counts_with_unet3p.csv"
    fields = [
        "model_order", "model_name", "run_name", "scope", "num_cases",
        "num_dice_lt_0_60", "num_dice_lt_0_70",
        "num_recall_lt_0_70", "num_precision_lt_0_70",
        "mean_dice", "mean_iou", "mean_precision", "mean_recall",
    ]
    write_csv_dict(out_csv, rows, fields)

    try:
        import pandas as pd
        out_xlsx = ERROR_DIR / "error_type_counts_with_unet3p.xlsx"
        pd.DataFrame(rows).to_excel(out_xlsx, index=False)
        print("Saved error counts Excel:", out_xlsx)
    except Exception as e:
        print("[WARN] failed to save error Excel:", repr(e))

    print("=" * 80)
    print("Error type count analysis finished.")
    print("Saved:", out_csv)
    print("=" * 80)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False, indent=2))


def main():
    for d in [DIFF_DIR, DELTA_DIR, ERROR_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Check required per_case_metrics.csv files")
    print("=" * 80)

    for m in MODELS:
        p = get_per_case_path(m["run_name"])
        print(m["model_name"], "=>", p, "exists=", p.exists())
        if not p.exists():
            print("[WARNING] Missing file. This model will be skipped if not generated.")

    group_map, group_rows = make_difficulty_groups()
    analyze_group_metrics(group_map)
    analyze_delta()
    analyze_error_counts(group_map)

    print("=" * 80)
    print("All analyses finished.")
    print("Difficulty dir:", DIFF_DIR)
    print("Delta dir     :", DELTA_DIR)
    print("Error dir     :", ERROR_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
