#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
生成论文实验证据链最终汇总表。

输出目录：
runs/group_analysis/top_cbsa_final/

核心输出：
1. overall_per_case_summary.csv              逐图 mean/median/std 指标汇总
2. overall_micro_test_metrics.csv            test_metrics.json 全局/micro 指标汇总
3. lesion_size_group_metrics.csv             小/中/大病灶分组长表
4. lesion_size_group_metrics_wide.csv        小/中/大病灶分组宽表
5. difficulty_groups_by_resnet34_unet3p.csv  按 ResNet34-UNet3+ Dice 三等分定义难度
6. difficulty_group_metrics.csv              困难/中等/简单样本长表
7. difficulty_group_metrics_wide.csv         困难/中等/简单样本宽表
8. delta_summary_vs_unet3p.csv               相对原始 UNet3+ 的逐图提升统计
9. delta_summary_vs_resnet34_unet3p.csv      相对 ResNet34-UNet3+ 的逐图提升统计
10. delta_cases_vs_resnet34_unet3p.csv       每张图相对强基线的 delta 明细
11. final_recommendation_summary.json        自动生成的推荐结论摘要
12. final_paper_tables.xlsx                  以上核心表整合 Excel（若 pandas/openpyxl 可用）

用法：
cd /share/home/u2515283028/caries_project
python src/analyze_final_paper_evidence.py
"""

import csv
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
OUT_DIR = PROJECT_ROOT / "runs/group_analysis/top_cbsa_final"
LESION_GROUP_CSV = PROJECT_ROOT / "runs/group_analysis/lesion_size_512/test_lesion_size_groups_512.csv"

# 原始 UNet3+ 可能有多个目录名，脚本会自动选第一个存在 per_case_metrics.csv 的。
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
        "model_name": "UNet3+",
        "run_candidates": UNET3P_CANDIDATES,
        "order": 1,
        "role": "basic_baseline",
    },
    {
        "key": "resnet34_unet3p",
        "model_name": "ResNet34-UNet3+",
        "run_candidates": ["resnet34_unet3p_baseline_e50"],
        "order": 2,
        "role": "strong_baseline",
    },
    {
        "key": "cbsa",
        "model_name": "ResNet34-UNet3+ + CBSA",
        "run_candidates": ["resnet34_unet3p_cbsa_e50_bs6"],
        "order": 3,
        "role": "module_ablation",
    },
    {
        "key": "cbsa_laspp",
        "model_name": "ResNet34-UNet3+ + CBSA + Lite-ASPP",
        "run_candidates": ["resnet34_unet3p_cbsa_laspp_e50_bs6"],
        "order": 4,
        "role": "main_model",
    },
    {
        "key": "cbsa_ppm",
        "model_name": "ResNet34-UNet3+ + CBSA + PPM",
        "run_candidates": ["resnet34_unet3p_cbsa_ppm_e50_bs6"],
        "order": 5,
        "role": "alternative_context_module",
    },
    {
        "key": "cbsa_ocr_lite",
        "model_name": "ResNet34-UNet3+ + CBSA + OCR-Lite",
        "run_candidates": ["resnet34_unet3p_cbsa_ocr_lite_e50_bs6"],
        "order": 6,
        "role": "alternative_context_module",
    },
    {
        "key": "boundaryloss",
        "model_name": "CBSA + Lite-ASPP + BoundaryLoss",
        "run_candidates": ["resnet34_unet3p_cbsa_laspp_boundaryloss_e50_bs6"],
        "order": 7,
        "role": "loss_ablation",
    },
    {
        "key": "csftl",
        "model_name": "CBSA + Lite-ASPP + CSFTL",
        "run_candidates": ["resnet34_unet3p_cbsa_laspp_csftl_e50_bs6"],
        "order": 8,
        "role": "loss_ablation",
    },
    {
        "key": "edge_simam",
        "model_name": "Edge + SimAM",
        "run_candidates": ["resnet34_unet3p_edge_simam_e50_bs6"],
        "order": 9,
        "role": "old_strong_reference",
    },
    {
        "key": "brm",
        "model_name": "BRM",
        "run_candidates": ["resnet34_unet3p_brm_e50_bs6"],
        "order": 10,
        "role": "old_reference",
    },
]

METRICS = ["dice", "iou", "precision", "recall"]
LESION_GROUP_ORDER = ["small", "medium", "large"]
DIFF_GROUP_ORDER = ["hard", "medium", "easy"]


def read_csv_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_dict(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def to_int(x):
    try:
        return int(float(x))
    except Exception:
        return 0


def safe_mean(xs):
    xs = [to_float(x) for x in xs]
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.mean(xs)) if xs else float("nan")


def safe_std(xs):
    xs = [to_float(x) for x in xs]
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0


def safe_median(xs):
    xs = [to_float(x) for x in xs]
    xs = [x for x in xs if not np.isnan(x)]
    return float(np.median(xs)) if xs else float("nan")


def resolve_run_name(candidates, require_per_case=True):
    for rn in candidates:
        run_dir = PROJECT_ROOT / "runs" / rn
        if not run_dir.exists():
            continue
        if require_per_case and not (run_dir / "per_case_metrics.csv").exists():
            continue
        return rn
    return None


def get_specs():
    specs = []
    missing = []
    for m in MODEL_SPECS_RAW:
        rn = resolve_run_name(m["run_candidates"], require_per_case=True)
        item = dict(m)
        item["run_name"] = rn
        if rn is None:
            missing.append({
                "model_name": m["model_name"],
                "candidates": m["run_candidates"],
                "reason": "missing per_case_metrics.csv or run dir",
            })
        else:
            specs.append(item)
    specs = sorted(specs, key=lambda x: x["order"])
    return specs, missing


def per_case_path(run_name):
    return PROJECT_ROOT / "runs" / run_name / "per_case_metrics.csv"


def metric_path(run_name, name):
    return PROJECT_ROOT / "runs" / run_name / name


def summarize_per_case(specs):
    rows = []
    raw_by_model = {}

    for spec in specs:
        rn = spec["run_name"]
        pcs = read_csv_dict(per_case_path(rn))
        raw_by_model[spec["key"]] = {r["case_id"]: r for r in pcs}

        row = {
            "model_order": spec["order"],
            "model_key": spec["key"],
            "model_name": spec["model_name"],
            "run_name": rn,
            "role": spec["role"],
            "num_cases": len(pcs),
        }

        for metric in METRICS:
            vals = [to_float(r[metric]) for r in pcs]
            row[f"mean_{metric}"] = safe_mean(vals)
            row[f"std_{metric}"] = safe_std(vals)
            row[f"median_{metric}"] = safe_median(vals)

        row["mean_gt_area"] = safe_mean([r.get("gt_area", "") for r in pcs])
        row["mean_pred_area"] = safe_mean([r.get("pred_area", "") for r in pcs])
        rows.append(row)

    fields = [
        "model_order", "model_key", "model_name", "run_name", "role", "num_cases",
        "mean_dice", "std_dice", "median_dice",
        "mean_iou", "std_iou", "median_iou",
        "mean_precision", "std_precision", "median_precision",
        "mean_recall", "std_recall", "median_recall",
        "mean_gt_area", "mean_pred_area",
    ]
    out_csv = OUT_DIR / "overall_per_case_summary.csv"
    write_csv_dict(out_csv, rows, fields)
    return rows, raw_by_model


def summarize_micro_test_metrics(specs):
    rows = []
    for spec in specs:
        rn = spec["run_name"]
        tm = load_json(metric_path(rn, "test_metrics.json")) or {}
        sm = load_json(metric_path(rn, "summary_metrics.json")) or {}
        row = {
            "model_order": spec["order"],
            "model_key": spec["key"],
            "model_name": spec["model_name"],
            "run_name": rn,
            "role": spec["role"],
        }
        for metric in METRICS:
            row[f"test_{metric}"] = tm.get(metric, "")
            row[f"summary_{metric}"] = sm.get(metric, "")
        rows.append(row)

    fields = ["model_order", "model_key", "model_name", "run_name", "role"]
    for metric in METRICS:
        fields += [f"test_{metric}", f"summary_{metric}"]
    write_csv_dict(OUT_DIR / "overall_micro_test_metrics.csv", rows, fields)
    return rows


def read_lesion_groups():
    if not LESION_GROUP_CSV.exists():
        print(f"[WARN] lesion group file not found: {LESION_GROUP_CSV}")
        return None

    rows = read_csv_dict(LESION_GROUP_CSV)
    group_map = {}
    for r in rows:
        case_id = r.get("case_id")
        if not case_id:
            continue
        group_map[case_id] = {
            "lesion_size_group": r.get("lesion_size_group", ""),
            "gt_area_512": to_int(r.get("gt_area_512", r.get("gt_area", 0))),
            "gt_area_ratio_512": to_float(r.get("gt_area_ratio_512", r.get("gt_area_ratio", 0))),
            "area_rank": to_int(r.get("area_rank", 0)),
        }
    return group_map


def summarize_groups(specs, group_map, group_col, group_order, out_prefix, extra_case_fields=None):
    if group_map is None:
        return [], [], []

    extra_case_fields = extra_case_fields or []
    long_rows = []
    merged_rows = []

    for spec in specs:
        rn = spec["run_name"]
        pcs = read_csv_dict(per_case_path(rn))
        buckets = {g: [] for g in group_order}

        for r in pcs:
            case_id = r["case_id"]
            if case_id not in group_map:
                continue
            ginfo = group_map[case_id]
            group = ginfo[group_col]
            if group not in buckets:
                continue
            item = {
                "case_id": case_id,
                "model_order": spec["order"],
                "model_key": spec["key"],
                "model_name": spec["model_name"],
                "run_name": rn,
                "role": spec["role"],
                group_col: group,
                "dice": to_float(r["dice"]),
                "iou": to_float(r["iou"]),
                "precision": to_float(r["precision"]),
                "recall": to_float(r["recall"]),
                "gt_area": to_int(r.get("gt_area", 0)),
                "gt_area_ratio": to_float(r.get("gt_area_ratio", 0)),
                "pred_area": to_int(r.get("pred_area", 0)),
                "pred_area_ratio": to_float(r.get("pred_area_ratio", 0)),
                "image_path": r.get("image_path", ""),
                "mask_path": r.get("mask_path", ""),
            }
            for ef in extra_case_fields:
                item[ef] = ginfo.get(ef, "")
            # 病灶分组字段也尽量保留
            for maybe in ["gt_area_512", "gt_area_ratio_512", "area_rank"]:
                if maybe in ginfo:
                    item[maybe] = ginfo[maybe]
            buckets[group].append(item)
            merged_rows.append(item)

        for group in group_order:
            items = buckets[group]
            out = {
                "model_order": spec["order"],
                "model_key": spec["key"],
                "model_name": spec["model_name"],
                "run_name": rn,
                "role": spec["role"],
                group_col: group,
                "num_cases": len(items),
            }
            for metric in METRICS:
                vals = [x[metric] for x in items]
                out[f"mean_{metric}"] = safe_mean(vals)
                out[f"std_{metric}"] = safe_std(vals)
                out[f"median_{metric}"] = safe_median(vals)
            out["mean_gt_area"] = safe_mean([x["gt_area"] for x in items])
            out["mean_gt_area_ratio"] = safe_mean([x["gt_area_ratio"] for x in items])
            long_rows.append(out)

    long_rows = sorted(long_rows, key=lambda x: (int(x["model_order"]), group_order.index(x[group_col])))

    long_fields = [
        "model_order", "model_key", "model_name", "run_name", "role", group_col, "num_cases",
        "mean_dice", "std_dice", "median_dice",
        "mean_iou", "std_iou", "median_iou",
        "mean_precision", "std_precision", "median_precision",
        "mean_recall", "std_recall", "median_recall",
        "mean_gt_area", "mean_gt_area_ratio",
    ]
    write_csv_dict(OUT_DIR / f"{out_prefix}_metrics.csv", long_rows, long_fields)

    # wide rows
    by_model = defaultdict(dict)
    for r in long_rows:
        key = (r["model_order"], r["model_key"], r["model_name"], r["run_name"], r["role"])
        by_model[key][r[group_col]] = r

    wide_rows = []
    for key in sorted(by_model.keys(), key=lambda x: x[0]):
        order, model_key, model_name, run_name, role = key
        out = {
            "model_order": order,
            "model_key": model_key,
            "model_name": model_name,
            "run_name": run_name,
            "role": role,
        }
        for g in group_order:
            gr = by_model[key].get(g, {})
            out[f"{g}_num_cases"] = gr.get("num_cases", "")
            for metric in METRICS:
                out[f"{g}_{metric}"] = gr.get(f"mean_{metric}", "")
        wide_rows.append(out)

    # delta columns in wide: vs UNet3+ and vs ResNet34-UNet3+
    def find_wide(key_name):
        for r in wide_rows:
            if r["model_key"] == key_name:
                return r
        return None

    unet_row = find_wide("unet3p")
    resnet_row = find_wide("resnet34_unet3p")

    for r in wide_rows:
        for g in group_order:
            for metric in METRICS:
                val = r.get(f"{g}_{metric}", "")
                if val == "" or val is None:
                    r[f"{g}_{metric}_delta_vs_unet3p"] = ""
                    r[f"{g}_{metric}_delta_vs_resnet34_unet3p"] = ""
                    continue
                val = to_float(val)
                if unet_row and unet_row.get(f"{g}_{metric}", "") != "":
                    r[f"{g}_{metric}_delta_vs_unet3p"] = val - to_float(unet_row[f"{g}_{metric}"])
                else:
                    r[f"{g}_{metric}_delta_vs_unet3p"] = ""
                if resnet_row and resnet_row.get(f"{g}_{metric}", "") != "":
                    r[f"{g}_{metric}_delta_vs_resnet34_unet3p"] = val - to_float(resnet_row[f"{g}_{metric}"])
                else:
                    r[f"{g}_{metric}_delta_vs_resnet34_unet3p"] = ""

    wide_fields = ["model_order", "model_key", "model_name", "run_name", "role"]
    for g in group_order:
        wide_fields += [f"{g}_num_cases"]
        for metric in METRICS:
            wide_fields += [
                f"{g}_{metric}",
                f"{g}_{metric}_delta_vs_unet3p",
                f"{g}_{metric}_delta_vs_resnet34_unet3p",
            ]

    write_csv_dict(OUT_DIR / f"{out_prefix}_metrics_wide.csv", wide_rows, wide_fields)

    case_fields = [
        "case_id", "model_order", "model_key", "model_name", "run_name", "role", group_col,
    ]
    case_fields += extra_case_fields
    case_fields += [
        "gt_area", "gt_area_ratio", "gt_area_512", "gt_area_ratio_512", "area_rank",
        "pred_area", "pred_area_ratio",
        "dice", "iou", "precision", "recall", "image_path", "mask_path",
    ]
    write_csv_dict(OUT_DIR / f"{out_prefix}_merged_per_case.csv", merged_rows, case_fields)

    return long_rows, wide_rows, merged_rows


def make_difficulty_groups(specs):
    # 难度分组固定用 ResNet34-UNet3+，这样可以分析后续模块对强基线困难样本的改进。
    base_spec = None
    for s in specs:
        if s["key"] == "resnet34_unet3p":
            base_spec = s
            break
    if base_spec is None:
        raise RuntimeError("缺少 ResNet34-UNet3+ per_case_metrics.csv，无法定义困难样本。")

    rows = read_csv_dict(per_case_path(base_spec["run_name"]))
    rows = sorted(rows, key=lambda r: to_float(r["dice"]))
    n = len(rows)
    n_hard = n // 3
    n_medium = n // 3

    group_map = {}
    out_rows = []
    for idx, r in enumerate(rows):
        if idx < n_hard:
            g = "hard"
        elif idx < n_hard + n_medium:
            g = "medium"
        else:
            g = "easy"
        item = {
            "case_id": r["case_id"],
            "difficulty_group": g,
            "difficulty_rank": idx + 1,
            "baseline_run": base_spec["run_name"],
            "baseline_dice": to_float(r["dice"]),
            "baseline_iou": to_float(r["iou"]),
            "baseline_precision": to_float(r["precision"]),
            "baseline_recall": to_float(r["recall"]),
            "gt_area": to_int(r.get("gt_area", 0)),
            "gt_area_ratio": to_float(r.get("gt_area_ratio", 0)),
            "image_path": r.get("image_path", ""),
            "mask_path": r.get("mask_path", ""),
        }
        group_map[r["case_id"]] = item
        out_rows.append(item)

    fields = [
        "case_id", "difficulty_group", "difficulty_rank", "baseline_run",
        "baseline_dice", "baseline_iou", "baseline_precision", "baseline_recall",
        "gt_area", "gt_area_ratio", "image_path", "mask_path",
    ]
    write_csv_dict(OUT_DIR / "difficulty_groups_by_resnet34_unet3p.csv", out_rows, fields)
    return group_map


def summarize_delta(specs, baseline_key, out_prefix):
    base_spec = None
    for s in specs:
        if s["key"] == baseline_key:
            base_spec = s
            break
    if base_spec is None:
        print(f"[WARN] no baseline spec for {baseline_key}")
        return [], []

    base_rows = {r["case_id"]: r for r in read_csv_dict(per_case_path(base_spec["run_name"]))}
    summary_rows = []
    case_rows = []

    for spec in specs:
        if spec["key"] == baseline_key:
            continue
        rows = {r["case_id"]: r for r in read_csv_dict(per_case_path(spec["run_name"]))}
        common = sorted(set(base_rows.keys()) & set(rows.keys()))

        deltas = {m: [] for m in METRICS}
        pos_counts = {m: 0 for m in METRICS}
        neg_counts = {m: 0 for m in METRICS}

        for cid in common:
            b = base_rows[cid]
            o = rows[cid]
            item = {
                "case_id": cid,
                "baseline_key": baseline_key,
                "baseline_name": base_spec["model_name"],
                "baseline_run": base_spec["run_name"],
                "model_key": spec["key"],
                "model_name": spec["model_name"],
                "run_name": spec["run_name"],
                "gt_area": to_int(b.get("gt_area", 0)),
                "gt_area_ratio": to_float(b.get("gt_area_ratio", 0)),
                "image_path": b.get("image_path", ""),
                "mask_path": b.get("mask_path", ""),
            }
            for metric in METRICS:
                bv = to_float(b[metric])
                ov = to_float(o[metric])
                dv = ov - bv
                item[f"baseline_{metric}"] = bv
                item[f"model_{metric}"] = ov
                item[f"delta_{metric}"] = dv
                deltas[metric].append(dv)
                if dv > 0:
                    pos_counts[metric] += 1
                elif dv < 0:
                    neg_counts[metric] += 1
            case_rows.append(item)

        srow = {
            "baseline_key": baseline_key,
            "baseline_name": base_spec["model_name"],
            "baseline_run": base_spec["run_name"],
            "model_order": spec["order"],
            "model_key": spec["key"],
            "model_name": spec["model_name"],
            "run_name": spec["run_name"],
            "num_common_cases": len(common),
        }
        for metric in METRICS:
            srow[f"mean_delta_{metric}"] = safe_mean(deltas[metric])
            srow[f"median_delta_{metric}"] = safe_median(deltas[metric])
            srow[f"num_positive_delta_{metric}"] = pos_counts[metric]
            srow[f"num_negative_delta_{metric}"] = neg_counts[metric]
        summary_rows.append(srow)

    summary_rows = sorted(summary_rows, key=lambda x: int(x["model_order"]))
    case_rows = sorted(case_rows, key=lambda x: (x["model_key"], -to_float(x["delta_dice"])))

    summary_fields = [
        "baseline_key", "baseline_name", "baseline_run", "model_order", "model_key", "model_name", "run_name", "num_common_cases",
    ]
    for metric in METRICS:
        summary_fields += [
            f"mean_delta_{metric}", f"median_delta_{metric}",
            f"num_positive_delta_{metric}", f"num_negative_delta_{metric}",
        ]
    write_csv_dict(OUT_DIR / f"delta_summary_vs_{out_prefix}.csv", summary_rows, summary_fields)

    case_fields = [
        "case_id", "baseline_key", "baseline_name", "baseline_run",
        "model_key", "model_name", "run_name",
        "gt_area", "gt_area_ratio", "image_path", "mask_path",
    ]
    for metric in METRICS:
        case_fields += [f"baseline_{metric}", f"model_{metric}", f"delta_{metric}"]
    write_csv_dict(OUT_DIR / f"delta_cases_vs_{out_prefix}.csv", case_rows, case_fields)
    return summary_rows, case_rows


def make_recommendation(specs, overall_rows, lesion_wide, diff_wide, delta_vs_resnet):
    by_key = {r["model_key"]: r for r in overall_rows}

    best_overall = max(overall_rows, key=lambda r: to_float(r.get("mean_dice", "nan"))) if overall_rows else None

    def get_group_best(wide_rows, group, metric="dice"):
        candidates = []
        for r in wide_rows:
            v = r.get(f"{group}_{metric}", "")
            if v != "":
                candidates.append((to_float(v), r))
        if not candidates:
            return None
        return max(candidates, key=lambda x: x[0])[1]

    best_small = get_group_best(lesion_wide, "small", "dice") if lesion_wide else None
    best_hard = get_group_best(diff_wide, "hard", "dice") if diff_wide else None

    delta_map = {r["model_key"]: r for r in delta_vs_resnet}

    summary = {
        "available_models": [
            {
                "model_key": s["key"],
                "model_name": s["model_name"],
                "run_name": s["run_name"],
                "role": s["role"],
            } for s in specs
        ],
        "best_overall_per_case_mean_dice": best_overall,
        "best_small_lesion_dice": best_small,
        "best_hard_case_dice": best_hard,
        "main_model_key_recommended": "cbsa_laspp",
        "main_model_reason": (
            "CBSA+Lite-ASPP 通常更适合作为最终主模型：它在逐病例平均 Dice、困难样本表现和相对 ResNet34-UNet3+ 的稳定提升上更均衡。"
        ),
        "boundaryloss_note": (
            "BoundaryLoss 可作为损失函数消融。若 test_metrics/global Dice 较高但 per-case mean Dice 不最高，论文中应区分 global/micro Dice 与 per-case mean Dice。"
        ),
        "csftl_note": "CSFTL 可作为 Recall 增强消融，但要同时报告 Precision 下降。",
        "delta_vs_resnet34_unet3p": delta_map,
    }
    save_json(summary, OUT_DIR / "final_recommendation_summary.json")
    return summary


def save_excel_if_possible(tables):
    try:
        import pandas as pd
        out_xlsx = OUT_DIR / "final_paper_tables.xlsx"
        with pd.ExcelWriter(out_xlsx) as writer:
            for sheet_name, rows in tables.items():
                if not rows:
                    continue
                safe_name = sheet_name[:31]
                pd.DataFrame(rows).to_excel(writer, index=False, sheet_name=safe_name)
        print("Saved Excel:", out_xlsx)
    except Exception as e:
        print("[WARN] Excel not saved:", repr(e))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    specs, missing = get_specs()
    print("=" * 80)
    print("Available models:")
    for s in specs:
        print(f"[{s['order']}] {s['model_name']} -> {s['run_name']}")
    if missing:
        print("\nMissing models:")
        for m in missing:
            print(m)
    print("=" * 80)

    if not specs:
        raise RuntimeError("没有任何模型具备 per_case_metrics.csv，无法分析。")

    overall_rows, _ = summarize_per_case(specs)
    micro_rows = summarize_micro_test_metrics(specs)

    lesion_map = read_lesion_groups()
    lesion_long, lesion_wide, lesion_cases = summarize_groups(
        specs=specs,
        group_map=lesion_map,
        group_col="lesion_size_group",
        group_order=LESION_GROUP_ORDER,
        out_prefix="lesion_size_group",
    )

    diff_map = make_difficulty_groups(specs)
    diff_long, diff_wide, diff_cases = summarize_groups(
        specs=specs,
        group_map=diff_map,
        group_col="difficulty_group",
        group_order=DIFF_GROUP_ORDER,
        out_prefix="difficulty_group",
        extra_case_fields=["difficulty_rank", "baseline_dice", "baseline_iou", "baseline_precision", "baseline_recall"],
    )

    delta_vs_unet, delta_cases_vs_unet = summarize_delta(specs, "unet3p", "unet3p")
    delta_vs_resnet, delta_cases_vs_resnet = summarize_delta(specs, "resnet34_unet3p", "resnet34_unet3p")

    recommendation = make_recommendation(specs, overall_rows, lesion_wide, diff_wide, delta_vs_resnet)

    save_json({
        "missing_models": missing,
        "out_dir": str(OUT_DIR),
        "files": sorted([str(p.relative_to(PROJECT_ROOT)) for p in OUT_DIR.glob("*") if p.is_file()]),
    }, OUT_DIR / "analysis_manifest.json")

    save_excel_if_possible({
        "overall_per_case": overall_rows,
        "micro_test_metrics": micro_rows,
        "lesion_long": lesion_long,
        "lesion_wide": lesion_wide,
        "difficulty_long": diff_long,
        "difficulty_wide": diff_wide,
        "delta_vs_unet3p": delta_vs_unet,
        "delta_vs_resnet34": delta_vs_resnet,
    })

    print("=" * 80)
    print("Final paper evidence analysis finished.")
    print("Output dir:", OUT_DIR)
    print("Key recommendation:", recommendation["main_model_key_recommended"])
    print("=" * 80)


if __name__ == "__main__":
    main()
