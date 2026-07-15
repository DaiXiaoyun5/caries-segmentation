#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Export official U-Mamba/nnU-Net artifacts into the project result schema."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def safe_value(values: List[Any], index: int, default: Any = None) -> Any:
    if index < 0 or index >= len(values):
        return default
    value = values[index]
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    return value


def read_per_case(csv_path: Path) -> List[Dict[str, Any]]:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No per-case rows found in {csv_path}")
    numeric = ("dice", "iou", "precision", "recall", "tp", "fp", "fn")
    for row in rows:
        for key in numeric:
            row[key] = float(row[key])
    return rows


def aggregate_test_metrics(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    tp = float(sum(row["tp"] for row in rows))
    fp = float(sum(row["fp"] for row in rows))
    fn = float(sum(row["fn"] for row in rows))
    total_pixels = 0
    for row in rows:
        with Image.open(row["gt_path"]) as image:
            total_pixels += int(image.width * image.height)
    tn = float(total_pixels) - tp - fp - fn
    eps = 1e-6
    fg_iou = (tp + eps) / (tp + fp + fn + eps)
    bg_iou = (tn + eps) / (tn + fp + fn + eps)
    return {
        "loss": None,
        "dice": (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps),
        "iou": fg_iou,
        "mean_iou": 0.5 * (fg_iou + bg_iou),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
        "accuracy": (tp + tn + eps) / (tp + tn + fp + fn + eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "num_cases": len(rows),
        "aggregation": "global pixel confusion over the fixed test split",
    }


def per_case_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"num_cases": len(rows)}
    for key in ("dice", "iou", "precision", "recall"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        result[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
        }
    return result


def main(args: argparse.Namespace) -> None:
    checkpoint_path = Path(args.checkpoint)
    prediction_dir = Path(args.pred_dir)
    validation_summary_path = Path(args.validation_summary)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    native_test_path = prediction_dir / "nnunet_test_metrics.json"
    native_case_path = prediction_dir / "nnunet_per_case_metrics.csv"
    for path in (checkpoint_path, native_test_path, native_case_path):
        if not path.exists():
            raise FileNotFoundError(path)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    logging = checkpoint.get("logging", {})
    epochs = min(
        (
            len(logging.get("train_losses", [])),
            len(logging.get("val_losses", [])),
            len(logging.get("mean_fg_dice", [])),
            len(logging.get("ema_fg_dice", [])),
            len(logging.get("lrs", [])),
        )
    )
    if epochs <= 0:
        raise RuntimeError("U-Mamba checkpoint does not contain epoch logging")

    ema_values = np.asarray(logging["ema_fg_dice"][:epochs], dtype=np.float64)
    best_index = int(np.nanargmax(ema_values))
    history_path = output_dir / "history.csv"
    history_fields = (
        "epoch",
        "train_loss",
        "val_loss",
        "mean_fg_dice",
        "ema_fg_dice",
        "lr",
        "epoch_seconds",
    )
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history_fields)
        writer.writeheader()
        for index in range(epochs):
            start = safe_value(logging.get("epoch_start_timestamps", []), index)
            end = safe_value(logging.get("epoch_end_timestamps", []), index)
            writer.writerow(
                {
                    "epoch": index + 1,
                    "train_loss": safe_value(logging["train_losses"], index),
                    "val_loss": safe_value(logging["val_losses"], index),
                    "mean_fg_dice": safe_value(logging["mean_fg_dice"], index),
                    "ema_fg_dice": safe_value(logging["ema_fg_dice"], index),
                    "lr": safe_value(logging["lrs"], index),
                    "epoch_seconds": None if start is None or end is None else end - start,
                }
            )

    native_test = json.loads(native_test_path.read_text(encoding="utf-8"))
    case_rows = read_per_case(native_case_path)
    test_metrics = aggregate_test_metrics(case_rows)
    test_metrics["per_case_mean"] = {
        key: native_test.get(key) for key in ("dice", "iou", "precision", "recall")
    }

    full_validation = None
    if validation_summary_path.exists():
        full_validation = json.loads(validation_summary_path.read_text(encoding="utf-8"))

    best_train_metrics = {
        "best_validation_epoch": best_index + 1,
        "train_loss": safe_value(logging["train_losses"], best_index),
        "metric_scope": "official online training loss at the best EMA validation epoch",
        "note": "The official nnU-Net trainer does not run a full train-set segmentation evaluation.",
    }
    best_val_metrics = {
        "best_epoch": best_index + 1,
        "loss": safe_value(logging["val_losses"], best_index),
        "dice": safe_value(logging["mean_fg_dice"], best_index),
        "ema_dice": safe_value(logging["ema_fg_dice"], best_index),
        "selection_metric": "EMA foreground pseudo Dice",
        "full_validation_summary": full_validation,
    }

    config = {
        "run_name": args.run_name,
        "script": "official U-Mamba nnUNetTrainerUMambaBot + project result exporter",
        "method": "U-Mamba-Bot 2D",
        "architecture": "official convolutional U-Net with a bottleneck Mamba layer",
        "dataset_id": 711,
        "dataset_name": "Dataset711_CariXray",
        "configuration": "2d",
        "fold": 0,
        "trainer": "nnUNetTrainerUMambaBot",
        "training_recipe": "official U-Mamba embedded nnU-Net v2 recipe",
        "num_epochs": int(checkpoint.get("current_epoch", epochs)),
        "checkpoint": str(checkpoint_path),
        "validation_summary": str(validation_summary_path),
        "prediction_dir": str(prediction_dir),
        "software": {
            "python_torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mamba_ssm": version("mamba-ssm"),
            "causal_conv1d": version("causal-conv1d"),
            "nnunetv2": version("nnunetv2"),
        },
        "reference": {
            "paper": "https://arxiv.org/abs/2401.04722",
            "official_code": "https://github.com/bowang-lab/U-Mamba",
        },
        "evaluation": {
            "primary_metrics": "global pixel confusion at source GT resolution",
            "secondary_metrics": "mean per-case metrics retained under per_case_mean",
            "test_cases": len(case_rows),
        },
    }

    save_json(config, output_dir / "config.json")
    save_json(best_train_metrics, output_dir / "best_train_metrics.json")
    save_json(best_val_metrics, output_dir / "best_val_metrics.json")
    save_json(test_metrics, output_dir / "test_metrics.json")
    save_json(per_case_summary(case_rows), output_dir / "per_case_metrics_summary.json")
    shutil.copyfile(native_case_path, output_dir / "per_case_metrics.csv")
    save_json(
        {
            "best_epoch": best_index + 1,
            "checkpoint_selection": "maximum EMA foreground pseudo Dice",
            "completed_epochs": epochs,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        },
        output_dir / "summary_metrics.json",
    )
    print(json.dumps(test_metrics, indent=2, ensure_ascii=False))
    print("Exported project-standard results to:", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="umamba_bot_official_2d_fold0")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-summary", required=True)
    parser.add_argument("--pred-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    main(parser.parse_args())
