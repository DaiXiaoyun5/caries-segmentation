#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared, controlled training loop for external comparison baselines.

This module intentionally keeps the data split, AugLite augmentation, loss,
optimizer, checkpoint selection, and metric files aligned with the B30 E260
protocol. Each comparison model still has its own executable training script.
"""

from pathlib import Path
import csv
from typing import Callable, Dict, Any

import torch
from torch.utils.data import DataLoader

from train_resnet34_unet3p import (
    CariesMaskDataset,
    bce_dice_loss,
    save_json,
    save_prediction_panel,
    set_seed,
)
from train_resnet34_unet_auglite_e200_constlr import AugmentedCariesMaskDataset


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")


def run_one_epoch_comparison(model, loader, optimizer, device, train=True):
    """Use the same global TP/FP/FN/TN metric convention as B2/B30."""
    model.train(train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = bce_dice_loss(logits, masks)
            if train:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits.detach()) > 0.5).float()
            gt = (masks > 0.5).float()
            total_tp += float((preds * gt).sum().cpu())
            total_fp += float((preds * (1.0 - gt)).sum().cpu())
            total_fn += float(((1.0 - preds) * gt).sum().cpu())
            total_tn += float(((1.0 - preds) * (1.0 - gt)).sum().cpu())

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    eps = 1e-6
    dice = (2.0 * total_tp + eps) / (
        2.0 * total_tp + total_fp + total_fn + eps
    )
    iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    precision = (total_tp + eps) / (total_tp + total_fp + eps)
    recall = (total_tp + eps) / (total_tp + total_fn + eps)

    return {
        "loss": total_loss / max(n_batches, 1),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


def build_augmentation_config(profile: str) -> Dict[str, Any]:
    """Return a paper-friendly record of the existing AugLite policy."""
    if profile == "none":
        return {"type": "none", "train_only": True}

    if profile == "lite":
        return {
            "type": "online",
            "profile": "lite",
            "train_only": True,
            "geometry": {
                "rotation_deg": [-15.0, 15.0],
                "scale": [0.85, 1.20],
                "translate_ratio": 0.05,
                "horizontal_flip_p": 0.5,
                "vertical_flip_p": 0.0,
            },
            "photometric": {
                "brightness_factor": [0.85, 1.15],
                "contrast_factor": [0.85, 1.15],
                "gamma": [0.80, 1.25],
                "gaussian_noise_p": 0.10,
                "noise_std": [0.005, 0.020],
                "gaussian_blur_p": 0.10,
                "blur_radius": [0.2, 0.8],
            },
        }

    return {
        "type": "online",
        "profile": "strong",
        "train_only": True,
        "note": "Uses the existing strong profile in AugmentedCariesMaskDataset.",
    }


def run_comparison_experiment(
    args,
    model_builder: Callable[[Any], torch.nn.Module],
    model_metadata: Dict[str, Any],
) -> None:
    """Train and evaluate one external baseline with the controlled protocol."""
    set_seed(args.seed)

    train_split = PROJECT_ROOT / "data/splits/train.txt"
    val_split = PROJECT_ROOT / "data/splits/val.txt"
    test_split = PROJECT_ROOT / "data/splits/test.txt"

    run_dir = PROJECT_ROOT / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    pred_dir = run_dir / "preds"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update(model_metadata)
    config["training_protocol"] = {
        "split": "data/splits/{train,val,test}.txt",
        "image_size": args.image_size,
        "augment_profile": args.augment_profile,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "learning_rate_schedule": "constant",
        "weight_decay": args.weight_decay,
        "loss": "BCEWithLogits + soft Dice",
        "checkpoint_selection": "highest validation Dice",
        "threshold": 0.5,
        "metric_aggregation": "global TP/FP/FN/TN over the complete split",
        "seed": args.seed,
    }
    config["augmentation"] = build_augmentation_config(args.augment_profile)
    save_json(config, run_dir / "config.json")

    train_ds = AugmentedCariesMaskDataset(
        train_split,
        target_size=(args.image_size, args.image_size),
        profile=args.augment_profile,
    )
    val_ds = CariesMaskDataset(
        val_split,
        target_size=(args.image_size, args.image_size),
    )
    test_ds = CariesMaskDataset(
        test_split,
        target_size=(args.image_size, args.image_size),
    )

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)

    print("=" * 80)
    print("Run name:", args.run_name)
    print("Model:", model_metadata.get("model_name"))
    print("Train/val/test:", len(train_ds), len(val_ds), len(test_ds))
    print("Protocol: AugLite, E%d, constant lr=%g" % (args.epochs, args.lr))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)
    model = model_builder(args).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    sample_images, _, _, _ = next(iter(train_loader))
    sample_images = sample_images.to(device)
    with torch.no_grad():
        sample_logits = model(sample_images)
    print("input shape =", tuple(sample_images.shape))
    print("logits shape =", tuple(sample_logits.shape))

    if sample_logits.shape[-2:] != sample_images.shape[-2:]:
        raise RuntimeError(
            "Comparison model must return full-resolution logits, got "
            f"{tuple(sample_logits.shape)} for {tuple(sample_images.shape)}"
        )
    if sample_logits.shape[1] != 1:
        raise RuntimeError(
            f"Binary segmentation requires one output channel, got {sample_logits.shape[1]}"
        )

    history_path = run_dir / "history.csv"
    fieldnames = [
        "epoch",
        "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
        "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
    ]
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_comparison(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
        )
        val_metrics = run_one_epoch_comparison(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
        )

        row = {"epoch": epoch}
        metric_keys = ("loss", "dice", "iou", "precision", "recall")
        for key in metric_keys:
            value = train_metrics[key]
            row[f"train_{key}"] = value
        for key in metric_keys:
            value = val_metrics[key]
            row[f"val_{key}"] = value
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} "
            f"val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
        )

        torch.save(model.state_dict(), ckpt_dir / "last.pth")
        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(
                f"  -> best model updated: epoch={best_epoch}, "
                f"val_dice={best_val_dice:.4f}"
            )

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))
    best_train_metrics = run_one_epoch_comparison(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
    )
    best_val_metrics = run_one_epoch_comparison(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
    )
    test_metrics = run_one_epoch_comparison(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
    )

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(
        {
            "best_epoch": best_epoch,
            "best_val_dice_during_training": best_val_dice,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        },
        run_dir / "summary_metrics.json",
    )
    save_prediction_panel(
        model,
        test_loader,
        device,
        pred_dir / "test_preview.png",
        max_items=4,
    )

    print("=" * 80)
    print("Finished:", args.run_name)
    print("Best epoch:", best_epoch)
    print("Best validation Dice:", best_val_dice)
    print("Test metrics:", test_metrics)
    print("=" * 80)
