from pathlib import Path
import csv
import argparse
import json

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_resnet34_unet3p_cbsa_laspp import (
    set_seed,
    save_json,
    CariesMaskDataset,
    ResNet34UNet3PlusCBSALiteASPP,
    get_main_logits,
    save_prediction_panel,
    parse_dilations,
)


def soft_dice_loss(logits, targets, smooth=1.0):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = torch.sum(probs * targets, dims)
    union = torch.sum(probs, dims) + torch.sum(targets, dims)
    dice = (2.0 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def dice_bce_loss(logits, targets, bce_weight=0.5, dice_weight=0.5):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss(logits, targets)
    return bce_weight * bce + dice_weight * dice


def caries_size_aware_focal_tversky_loss(
    logits,
    targets,
    alpha=0.3,
    beta=0.7,
    gamma=0.75,
    size_power=0.5,
    max_weight=3.0,
    smooth=1.0,
):
    """
    CS-FTL: Caries Size-aware Focal Tversky Loss.

    beta > alpha: penalize FN more, useful for small caries recall.
    size-aware weight: smaller foreground area gets larger sample weight.
    """
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)

    tp = torch.sum(probs * targets, dims)
    fp = torch.sum(probs * (1.0 - targets), dims)
    fn = torch.sum((1.0 - probs) * targets, dims)

    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    ftl = torch.pow(1.0 - tversky, gamma)

    # foreground ratio per image
    fg_ratio = torch.mean(targets, dim=dims)

    # avoid over-weighting empty masks; if no foreground, weight = 1
    positive = fg_ratio > 0
    if positive.any():
        mean_fg = fg_ratio[positive].mean().detach()
    else:
        mean_fg = torch.tensor(0.01, device=targets.device)

    size_weight = torch.ones_like(fg_ratio)
    size_weight[positive] = torch.pow(mean_fg / (fg_ratio[positive] + 1e-6), size_power)
    size_weight = torch.clamp(size_weight, 1.0, max_weight)

    return torch.mean(ftl * size_weight)


def make_signed_distance_map_np(mask_np):
    """
    mask_np: [B, 1, H, W], value 0/1
    return signed distance map:
      inside GT foreground: negative
      outside GT foreground: positive
    normalized to [-1, 1].
    """
    mask_np = (mask_np > 0.5).astype(np.uint8)
    sdf = np.zeros_like(mask_np, dtype=np.float32)

    b = mask_np.shape[0]
    for i in range(b):
        m = mask_np[i, 0]

        if m.max() == 0:
            # all background
            neg_dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
            s = neg_dist.astype(np.float32)
        elif m.min() == 1:
            # all foreground
            pos_dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            s = -pos_dist.astype(np.float32)
        else:
            pos_dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
            neg_dist = cv2.distanceTransform(1 - m, cv2.DIST_L2, 5)
            s = neg_dist.astype(np.float32) - pos_dist.astype(np.float32)

        max_abs = np.max(np.abs(s))
        if max_abs > 1e-6:
            s = s / max_abs

        sdf[i, 0] = s

    return sdf


def boundary_loss(logits, targets):
    """
    Distance-map based boundary loss.

    It encourages high probabilities inside GT region and low probabilities outside.
    Used together with Dice+BCE, with a small weight.
    """
    probs = torch.sigmoid(logits)
    targets_np = targets.detach().cpu().numpy()
    sdf_np = make_signed_distance_map_np(targets_np)
    sdf = torch.from_numpy(sdf_np).to(device=logits.device, dtype=logits.dtype)
    return torch.mean(probs * sdf)


def combined_loss(logits, targets, args):
    base = dice_bce_loss(
        logits,
        targets,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
    )

    if args.loss_mode == "csftl":
        aux = caries_size_aware_focal_tversky_loss(
            logits,
            targets,
            alpha=args.ftl_alpha,
            beta=args.ftl_beta,
            gamma=args.ftl_gamma,
            size_power=args.size_power,
            max_weight=args.max_size_weight,
        )
        return base + args.ftl_weight * aux

    if args.loss_mode == "boundary":
        aux = boundary_loss(logits, targets)
        return base + args.boundary_weight * aux

    raise ValueError(f"Unknown loss_mode: {args.loss_mode}")


def deep_supervision_loss(outputs, targets, args):
    if isinstance(outputs, (list, tuple)):
        weights = args.ds_weights
        if len(weights) < len(outputs):
            weights = weights + [weights[-1]] * (len(outputs) - len(weights))
        weights = weights[:len(outputs)]

        total = 0.0
        weight_sum = 0.0
        for out, w in zip(outputs, weights):
            total = total + float(w) * combined_loss(out, targets, args)
            weight_sum += float(w)
        return total / max(weight_sum, 1e-6)

    return combined_loss(outputs, targets, args)


def compute_batch_stats(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    targets = (targets > 0.5).float()

    tp = torch.sum(preds * targets).item()
    fp = torch.sum(preds * (1.0 - targets)).item()
    fn = torch.sum((1.0 - preds) * targets).item()

    return tp, fp, fn


def metrics_from_counts(tp, fp, fn, eps=1e-7):
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
    }


def run_one_epoch_custom(model, loader, optimizer, device, args, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_n = 0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0

    for batch in loader:
        images = batch[0].to(device, non_blocking=True)
        masks = batch[1].to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = deep_supervision_loss(outputs, masks, args)

            if train:
                loss.backward()
                optimizer.step()

        main_logits = get_main_logits(outputs)
        tp, fp, fn = compute_batch_stats(main_logits.detach(), masks.detach())

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_n += bs
        total_tp += tp
        total_fp += fp
        total_fn += fn

    m = metrics_from_counts(total_tp, total_fp, total_fn)
    m["loss"] = float(total_loss / max(total_n, 1))
    return m


def main(args):
    set_seed(args.seed)

    project_root = Path("/share/home/u2515283028/caries_project")
    train_split = project_root / "data" / "splits" / "train.txt"
    val_split = project_root / "data" / "splits" / "val.txt"
    test_split = project_root / "data" / "splits" / "test.txt"

    run_dir = project_root / "runs" / args.run_name
    ckpt_dir = run_dir / "checkpoints"
    pred_dir = run_dir / "preds"

    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    args.ds_weights = [float(x) for x in args.ds_weights.split(",") if x.strip()]

    config = vars(args).copy()
    save_json(config, run_dir / "config.json")

    train_ds = CariesMaskDataset(train_split, target_size=(args.image_size, args.image_size))
    val_ds = CariesMaskDataset(val_split, target_size=(args.image_size, args.image_size))
    test_ds = CariesMaskDataset(test_split, target_size=(args.image_size, args.image_size))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print("=" * 80)
    print("Dataset ready")
    print("train:", len(train_ds))
    print("val  :", len(val_ds))
    print("test :", len(test_ds))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    deep_supervision = bool(args.deep_supervision)
    aspp_dilations = parse_dilations(args.aspp_dilations)

    model = ResNet34UNet3PlusCBSALiteASPP(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cat_ch=args.cat_channels,
        deep_supervision=deep_supervision,
        cbsa_position=args.cbsa_position,
        cbsa_lambda=args.cbsa_lambda,
        cbsa_alpha=args.cbsa_alpha,
        cbsa_beta=args.cbsa_beta,
        aspp_hidden=args.aspp_hidden,
        aspp_dilations=aspp_dilations,
        aspp_dropout=args.aspp_dropout,
        aspp_res_scale=args.aspp_res_scale,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    sample_batch = next(iter(train_loader))
    sample_images = sample_batch[0].to(device)

    with torch.no_grad():
        outputs = model(sample_images)
        main_logits = get_main_logits(outputs)

    print("=" * 80)
    print("First forward check")
    print("input images.shape =", sample_images.shape)
    print("main_logits.shape  =", main_logits.shape)
    if isinstance(outputs, list):
        print("deep outputs       =", [tuple(o.shape) for o in outputs])
    print("loss_mode          =", args.loss_mode)
    print("bce_weight         =", args.bce_weight)
    print("dice_weight        =", args.dice_weight)
    print("ds_weights         =", args.ds_weights)
    print("cbsa_position      =", args.cbsa_position)
    print("aspp_hidden        =", args.aspp_hidden)
    print("aspp_dilations     =", aspp_dilations)
    if args.loss_mode == "csftl":
        print("ftl_weight         =", args.ftl_weight)
        print("ftl_alpha          =", args.ftl_alpha)
        print("ftl_beta           =", args.ftl_beta)
        print("ftl_gamma          =", args.ftl_gamma)
        print("size_power         =", args.size_power)
        print("max_size_weight    =", args.max_size_weight)
    if args.loss_mode == "boundary":
        print("boundary_weight    =", args.boundary_weight)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_custom(
            model,
            train_loader,
            optimizer,
            device,
            args,
            train=True,
        )

        val_metrics = run_one_epoch_custom(
            model,
            val_loader,
            optimizer=None,
            device=device,
            args=args,
            train=False,
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"], train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"], val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    best_path = ckpt_dir / "best.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    best_train_metrics = run_one_epoch_custom(
        model,
        train_loader,
        optimizer=None,
        device=device,
        args=args,
        train=False,
    )
    best_val_metrics = run_one_epoch_custom(
        model,
        val_loader,
        optimizer=None,
        device=device,
        args=args,
        train=False,
    )
    test_metrics = run_one_epoch_custom(
        model,
        test_loader,
        optimizer=None,
        device=device,
        args=args,
        train=False,
    )

    print("=" * 80)
    print("Best checkpoint summary")
    print("best_epoch:", best_epoch)
    print("best_val_dice_during_training:", best_val_dice)
    print("best_train_metrics:", best_train_metrics)
    print("best_val_metrics  :", best_val_metrics)
    print("test_metrics      :", test_metrics)
    print("=" * 80)

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }
    save_json(summary, run_dir / "summary_metrics.json")

    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--loss-mode", type=str, required=True, choices=["csftl", "boundary"])

    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--cat-channels", type=int, default=64)
    parser.add_argument("--deep-supervision", type=int, default=1)
    parser.add_argument("--ds-weights", type=str, default="1.0,0.4,0.3,0.2,0.1")

    parser.add_argument("--cbsa-position", type=str, default="all", choices=["all", "hd1", "hd2", "hd3", "hd4", "none"])
    parser.add_argument("--cbsa-lambda", type=float, default=1e-4)
    parser.add_argument("--cbsa-alpha", type=float, default=0.2)
    parser.add_argument("--cbsa-beta", type=float, default=0.1)

    parser.add_argument("--aspp-hidden", type=int, default=128)
    parser.add_argument("--aspp-dilations", type=str, default="1,3,5")
    parser.add_argument("--aspp-dropout", type=float, default=0.0)
    parser.add_argument("--aspp-res-scale", type=float, default=0.1)

    parser.add_argument("--bce-weight", type=float, default=0.5)
    parser.add_argument("--dice-weight", type=float, default=0.5)

    parser.add_argument("--ftl-weight", type=float, default=0.5)
    parser.add_argument("--ftl-alpha", type=float, default=0.3)
    parser.add_argument("--ftl-beta", type=float, default=0.7)
    parser.add_argument("--ftl-gamma", type=float, default=0.75)
    parser.add_argument("--size-power", type=float, default=0.5)
    parser.add_argument("--max-size-weight", type=float, default=3.0)

    parser.add_argument("--boundary-weight", type=float, default=0.05)

    args = parser.parse_args()
    main(args)
