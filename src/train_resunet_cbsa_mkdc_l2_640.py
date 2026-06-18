from pathlib import Path
import csv
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    save_prediction_panel,
)

from train_resunet_cbsa_mkdc_p2 import ResNet34UNetCBSAMKDC


def dice_loss_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    inter = (probs * targets).sum(dims)
    union = probs.sum(dims) + targets.sum(dims)
    dice = (2.0 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def focal_tversky_loss_from_logits(
    logits,
    targets,
    alpha=0.3,
    beta=0.7,
    gamma=0.75,
    eps=1e-7,
):
    """
    Focal Tversky Loss.

    alpha controls FP penalty.
    beta controls FN penalty.
    beta > alpha means more penalty for missed caries pixels, which is suitable
    for small-lesion caries segmentation where false negatives are important.
    """
    probs = torch.sigmoid(logits)

    dims = (1, 2, 3)
    tp = (probs * targets).sum(dims)
    fp = (probs * (1.0 - targets)).sum(dims)
    fn = ((1.0 - probs) * targets).sum(dims)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = torch.pow((1.0 - tversky), gamma)
    return loss.mean()


def bce_dice_focal_tversky_loss(
    logits,
    targets,
    bce_weight=0.3,
    dice_weight=0.3,
    ft_weight=0.4,
    ft_alpha=0.3,
    ft_beta=0.7,
    ft_gamma=0.75,
):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    ft = focal_tversky_loss_from_logits(
        logits,
        targets,
        alpha=ft_alpha,
        beta=ft_beta,
        gamma=ft_gamma,
    )
    total = bce_weight * bce + dice_weight * dice + ft_weight * ft
    return total, {
        "bce": float(bce.detach().cpu()),
        "dice_loss": float(dice.detach().cpu()),
        "focal_tversky": float(ft.detach().cpu()),
    }


@torch.no_grad()
def binary_metrics_from_logits(logits, targets, eps=1e-7):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = (targets > 0.5).float()

    dims = (1, 2, 3)
    tp = (preds * targets).sum(dims)
    fp = (preds * (1.0 - targets)).sum(dims)
    fn = ((1.0 - preds) * targets).sum(dims)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice.mean().detach().cpu()),
        "iou": float(iou.mean().detach().cpu()),
        "precision": float(precision.mean().detach().cpu()),
        "recall": float(recall.mean().detach().cpu()),
    }


def run_one_epoch_l2(
    model,
    loader,
    optimizer,
    device,
    train,
    bce_weight,
    dice_weight,
    ft_weight,
    ft_alpha,
    ft_beta,
    ft_gamma,
):
    if train:
        model.train()
    else:
        model.eval()

    totals = {
        "loss": 0.0,
        "bce": 0.0,
        "dice_loss": 0.0,
        "focal_tversky": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "precision": 0.0,
        "recall": 0.0,
    }
    n_samples = 0

    for batch in loader:
        if isinstance(batch, dict):
            images = batch["image"]
            masks = batch["mask"]
        else:
            images, masks = batch[:2]

        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        if train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss, loss_parts = bce_dice_focal_tversky_loss(
                logits,
                masks,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                ft_weight=ft_weight,
                ft_alpha=ft_alpha,
                ft_beta=ft_beta,
                ft_gamma=ft_gamma,
            )
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss, loss_parts = bce_dice_focal_tversky_loss(
                    logits,
                    masks,
                    bce_weight=bce_weight,
                    dice_weight=dice_weight,
                    ft_weight=ft_weight,
                    ft_alpha=ft_alpha,
                    ft_beta=ft_beta,
                    ft_gamma=ft_gamma,
                )

        metrics = binary_metrics_from_logits(logits.detach(), masks)

        bs = images.shape[0]
        n_samples += bs
        totals["loss"] += float(loss.detach().cpu()) * bs
        totals["bce"] += loss_parts["bce"] * bs
        totals["dice_loss"] += loss_parts["dice_loss"] * bs
        totals["focal_tversky"] += loss_parts["focal_tversky"] * bs

        for k in ["dice", "iou", "precision", "recall"]:
            totals[k] += metrics[k] * bs

    return {k: v / max(n_samples, 1) for k, v in totals.items()}


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

    config = vars(args).copy()
    config["model_class"] = "ResNet34UNetCBSAMKDC"
    config["experiment"] = "L2: R1-640 + BCE-Dice-FocalTversky"
    config["loss"] = "0.3*BCE + 0.3*Dice + 0.4*FocalTversky(alpha=0.3,beta=0.7,gamma=0.75)"
    config["note"] = "Same structure as P2/R1: ResNet34-U-Net + CBSA + MKDC. Only training loss is changed."
    save_json(config, run_dir / "config.json")

    train_ds = CariesMaskDataset(train_split, target_size=(args.image_size, args.image_size))
    val_ds = CariesMaskDataset(val_split, target_size=(args.image_size, args.image_size))
    test_ds = CariesMaskDataset(test_split, target_size=(args.image_size, args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("=" * 80)
    print("Dataset ready")
    print("train:", len(train_ds))
    print("val  :", len(val_ds))
    print("test :", len(test_ds))
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    model = ResNet34UNetCBSAMKDC(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cbsa_lambda=args.cbsa_lambda,
        cbsa_alpha=args.cbsa_alpha,
        cbsa_beta=args.cbsa_beta,
        mk_expansion_factor=args.mk_expansion_factor,
        mk_kernel_sizes=tuple(int(v) for v in args.mk_kernel_sizes.split(",")),
        mk_depth=args.mk_depth,
        mk_add=args.mk_add,
        mk_activation=args.mk_activation,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: L2 ResNet34-U-Net + CBSA + MKDC, 640 input, BCE-Dice-FocalTversky")
    print("run_name             =", args.run_name)
    print("image_size           =", args.image_size)
    print("batch_size           =", args.batch_size)
    print("epochs               =", args.epochs)
    print("lr                   =", args.lr)
    print("bce_weight           =", args.bce_weight)
    print("dice_weight          =", args.dice_weight)
    print("ft_weight            =", args.ft_weight)
    print("ft_alpha             =", args.ft_alpha)
    print("ft_beta              =", args.ft_beta)
    print("ft_gamma             =", args.ft_gamma)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_bce", "train_dice_loss", "train_focal_tversky",
            "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_bce", "val_dice_loss", "val_focal_tversky",
            "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_l2(
            model, train_loader, optimizer, device, train=True,
            bce_weight=args.bce_weight, dice_weight=args.dice_weight, ft_weight=args.ft_weight,
            ft_alpha=args.ft_alpha, ft_beta=args.ft_beta, ft_gamma=args.ft_gamma,
        )

        val_metrics = run_one_epoch_l2(
            model, val_loader, optimizer=None, device=device, train=False,
            bce_weight=args.bce_weight, dice_weight=args.dice_weight, ft_weight=args.ft_weight,
            ft_alpha=args.ft_alpha, ft_beta=args.ft_beta, ft_gamma=args.ft_gamma,
        )

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

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["bce"], train_metrics["dice_loss"], train_metrics["focal_tversky"],
                train_metrics["dice"], train_metrics["iou"], train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["bce"], val_metrics["dice_loss"], val_metrics["focal_tversky"],
                val_metrics["dice"], val_metrics["iou"], val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    best_path = ckpt_dir / "best.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    best_train_metrics = run_one_epoch_l2(
        model, train_loader, optimizer=None, device=device, train=False,
        bce_weight=args.bce_weight, dice_weight=args.dice_weight, ft_weight=args.ft_weight,
        ft_alpha=args.ft_alpha, ft_beta=args.ft_beta, ft_gamma=args.ft_gamma,
    )
    best_val_metrics = run_one_epoch_l2(
        model, val_loader, optimizer=None, device=device, train=False,
        bce_weight=args.bce_weight, dice_weight=args.dice_weight, ft_weight=args.ft_weight,
        ft_alpha=args.ft_alpha, ft_beta=args.ft_beta, ft_gamma=args.ft_gamma,
    )
    test_metrics = run_one_epoch_l2(
        model, test_loader, optimizer=None, device=device, train=False,
        bce_weight=args.bce_weight, dice_weight=args.dice_weight, ft_weight=args.ft_weight,
        ft_alpha=args.ft_alpha, ft_beta=args.ft_beta, ft_gamma=args.ft_gamma,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
    }

    print("=" * 80)
    print("Best checkpoint summary")
    print(summary)
    print("=" * 80)

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")

    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default="resunet_cbsa_mkdc_l2_640_e80_bs4")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--cbsa-lambda", type=float, default=1e-4)
    parser.add_argument("--cbsa-alpha", type=float, default=1.0)
    parser.add_argument("--cbsa-beta", type=float, default=0.1)

    parser.add_argument("--mk-expansion-factor", type=int, default=2)
    parser.add_argument("--mk-kernel-sizes", type=str, default="1,3,5")
    parser.add_argument("--mk-depth", type=int, default=1)
    parser.add_argument("--mk-add", action="store_true", default=True)
    parser.add_argument("--mk-activation", type=str, default="relu6")

    parser.add_argument("--bce-weight", type=float, default=0.3)
    parser.add_argument("--dice-weight", type=float, default=0.3)
    parser.add_argument("--ft-weight", type=float, default=0.4)
    parser.add_argument("--ft-alpha", type=float, default=0.3)
    parser.add_argument("--ft-beta", type=float, default=0.7)
    parser.add_argument("--ft-gamma", type=float, default=0.75)

    args = parser.parse_args()
    main(args)
