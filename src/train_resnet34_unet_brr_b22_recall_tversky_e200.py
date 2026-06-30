import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    save_prediction_panel,
)
from train_resnet34_unet_auglite_e200_constlr import AugmentedCariesMaskDataset
from train_resnet34_unet_brr_b1_aux_e200 import (
    bce_dice_loss,
    dice_loss_from_logits,
    mask_to_boundary_and_rim,
)
from train_resnet34_unet_brr_b2_rbsg_e200 import ResNet34UNetBRRB2RBSG


class ResNet34UNetBRRB22RecallTversky(ResNet34UNetBRRB2RBSG):
    """
    B22: B2 architecture unchanged, recall-aware main segmentation loss.

    Several post-B2 experiments increased precision or regularity but lost
    recall. B22 therefore avoids adding another gate/attention branch and only
    makes the main mask loss slightly asymmetric against false negatives.
    """


def tversky_loss_from_logits(logits, targets, alpha=0.40, beta=0.60, gamma=1.0, eps=1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    tp = (probs * targets).sum(dims)
    fp = (probs * (1.0 - targets)).sum(dims)
    fn = ((1.0 - probs) * targets).sum(dims)

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss = 1.0 - tversky
    if abs(float(gamma) - 1.0) > 1e-6:
        loss = loss.pow(float(gamma))
    return loss.mean()


def recall_tversky_seg_loss(
    logits,
    targets,
    alpha,
    beta,
    gamma,
    dice_weight,
    tversky_weight,
):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    tversky = tversky_loss_from_logits(
        logits,
        targets,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
    )
    return bce + float(dice_weight) * dice + float(tversky_weight) * tversky, {
        "loss_bce": float(bce.detach().cpu()),
        "loss_dice": float(dice.detach().cpu()),
        "loss_tversky": float(tversky.detach().cpu()),
        "tversky_alpha": float(alpha),
        "tversky_beta": float(beta),
    }


def compute_b22_loss(outputs, masks, epoch, args):
    mask_logits = outputs["mask"]
    boundary_logits = outputs["boundary"]
    rim_logits = outputs["rim"]

    if epoch <= args.tversky_warmup_epochs:
        alpha = args.tversky_alpha_warmup
        beta = args.tversky_beta_warmup
    else:
        alpha = args.tversky_alpha
        beta = args.tversky_beta

    loss_seg, seg_details = recall_tversky_seg_loss(
        mask_logits,
        masks,
        alpha=alpha,
        beta=beta,
        gamma=args.tversky_gamma,
        dice_weight=args.dice_weight,
        tversky_weight=args.tversky_weight,
    )

    boundary_gt, rim_gt = mask_to_boundary_and_rim(masks, rb=3, rr=9)
    loss_boundary = bce_dice_loss(boundary_logits, boundary_gt)
    loss_rim = F.binary_cross_entropy_with_logits(rim_logits, rim_gt)

    if epoch <= 20:
        wb = 0.10
        wr = 0.05
    else:
        wb = 0.20
        wr = 0.10

    total = loss_seg + wb * loss_boundary + wr * loss_rim
    seg_details.update(
        {
            "loss_seg": float(loss_seg.detach().cpu()),
            "loss_boundary": float(loss_boundary.detach().cpu()),
            "loss_rim": float(loss_rim.detach().cpu()),
            "wb": wb,
            "wr": wr,
        }
    )
    return total, seg_details


def run_one_epoch_b22(model, loader, optimizer, device, train=True, epoch=1, args=None):
    model.train(train)

    total_loss = 0.0
    total_loss_tversky = 0.0
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
            outputs = model(images, return_aux=True)
            loss, details = compute_b22_loss(outputs, masks, epoch=epoch, args=args)

            if train:
                loss.backward()
                optimizer.step()

        logits = outputs["mask"].detach()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        gt = (masks > 0.5).float()

        tp = (preds * gt).sum()
        fp = (preds * (1.0 - gt)).sum()
        fn = ((1.0 - preds) * gt).sum()
        tn = ((1.0 - preds) * (1.0 - gt)).sum()

        total_loss += float(loss.detach().cpu())
        total_loss_tversky += float(details["loss_tversky"])
        total_tp += float(tp.detach().cpu())
        total_fp += float(fp.detach().cpu())
        total_fn += float(fn.detach().cpu())
        total_tn += float(tn.detach().cpu())
        n_batches += 1

    eps = 1e-6
    dice = (2.0 * total_tp + eps) / (2.0 * total_tp + total_fp + total_fn + eps)
    iou = (total_tp + eps) / (total_tp + total_fp + total_fn + eps)
    precision = (total_tp + eps) / (total_tp + total_fp + eps)
    recall = (total_tp + eps) / (total_tp + total_fn + eps)

    return {
        "loss": total_loss / max(n_batches, 1),
        "loss_tversky": total_loss_tversky / max(n_batches, 1),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
    }


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
    config["model_class"] = "ResNet34UNetBRRB22RecallTversky"
    config["experiment"] = "B22: B2 unchanged + mild recall-biased Tversky main mask loss"
    config["note"] = (
        "B22 keeps the B2 architecture, boundary/rim targets, auxiliary weights, "
        "and RBSG-lite unchanged. It only replaces the main mask Dice term with "
        "a Dice/Tversky mixture that penalizes false negatives slightly more than "
        "false positives. This directly targets the recall drop observed in B19, "
        "B20 and B21 without introducing another boundary/rim gate."
    )
    config["result_reflection"] = {
        "b18": "EMA nearly tied B2 but shifted toward higher precision and lower recall",
        "b19": "flip consistency reduced both test Dice and recall",
        "b20": "inner distance supervision was too weak and core-biased; recall dropped",
        "b21": "IFE scale stayed at 0.0, so the new branch was effectively inactive",
        "design_implication": "use a small, interpretable loss-side correction for false negatives instead of more spatial modules",
    }
    config["design_references"] = [
        {
            "title": "Focal Tversky Loss for Image Segmentation",
            "adopted_idea": "asymmetric Tversky weighting is useful for highly imbalanced medical foregrounds and small lesions",
        },
        {
            "title": "nnU-Net Revisited",
            "adopted_idea": "retain a strong U-Net recipe and validate conservative, isolated changes",
        },
    ]
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L_aux = 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_aux = 0.20*L_boundary + 0.10*L_rim",
    }
    config["recall_tversky_loss"] = {
        "main_loss": "BCE + dice_weight*Dice + tversky_weight*Tversky",
        "warmup": {
            "epochs": args.tversky_warmup_epochs,
            "alpha_fp": args.tversky_alpha_warmup,
            "beta_fn": args.tversky_beta_warmup,
        },
        "after_warmup": {
            "alpha_fp": args.tversky_alpha,
            "beta_fn": args.tversky_beta,
            "gamma": args.tversky_gamma,
            "dice_weight": args.dice_weight,
            "tversky_weight": args.tversky_weight,
        },
    }
    config["augmentation"] = {
        "type": "online",
        "profile": args.augment_profile,
        "train_only": True,
    }
    save_json(config, run_dir / "config.json")

    train_ds = AugmentedCariesMaskDataset(
        train_split,
        target_size=(args.image_size, args.image_size),
        profile=args.augment_profile,
    )
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34UNetBRRB22RecallTversky(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B22 Recall-Tversky + B2 BRR RBSG-lite ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_loss_tversky",
            "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_loss_tversky",
            "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b22(
            model, train_loader, optimizer, device, train=True, epoch=epoch, args=args
        )
        val_metrics = run_one_epoch_b22(
            model, val_loader, optimizer=None, device=device, train=False, epoch=epoch, args=args
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} val_tversky={val_metrics['loss_tversky']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["loss_tversky"],
                train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["loss_tversky"],
                val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))

    best_train_metrics = run_one_epoch_b22(
        model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs, args=args
    )
    best_val_metrics = run_one_epoch_b22(
        model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs, args=args
    )
    test_metrics = run_one_epoch_b22(
        model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs, args=args
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

    parser.add_argument("--run-name", type=str, default="b22_recall_tversky_auglite_e200_constlr_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--augment-profile", type=str, default="lite", choices=["lite", "strong", "none"])
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--tversky-warmup-epochs", type=int, default=20)
    parser.add_argument("--tversky-alpha-warmup", type=float, default=0.45)
    parser.add_argument("--tversky-beta-warmup", type=float, default=0.55)
    parser.add_argument("--tversky-alpha", type=float, default=0.40)
    parser.add_argument("--tversky-beta", type=float, default=0.60)
    parser.add_argument("--tversky-gamma", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=0.60)
    parser.add_argument("--tversky-weight", type=float, default=0.40)

    args = parser.parse_args()
    main(args)
