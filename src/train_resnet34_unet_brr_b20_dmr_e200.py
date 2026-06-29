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
from train_resnet34_unet_brr_b1_aux_e200 import compute_b1_loss
from train_resnet34_unet_brr_b2_rbsg_e200 import ResNet34UNetBRRB2RBSG


def mask_to_inner_distance_target(mask, max_radius=16):
    """
    Approximate an inner normalized distance map with repeated erosions.

    Dental caries masks are sparse and often low contrast. A binary mask only
    tells the network foreground/background, while an inner distance target
    softly distinguishes lesion core pixels from uncertain lesion margins.
    This is intentionally different from B2 boundary/rim supervision: it
    encourages lesion completeness instead of adding another boundary/rim gate.
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    max_radius = max(int(max_radius), 1)
    cur = (mask > 0.5).float()
    dist = torch.zeros_like(cur)

    for _ in range(max_radius):
        dist = dist + cur
        cur = -F.max_pool2d(-cur, kernel_size=3, stride=1, padding=1)
        cur = cur.clamp(0.0, 1.0)

    return (dist / float(max_radius)).clamp(0.0, 1.0)


class ResNet34UNetBRRB20DMR(ResNet34UNetBRRB2RBSG):
    """
    B20: B2 + lightweight Distance Map Regression auxiliary head.

    The B2 path is kept intact:
      ResNet34-U-Net + boundary/rim auxiliary supervision + RBSG-lite.

    B20 adds only one 1x1 distance-regression head on the refined final decoder
    feature. Inference still uses only the mask logits.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        seg_conv = self.unet.segmentation_head[0]
        dec_ch = seg_conv.in_channels
        self.distance_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)
        mask_logits = self.mask_head(refined)

        if not return_aux:
            return mask_logits

        return {
            "mask": mask_logits,
            "boundary": self.boundary_head(refined),
            "rim": self.rim_head(refined),
            "distance": self.distance_head(refined),
        }


def compute_b20_dmr_loss(
    outputs,
    masks,
    epoch,
    distance_max_radius=16,
    distance_weight_warmup=0.02,
    distance_weight=0.05,
):
    base_loss, details = compute_b1_loss(outputs, masks, epoch=epoch)

    distance_gt = mask_to_inner_distance_target(
        masks,
        max_radius=distance_max_radius,
    )
    distance_prob = torch.sigmoid(outputs["distance"])
    loss_distance = F.smooth_l1_loss(distance_prob, distance_gt)

    if epoch <= 20:
        wd = float(distance_weight_warmup)
    else:
        wd = float(distance_weight)

    total = base_loss + wd * loss_distance
    details = details.copy()
    details.update(
        {
            "loss_distance": float(loss_distance.detach().cpu()),
            "wd": wd,
            "distance_max_radius": int(distance_max_radius),
        }
    )
    return total, details


def run_one_epoch_b20_dmr(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
    distance_max_radius=16,
    distance_weight_warmup=0.02,
    distance_weight=0.05,
):
    model.train(train)

    total_loss = 0.0
    total_loss_distance = 0.0
    total_wd = 0.0
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
            loss, details = compute_b20_dmr_loss(
                outputs,
                masks,
                epoch=epoch,
                distance_max_radius=distance_max_radius,
                distance_weight_warmup=distance_weight_warmup,
                distance_weight=distance_weight,
            )

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
        total_loss_distance += float(details["loss_distance"])
        total_wd += float(details["wd"])
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
        "loss_distance": total_loss_distance / max(n_batches, 1),
        "wd": total_wd / max(n_batches, 1),
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
    config["model_class"] = "ResNet34UNetBRRB20DMR"
    config["experiment"] = "B20: B2 + lightweight inner Distance Map Regression auxiliary supervision"
    config["note"] = (
        "B20 keeps the B2 boundary/rim auxiliary supervision and RBSG-lite "
        "unchanged, then adds a single distance-map regression head on the "
        "refined final decoder feature. The goal is to improve sparse caries "
        "lesion completeness without increasing rim suppression or spatial "
        "attention strength."
    )
    config["failure_reflection"] = {
        "b2": "stable high-recall baseline; test Dice around 0.8842",
        "b2_preaux_and_multiscale_gate": "often raised precision but reduced recall",
        "b17_wavelet_rescue": "validation looked promising but test Dice stayed below B2, suggesting candidate-residual texture rescue can overfit",
        "design_implication": "try an auxiliary target that supplies shape/core information while preserving the exact B2 inference path",
    }
    config["design_references"] = [
        {
            "title": "Deep Distance Map Regression Network with Shape-aware Loss for Imbalanced Medical Image Segmentation",
            "year": 2025,
            "adopted_idea": "distance maps provide richer small-object shape supervision than binary masks",
        },
        {
            "title": "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation",
            "year": 2024,
            "adopted_idea": "keep a strong CNN/U-Net recipe and make conservative, well-controlled additions",
        },
        {
            "title": "When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation",
            "year": 2025,
            "adopted_idea": "dental caries segmentation can favor CNN inductive bias over heavy transformer/state-space complexity",
        },
    ]
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["dmr"] = {
        "target": "normalized inner distance target approximated by repeated binary erosions",
        "position": "1x1 head on RBSG-refined final decoder feature",
        "loss": "SmoothL1(sigmoid(distance_logits), inner_distance_gt)",
        "distance_max_radius": args.distance_max_radius,
        "epoch_1_to_20": f"{args.distance_weight_warmup} * L_distance",
        "epoch_21_to_end": f"{args.distance_weight} * L_distance",
        "inference": "mask logits only; distance head is auxiliary",
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
    model = ResNet34UNetBRRB20DMR(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("=" * 80)
    print("Model ready: B20 DMR + B2 BRR RBSG-lite ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("distance_radius =", args.distance_max_radius)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_loss_distance", "train_wd",
            "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_loss_distance", "val_wd",
            "val_dice", "val_iou", "val_precision", "val_recall",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b20_dmr(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            distance_max_radius=args.distance_max_radius,
            distance_weight_warmup=args.distance_weight_warmup,
            distance_weight=args.distance_weight,
        )
        val_metrics = run_one_epoch_b20_dmr(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
            distance_max_radius=args.distance_max_radius,
            distance_weight_warmup=args.distance_weight_warmup,
            distance_weight=args.distance_weight,
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"val_dmr={val_metrics['loss_distance']:.4f} wd={val_metrics['wd']:.3f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["loss_distance"], train_metrics["wd"],
                train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["loss_distance"], val_metrics["wd"],
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

    best_train_metrics = run_one_epoch_b20_dmr(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        distance_max_radius=args.distance_max_radius,
        distance_weight_warmup=args.distance_weight_warmup,
        distance_weight=args.distance_weight,
    )
    best_val_metrics = run_one_epoch_b20_dmr(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        distance_max_radius=args.distance_max_radius,
        distance_weight_warmup=args.distance_weight_warmup,
        distance_weight=args.distance_weight,
    )
    test_metrics = run_one_epoch_b20_dmr(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        distance_max_radius=args.distance_max_radius,
        distance_weight_warmup=args.distance_weight_warmup,
        distance_weight=args.distance_weight,
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

    parser.add_argument("--run-name", type=str, default="b20_dmr_auglite_e200_constlr_bs6")
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

    parser.add_argument("--distance-max-radius", type=int, default=16)
    parser.add_argument("--distance-weight-warmup", type=float, default=0.02)
    parser.add_argument("--distance-weight", type=float, default=0.05)

    args = parser.parse_args()
    main(args)
