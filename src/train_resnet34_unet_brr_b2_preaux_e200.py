import argparse
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    save_prediction_panel,
)

from train_resnet34_unet_auglite_e200_constlr import AugmentedCariesMaskDataset

from train_resnet34_unet_brr_b1_aux_e200 import (
    bce_dice_loss,
    mask_to_boundary_and_rim,
)


class RBSGLite(nn.Module):
    """
    Rim-Boundary Skip Gate lite.

    First B2 version:
      - does not rewrite SMP decoder internals
      - applies boundary/rim-guided gate on final decoder feature
      - inference still uses only mask logits
    """
    def __init__(self, channels, lambda_pos=0.20, lambda_rim=0.10):
        super().__init__()
        self.lambda_pos = float(lambda_pos)
        self.lambda_rim = float(lambda_rim)

        self.pos_gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.rim_gate = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        self.out = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def feature_gradient(x):
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gy = F.pad(gy, (0, 0, 0, 1))
        g = (gx + gy).mean(dim=1, keepdim=True)

        g_min = g.amin(dim=(2, 3), keepdim=True)
        g_max = g.amax(dim=(2, 3), keepdim=True)
        return (g - g_min) / (g_max - g_min + 1e-6)

    def forward(self, feat, boundary_logits, rim_logits):
        avg_map = feat.mean(dim=1, keepdim=True)
        max_map = feat.max(dim=1, keepdim=True)[0]
        grad_map = self.feature_gradient(feat)

        # detach aux maps to avoid unstable early mutual feedback
        b_prob = torch.sigmoid(boundary_logits.detach())
        r_prob = torch.sigmoid(rim_logits.detach())

        pos = self.pos_gate(torch.cat([avg_map, max_map, b_prob, grad_map], dim=1))
        rim = self.rim_gate(torch.cat([avg_map, max_map, r_prob, grad_map], dim=1))

        feat_pos = feat * (1.0 + self.lambda_pos * pos)
        feat_rim = feat * (1.0 - self.lambda_rim * rim)

        return self.out(torch.cat([feat, feat_pos, feat_rim], dim=1))


class ResNet34UNetBRRB2RBSG(nn.Module):
    """
    B2: B1 + RBSG-lite decoder feature gate.

    Output:
      return_aux=False:
        mask_logits

      return_aux=True:
        {
          "mask": mask_logits,
          "boundary": boundary_logits,
          "rim": rim_logits
        }
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        seg_conv = self.unet.segmentation_head[0]
        dec_ch = seg_conv.in_channels

        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)

        self.rbsg = RBSGLite(dec_ch, lambda_pos=0.20, lambda_rim=0.10)

        self.mask_head = self.unet.segmentation_head
        self.boundary_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)

        mask_logits = self.mask_head(refined)

        if not return_aux:
            return mask_logits

        boundary_logits = self.boundary_head(refined)
        rim_logits = self.rim_head(refined)

        return {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "rim": rim_logits,
            "boundary_pre": b_pre,
            "rim_pre": r_pre,
        }


def compute_b2_preaux_loss(
    outputs,
    masks,
    epoch,
    warmup_epochs=20,
    pre_boundary_weight=0.10,
    pre_rim_weight=0.05,
):
    """
    B2-preaux loss.

    It keeps original B2:
      L_seg + wb * L_boundary_final + wr * L_rim_final

    And adds direct supervision to the pre-gate hints:
      + wpb * L_boundary_pre + wpr * L_rim_pre

    rb=3, rr=9 are kept exactly the same as original B2/B1.
    lambda_pos=0.20, lambda_rim=0.10 are kept inside RBSGLite.
    """
    mask_logits = outputs["mask"]
    boundary_logits = outputs["boundary"]
    rim_logits = outputs["rim"]
    boundary_pre_logits = outputs["boundary_pre"]
    rim_pre_logits = outputs["rim_pre"]

    boundary_gt, rim_gt = mask_to_boundary_and_rim(masks, rb=3, rr=9)

    loss_seg = bce_dice_loss(mask_logits, masks)

    loss_boundary = bce_dice_loss(boundary_logits, boundary_gt)
    loss_rim = F.binary_cross_entropy_with_logits(rim_logits, rim_gt)

    loss_boundary_pre = bce_dice_loss(boundary_pre_logits, boundary_gt)
    loss_rim_pre = F.binary_cross_entropy_with_logits(rim_pre_logits, rim_gt)

    if epoch <= warmup_epochs:
        wb = 0.10
        wr = 0.05
        wpb = min(float(pre_boundary_weight), 0.05)
        wpr = min(float(pre_rim_weight), 0.025)
    else:
        wb = 0.20
        wr = 0.10
        wpb = float(pre_boundary_weight)
        wpr = float(pre_rim_weight)

    total = (
        loss_seg
        + wb * loss_boundary
        + wr * loss_rim
        + wpb * loss_boundary_pre
        + wpr * loss_rim_pre
    )

    return total, {
        "loss_seg": float(loss_seg.detach().cpu()),
        "loss_boundary": float(loss_boundary.detach().cpu()),
        "loss_rim": float(loss_rim.detach().cpu()),
        "loss_boundary_pre": float(loss_boundary_pre.detach().cpu()),
        "loss_rim_pre": float(loss_rim_pre.detach().cpu()),
        "wb": wb,
        "wr": wr,
        "wpb": wpb,
        "wpr": wpr,
    }


def run_one_epoch_b2_preaux(
    model,
    loader,
    optimizer,
    device,
    train=True,
    epoch=1,
    pre_boundary_weight=0.10,
    pre_rim_weight=0.05,
):
    """
    Train/eval one epoch for B2-preaux.

    Metrics are computed only from final mask logits.
    Pre-boundary/pre-rim supervision only affects training loss.
    """
    model.train(train)

    total_loss = 0.0
    total_tp = 0.0
    total_fp = 0.0
    total_fn = 0.0
    total_tn = 0.0
    n_batches = 0

    last_items = {
        "loss_boundary_pre": 0.0,
        "loss_rim_pre": 0.0,
        "wpb": 0.0,
        "wpr": 0.0,
    }

    for images, masks, _, _ in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            outputs = model(images, return_aux=True)
            loss, loss_items = compute_b2_preaux_loss(
                outputs,
                masks,
                epoch=epoch,
                pre_boundary_weight=pre_boundary_weight,
                pre_rim_weight=pre_rim_weight,
            )

            if train:
                loss.backward()
                optimizer.step()

        last_items = loss_items

        logits = outputs["mask"].detach()
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        gt = (masks > 0.5).float()

        tp = (preds * gt).sum()
        fp = (preds * (1.0 - gt)).sum()
        fn = ((1.0 - preds) * gt).sum()
        tn = ((1.0 - preds) * (1.0 - gt)).sum()

        total_loss += float(loss.detach().cpu())
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
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "tn": total_tn,
        "loss_boundary_pre": last_items["loss_boundary_pre"],
        "loss_rim_pre": last_items["loss_rim_pre"],
        "wpb": last_items["wpb"],
        "wpr": last_items["wpr"],
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
    config["model_class"] = "ResNet34UNetBRRB2RBSG"
    config["experiment"] = "B2-preaux: B2 BRR-RBSG with pre-boundary/pre-rim supervision"
    config["note"] = "B2-preaux keeps B2 architecture and lambda_pos=0.20/lambda_rim=0.10/rb=3/rr=9, and adds direct supervision to boundary_head_pre and rim_head_pre so that RBSG hints become more semantically reliable."
    config["aux_targets"] = {
        "boundary": "dilate(mask,3)-erode(mask,3)",
        "rim": "dilate(mask,9)-dilate(mask,3)"
    }
    config["rbsg"] = {
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation"
    }
    config["loss"] = {
        "main": "BCEWithLogits + Dice",
        "boundary_final": "BCEWithLogits + Dice",
        "rim_final": "BCEWithLogits",
        "boundary_pre": "BCEWithLogits + Dice",
        "rim_pre": "BCEWithLogits",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim + 0.05*L_boundary_pre + 0.025*L_rim_pre",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim + pre_boundary_weight*L_boundary_pre + pre_rim_weight*L_rim_pre",
        "pre_boundary_weight": args.pre_boundary_weight,
        "pre_rim_weight": args.pre_rim_weight
    }
    config["augmentation"] = {
        "type": "online",
        "profile": args.augment_profile,
        "train_only": True,
        "geometry": {
            "rotation_deg": [-15.0, 15.0],
            "scale": [0.85, 1.20],
            "translate_ratio": 0.05,
            "horizontal_flip_p": 0.5,
            "vertical_flip_p": 0.0
        },
        "photometric": {
            "brightness_factor": [0.85, 1.15],
            "contrast_factor": [0.85, 1.15],
            "gamma": [0.80, 1.25],
            "gaussian_noise_p": 0.10,
            "noise_std": [0.005, 0.020],
            "gaussian_blur_p": 0.10,
            "blur_radius": [0.2, 0.8]
        }
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

    model = ResNet34UNetBRRB2RBSG(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B2-preaux BRR-RBSG ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("pre_boundary_weight =", args.pre_boundary_weight)
    print("pre_rim_weight      =", args.pre_rim_weight)
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
        train_metrics = run_one_epoch_b2_preaux(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
            pre_boundary_weight=args.pre_boundary_weight,
            pre_rim_weight=args.pre_rim_weight,
        )
        val_metrics = run_one_epoch_b2_preaux(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
            pre_boundary_weight=args.pre_boundary_weight,
            pre_rim_weight=args.pre_rim_weight,
        )

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} "
            f"pre_bd={train_metrics['loss_boundary_pre']:.4f} pre_rim={train_metrics['loss_rim_pre']:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))

    best_train_metrics = run_one_epoch_b2_preaux(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        pre_boundary_weight=args.pre_boundary_weight,
        pre_rim_weight=args.pre_rim_weight,
    )
    best_val_metrics = run_one_epoch_b2_preaux(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        pre_boundary_weight=args.pre_boundary_weight,
        pre_rim_weight=args.pre_rim_weight,
    )
    test_metrics = run_one_epoch_b2_preaux(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
        pre_boundary_weight=args.pre_boundary_weight,
        pre_rim_weight=args.pre_rim_weight,
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

    parser.add_argument("--run-name", type=str, default="b2_preaux_rbsg_auglite_e200_constlr_bs6")
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

    parser.add_argument("--pre-boundary-weight", type=float, default=0.10)
    parser.add_argument("--pre-rim-weight", type=float, default=0.05)

    args = parser.parse_args()
    main(args)
