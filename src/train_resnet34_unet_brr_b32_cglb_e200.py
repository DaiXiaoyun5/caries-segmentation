import argparse
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    save_prediction_panel,
)

from train_resnet34_unet_auglite_e200_constlr import AugmentedCariesMaskDataset
from train_resnet34_unet_brr_b1_aux_e200 import run_one_epoch_b1
from train_resnet34_unet_brr_b2_rbsg_e200 import RBSGLite
from train_resnet34_unet_brr_b30_ssgcb_light_e200 import build_augmentation_config


class GlobalResponseNorm2d(nn.Module):
    """
    Channel-first Global Response Normalization.

    This follows the ConvNeXt-v2/MedNeXt-v2 style idea: normalize each channel
    response by the sample-wise average global response so low-contrast
    medical structures can benefit from global activation competition without
    adding an explicit spatial suppression gate.
    """
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.eps = float(eps)
        self.gamma = nn.Parameter(torch.zeros(1, int(channels), 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, int(channels), 1, 1))

    def forward(self, x):
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + self.eps)
        return x + self.gamma * (x * nx) + self.beta


class ConvNeXtGRNLargeKernelBottleneck(nn.Module):
    """
    CGLB-lite: ConvNeXt/GRN large-kernel bottleneck context block.

    Input:
      encoder bottleneck feature F in R^{C x H x W}

    Steps:
      1) 1x1 reduction C -> c_mid
      2) depthwise large-kernel convolution for contextual receptive field
      3) GRN for global response calibration
      4) pointwise expansion/projection
      5) bounded residual fusion: F_out = F + s * context(F)

    Compared with B30 SSGCB, this block removes softmax scale selection and
    spatial gating. It is intentionally more conservative and should be less
    likely to suppress small caries positives.
    """
    def __init__(
        self,
        channels,
        hidden_channels=128,
        kernel_size=7,
        expansion=2.0,
        dropout=0.0,
        scale_init=0.05,
        scale_max=0.30,
    ):
        super().__init__()

        channels = int(channels)
        hidden_channels = int(hidden_channels)
        kernel_size = int(kernel_size)
        expanded_channels = int(round(hidden_channels * float(expansion)))
        padding = kernel_size // 2

        self.scale_max = float(scale_max)

        self.reduce = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
        )

        self.context = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=hidden_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            GlobalResponseNorm2d(hidden_channels),
        )

        self.ffn = nn.Sequential(
            nn.Conv2d(hidden_channels, expanded_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(expanded_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(float(dropout)) if float(dropout) > 0 else nn.Identity(),
            nn.Conv2d(expanded_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        init_ratio = min(max(float(scale_init) / max(self.scale_max, 1e-6), 1e-4), 1.0 - 1e-4)
        self.scale_logit = nn.Parameter(torch.tensor(math.log(init_ratio / (1.0 - init_ratio))))

    def residual_scale(self):
        return self.scale_max * torch.sigmoid(self.scale_logit)

    def forward(self, x):
        context = self.reduce(x)
        context = self.context(context)
        context = self.ffn(context)
        return x + self.residual_scale() * context


class ResNet34UNetBRRB32CGLB(nn.Module):
    """
    B32: B2 + CGLB-lite on the deepest encoder bottleneck feature.

    Keeps:
      - original B2 boundary/rim auxiliary supervision
      - original B2 RBSG-lite final decoder feature recalibration

    Adds:
      - a conservative ConvNeXt/GRN large-kernel bottleneck context module
        before the decoder.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cglb_hidden_channels=128,
        cglb_kernel_size=7,
        cglb_expansion=2.0,
        cglb_dropout=0.0,
        cglb_scale_init=0.05,
        cglb_scale_max=0.30,
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

        try:
            bottleneck_ch = int(self.unet.encoder.out_channels[-1])
        except Exception:
            bottleneck_ch = 512

        self.cglb = ConvNeXtGRNLargeKernelBottleneck(
            channels=bottleneck_ch,
            hidden_channels=cglb_hidden_channels,
            kernel_size=cglb_kernel_size,
            expansion=cglb_expansion,
            dropout=cglb_dropout,
            scale_init=cglb_scale_init,
            scale_max=cglb_scale_max,
        )

        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rbsg = RBSGLite(dec_ch, lambda_pos=0.20, lambda_rim=0.10)

        self.mask_head = self.unet.segmentation_head
        self.boundary_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))
        features[-1] = self.cglb(features[-1])

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
    config["model_class"] = "ResNet34UNetBRRB32CGLB"
    config["experiment"] = "B32: B2 BRR-RBSG + ConvNeXt-GRN Large-kernel Bottleneck"
    config["note"] = (
        "B32 keeps the original B2 boundary/rim auxiliary supervision and "
        "RBSG-lite. It replaces the B30 SSGCB idea with a more conservative "
        "ConvNeXt/GRN large-kernel bottleneck context module on the deepest "
        "encoder feature before the decoder."
    )
    config["aux_targets"] = {
        "boundary": "dilate(mask,3)-erode(mask,3)",
        "rim": "dilate(mask,9)-dilate(mask,3)",
    }
    config["rbsg"] = {
        "position": "final decoder feature before segmentation head",
        "lambda_pos": 0.20,
        "lambda_rim": 0.10,
        "positive_gate": "enhance boundary-supportive decoder feature",
        "rim_gate": "suppress likely peri-lesion hard-negative activation",
    }
    config["cglb"] = {
        "enabled": True,
        "position": "deepest encoder bottleneck feature before decoder",
        "hidden_channels": args.cglb_hidden_channels,
        "kernel_size": args.cglb_kernel_size,
        "expansion": args.cglb_expansion,
        "grn": True,
        "dropout": args.cglb_dropout,
        "scale_init": args.cglb_scale_init,
        "scale_max": args.cglb_scale_max,
        "fusion": "F_out = F + bounded_scale * CGLB(F)",
    }
    config["loss"] = {
        "main": "BCEWithLogits + Dice",
        "boundary": "BCEWithLogits + Dice",
        "rim": "BCEWithLogits",
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["augmentation"] = build_augmentation_config(args)
    save_json(config, run_dir / "config.json")

    train_ds = AugmentedCariesMaskDataset(
        train_split,
        target_size=(args.image_size, args.image_size),
        profile=args.augment_profile,
    )
    val_ds = CariesMaskDataset(val_split, target_size=(args.image_size, args.image_size))
    test_ds = CariesMaskDataset(test_split, target_size=(args.image_size, args.image_size))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet34UNetBRRB32CGLB(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cglb_hidden_channels=args.cglb_hidden_channels,
        cglb_kernel_size=args.cglb_kernel_size,
        cglb_expansion=args.cglb_expansion,
        cglb_dropout=args.cglb_dropout,
        cglb_scale_init=args.cglb_scale_init,
        cglb_scale_max=args.cglb_scale_max,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B32 CGLB + B2 BRR-RBSG ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("cglb_hidden_channels =", args.cglb_hidden_channels)
    print("cglb_kernel_size     =", args.cglb_kernel_size)
    print("cglb_expansion       =", args.cglb_expansion)
    print("cglb_scale_init      =", args.cglb_scale_init)
    print("cglb_scale_max       =", args.cglb_scale_max)
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
        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)
        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f}"
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

    best_train_metrics = run_one_epoch_b1(model, train_loader, optimizer=None, device=device, train=False, epoch=args.epochs)
    best_val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=args.epochs)
    test_metrics = run_one_epoch_b1(model, test_loader, optimizer=None, device=device, train=False, epoch=args.epochs)

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

    parser.add_argument("--run-name", type=str, default="b32_cglb_auglite_e200_constlr_bs6")
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

    parser.add_argument("--cglb-hidden-channels", type=int, default=128)
    parser.add_argument("--cglb-kernel-size", type=int, default=7)
    parser.add_argument("--cglb-expansion", type=float, default=2.0)
    parser.add_argument("--cglb-dropout", type=float, default=0.0)
    parser.add_argument("--cglb-scale-init", type=float, default=0.05)
    parser.add_argument("--cglb-scale-max", type=float, default=0.30)

    args = parser.parse_args()
    main(args)
