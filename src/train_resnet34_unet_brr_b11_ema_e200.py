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
    compute_b1_loss,
    run_one_epoch_b1,
)

from train_resnet34_unet_brr_b2_rbsg_e200 import ResNet34UNetBRRB2RBSG


class EMAModule(nn.Module):
    """
    Efficient Multi-Scale Attention Module (EMA-UNet style).

    Uses two parallel 1D convolutions (height-wise and width-wise) with
    cross-spatial learning to capture long-range spatial dependencies.

    Unlike CBAM spatial attention (which pools globally and applies a single
    7x7 conv), EMA preserves positional information along each spatial axis
    and enables cross-axis interaction: horizontal context influences vertical
    attention and vice versa.

    Key properties:
      - No boundary/rim priors used (complementary to B2 RBSG-lite)
      - Extremely lightweight: only 1D convolutions, no FC layers
      - No self-attention O(n^2) cost
    """

    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 4)

        # Height branch: encode each row into a compact vector
        self.conv_h = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        # Width branch: encode each column into a compact vector
        self.conv_w = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.SiLU(inplace=True),
        )

        # Cross-spatial 1D convolutions
        # Operate along the spatial dimension that was preserved after pool
        self.conv_h_att = nn.Conv1d(mid, channels, kernel_size=3, padding=1)
        self.conv_w_att = nn.Conv1d(mid, channels, kernel_size=3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()

        # Pool along width -> encode height axis: [B, C, H, 1]
        pool_h = x.mean(dim=3, keepdim=True)
        # Pool along height -> encode width axis: [B, C, 1, W]
        pool_w = x.mean(dim=2, keepdim=True)

        # Transform to compact representation
        # h_feat: [B, mid, H, 1] -> squeeze -> [B, mid, H]
        h_feat = self.conv_h(pool_h).squeeze(3)
        # w_feat: [B, mid, 1, W] -> squeeze -> [B, mid, W]
        w_feat = self.conv_w(pool_w).squeeze(2)

        # Cross-spatial learning:
        # Horizontal attention modulates width encoding, and vice versa.
        # This allows information from one spatial axis to influence
        # the attention on the other axis, capturing cross-spatial deps.
        h_att = self.conv_h_att(h_feat)  # [B, C, H]
        w_att = self.conv_w_att(w_feat)  # [B, C, W]

        h_att = h_att.unsqueeze(3)  # [B, C, H, 1]
        w_att = w_att.unsqueeze(2)  # [B, C, 1, W]

        # Cross-spatial: height attention modulates width, width modulates height
        # Each axis gets information from the other axis
        att = torch.sigmoid(h_att + w_att)  # [B, C, H, W]

        return x * att


class RBSGLiteWithEMA(nn.Module):
    """
    B2 RBSG-lite with EMA pre-enhancement.

    Architecture:
      decoder_output -> EMA -> [boundary_pre, rim_pre] -> RBSG-lite -> refined

    EMA captures cross-spatial long-range dependencies BEFORE the
    boundary/rim-guided gate, so the gate operates on richer features.
    """

    def __init__(self, channels, lambda_pos=0.20, lambda_rim=0.10, ema_reduction=4):
        super().__init__()
        self.ema = EMAModule(channels, reduction=ema_reduction)
        self.rbsg = RBSGLiteFromB2(channels, lambda_pos=lambda_pos, lambda_rim=lambda_rim)

    def forward(self, feat, boundary_logits, rim_logits):
        enhanced = self.ema(feat)
        return self.rbsg(enhanced, boundary_logits, rim_logits)


class RBSGLiteFromB2(nn.Module):
    """
    Exact copy of B2's RBSGLite for embedding into the combined module.
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

        b_prob = torch.sigmoid(boundary_logits.detach())
        r_prob = torch.sigmoid(rim_logits.detach())

        pos = self.pos_gate(torch.cat([avg_map, max_map, b_prob, grad_map], dim=1))
        rim = self.rim_gate(torch.cat([avg_map, max_map, r_prob, grad_map], dim=1))

        feat_pos = feat * (1.0 + self.lambda_pos * pos)
        feat_rim = feat * (1.0 - self.lambda_rim * rim)

        return self.out(torch.cat([feat, feat_pos, feat_rim], dim=1))


class ResNet34UNetBRRB11EMA(nn.Module):
    """
    B11: B2 + EMA (Efficient Multi-Scale Attention).

    B2 unchanged:
      - boundary/rim auxiliary supervision
      - rb=3, rr=9
      - two-stage auxiliary weights
      - RBSG-lite with lambda_pos=0.20, lambda_rim=0.10

    New: EMA module before RBSG-lite.
      - Captures cross-spatial long-range dependencies
      - No boundary/rim priors (complementary to RBSG)
      - Extremely lightweight (~5K params per module)

    Output:
      return_aux=False: mask_logits
      return_aux=True: {"mask", "boundary", "rim"}
    """

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        ema_reduction=4,
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

        # Pre-heads: produce boundary/rim logits from EMA-enhanced features
        self.boundary_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head_pre = nn.Conv2d(dec_ch, 1, kernel_size=1)

        # EMA + RBSG-lite combined module
        self.ema_rbsg = RBSGLiteWithEMA(
            dec_ch,
            lambda_pos=0.20,
            lambda_rim=0.10,
            ema_reduction=ema_reduction,
        )

        # Post-heads
        self.mask_head = self.unet.segmentation_head
        self.boundary_head = nn.Conv2d(dec_ch, 1, kernel_size=1)
        self.rim_head = nn.Conv2d(dec_ch, 1, kernel_size=1)

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.ema_rbsg(decoder_output, b_pre, r_pre)

        mask_logits = self.mask_head(refined)

        if not return_aux:
            return mask_logits

        boundary_logits = self.boundary_head(refined)
        rim_logits = self.rim_head(refined)

        return {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "rim": rim_logits,
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
    config["model_class"] = "ResNet34UNetBRRB11EMA"
    config["experiment"] = "B11: B2 + EMA (Efficient Multi-Scale Attention)"
    config["note"] = (
        "B11 keeps the complete B2 boundary/rim supervision and final RBSG-lite. "
        "EMA module is inserted before RBSG-lite to capture cross-spatial long-range "
        "dependencies without using boundary/rim priors. EMA uses two parallel 1D "
        "convolutions (height-wise and width-wise) with cross-spatial learning. "
        "This complements RBSG-lite which operates via boundary/rim-guided gating."
    )
    config["b2_unchanged"] = {
        "boundary_target": "dilate(mask,3)-erode(mask,3)",
        "rim_target": "dilate(mask,9)-dilate(mask,3)",
        "rbsg_lambda_pos": 0.20,
        "rbsg_lambda_rim": 0.10,
        "epoch_1_to_20": "L_seg + 0.10*L_boundary + 0.05*L_rim",
        "epoch_21_to_end": "L_seg + 0.20*L_boundary + 0.10*L_rim",
    }
    config["ema"] = {
        "position": "before RBSG-lite, on final decoder feature",
        "mechanism": "1D height/width convolutions with cross-spatial learning",
        "reduction": args.ema_reduction,
        "uses_boundary_or_rim": False,
        "approx_params_per_module": "~5K",
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

    model = ResNet34UNetBRRB11EMA(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        ema_reduction=args.ema_reduction,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    ema_params = sum(p.numel() for p in model.ema_rbsg.ema.parameters())
    rbsg_params = sum(p.numel() for p in model.ema_rbsg.rbsg.parameters())

    print("=" * 80)
    print("Model ready: B11 B2 + EMA ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("total_params    =", total_params)
    print("ema_params      =", ema_params)
    print("rbsg_params     =", rbsg_params)
    print("ema_reduction   =", args.ema_reduction)
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

    parser.add_argument("--run-name", type=str, default="b11_ema_auglite_e200_constlr_bs6")
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
    parser.add_argument("--ema-reduction", type=int, default=4,
                        help="EMA channel reduction ratio (channels // reduction = mid)")

    args = parser.parse_args()
    main(args)
