from pathlib import Path
import csv
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from train_resnet34_unet3p import (
    set_seed,
    save_json,
    CariesMaskDataset,
    run_one_epoch,
    save_prediction_panel,
)


class CBSABlock(nn.Module):
    def __init__(self, e_lambda=1e-4, alpha=1.0, beta=0.1):
        super().__init__()
        self.e_lambda = e_lambda
        self.alpha = alpha
        self.beta = beta
        self.sigmoid = nn.Sigmoid()

    def simam_attention(self, x):
        b, c, h, w = x.size()
        n = h * w - 1
        mean = x.mean(dim=(2, 3), keepdim=True)
        d = (x - mean).pow(2)
        v = d.sum(dim=(2, 3), keepdim=True) / max(n, 1)
        e_inv = d / (4 * (v + self.e_lambda)) + 0.5
        return self.sigmoid(e_inv)

    def boundary_gradient(self, x):
        gx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        gx = F.pad(gx, (0, 1, 0, 0))

        gy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])
        gy = F.pad(gy, (0, 0, 0, 1))

        g = gx + gy
        g = g.mean(dim=1, keepdim=True)

        g_min = g.amin(dim=(2, 3), keepdim=True)
        g_max = g.amax(dim=(2, 3), keepdim=True)
        g = (g - g_min) / (g_max - g_min + 1e-6)
        return g

    def forward(self, x):
        simam = self.simam_attention(x)
        boundary = self.boundary_gradient(x)
        att = simam * (1.0 + self.alpha * boundary)
        return x + self.beta * x * att


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class CDFELite(nn.Module):
    """
    CDFE-lite: Contrast-Driven Feature Enhancement.

    It uses a preliminary segmentation probability map to softly split feature responses into
    foreground, background, and uncertain boundary cues. The foreground-background contrast
    generates channel-wise enhancement, while uncertain regions generate spatial refinement.
    """
    def __init__(
        self,
        channels,
        reduction=4,
        fg_thr=0.6,
        bg_thr=0.3,
        contrast_scale_init=0.1,
        uncertain_scale_init=0.1,
        detach_prob=True,
    ):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fg_thr = float(fg_thr)
        self.bg_thr = float(bg_thr)
        self.detach_prob = bool(detach_prob)

        self.channel_gate = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.uncertain_refine = nn.Sequential(
            ConvBNReLU(channels, channels, kernel_size=3, padding=1),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.spatial_gate = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )

        self.out_norm = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.contrast_scale = nn.Parameter(torch.tensor(float(contrast_scale_init)))
        self.uncertain_scale = nn.Parameter(torch.tensor(float(uncertain_scale_init)))

    @staticmethod
    def weighted_avg_pool(x, weight, eps=1e-6):
        numerator = (x * weight).sum(dim=(2, 3), keepdim=True)
        denominator = weight.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
        return numerator / denominator

    def forward(self, x, logits):
        prob = torch.sigmoid(logits)
        if self.detach_prob:
            prob = prob.detach()

        fg_w = ((prob - self.fg_thr) / max(1.0 - self.fg_thr, 1e-6)).clamp(0.0, 1.0)
        bg_w = ((self.bg_thr - prob) / max(self.bg_thr, 1e-6)).clamp(0.0, 1.0)
        uncertain_w = (1.0 - torch.abs(prob - 0.5) / 0.5).clamp(0.0, 1.0)

        fg_context = self.weighted_avg_pool(x, fg_w)
        bg_context = self.weighted_avg_pool(x, bg_w)
        contrast_context = fg_context - bg_context

        channel_att = self.channel_gate(contrast_context)
        contrast_enhanced = x * channel_att

        uncertain_feat = self.uncertain_refine(x * uncertain_w)
        spatial_att = self.spatial_gate(x * uncertain_w)
        uncertain_enhanced = uncertain_feat * spatial_att

        out = x + self.contrast_scale * contrast_enhanced + self.uncertain_scale * uncertain_enhanced
        out = self.out_norm(out)
        return out


class ResNet34UNetCBSACDFE(nn.Module):
    """
    U6:
    ResNet34-U-Net + CBSA + CDFE-lite.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cbsa_lambda=1e-4,
        cbsa_alpha=1.0,
        cbsa_beta=0.1,
        cdfe_reduction=4,
        cdfe_fg_thr=0.6,
        cdfe_bg_thr=0.3,
        cdfe_contrast_scale_init=0.1,
        cdfe_uncertain_scale_init=0.1,
        cdfe_detach_prob=True,
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        decoder_channels = self.unet.segmentation_head[0].in_channels

        self.cbsa = CBSABlock(
            e_lambda=cbsa_lambda,
            alpha=cbsa_alpha,
            beta=cbsa_beta,
        )

        self.cdfe = CDFELite(
            channels=decoder_channels,
            reduction=cdfe_reduction,
            fg_thr=cdfe_fg_thr,
            bg_thr=cdfe_bg_thr,
            contrast_scale_init=cdfe_contrast_scale_init,
            uncertain_scale_init=cdfe_uncertain_scale_init,
            detach_prob=cdfe_detach_prob,
        )

    def forward(self, x):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)
        decoder_output = self.cbsa(decoder_output)

        preliminary_logits = self.unet.segmentation_head(decoder_output)
        decoder_output = self.cdfe(decoder_output, preliminary_logits)

        masks = self.unet.segmentation_head(decoder_output)
        return masks


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
    config["model_class"] = "ResNet34UNetCBSACDFE"
    config["experiment"] = "U6: ResNet34-U-Net + CBSA + CDFE-lite"
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

    model = ResNet34UNetCBSACDFE(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cbsa_lambda=args.cbsa_lambda,
        cbsa_alpha=args.cbsa_alpha,
        cbsa_beta=args.cbsa_beta,
        cdfe_reduction=args.cdfe_reduction,
        cdfe_fg_thr=args.cdfe_fg_thr,
        cdfe_bg_thr=args.cdfe_bg_thr,
        cdfe_contrast_scale_init=args.cdfe_contrast_scale_init,
        cdfe_uncertain_scale_init=args.cdfe_uncertain_scale_init,
        cdfe_detach_prob=args.cdfe_detach_prob,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: U6 ResNet34-U-Net + CBSA + CDFE-lite")
    print("run_name                   =", args.run_name)
    print("encoder_name               =", args.encoder_name)
    print("encoder_weights            =", args.encoder_weights)
    print("cbsa_lambda                =", args.cbsa_lambda)
    print("cbsa_alpha                 =", args.cbsa_alpha)
    print("cbsa_beta                  =", args.cbsa_beta)
    print("cdfe_reduction             =", args.cdfe_reduction)
    print("cdfe_fg_thr                =", args.cdfe_fg_thr)
    print("cdfe_bg_thr                =", args.cdfe_bg_thr)
    print("cdfe_contrast_scale_init   =", args.cdfe_contrast_scale_init)
    print("cdfe_uncertain_scale_init  =", args.cdfe_uncertain_scale_init)
    print("cdfe_detach_prob           =", args.cdfe_detach_prob)
    print("batch_size                 =", args.batch_size)
    print("epochs                     =", args.epochs)
    print("lr                         =", args.lr)
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
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, deep_supervision=False, train=True)
        val_metrics = run_one_epoch(model, val_loader, optimizer=None, device=device, deep_supervision=False, train=False)

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

    best_path = ckpt_dir / "best.pth"
    model.load_state_dict(torch.load(best_path, map_location=device))

    best_train_metrics = run_one_epoch(model, train_loader, optimizer=None, device=device, deep_supervision=False, train=False)
    best_val_metrics = run_one_epoch(model, val_loader, optimizer=None, device=device, deep_supervision=False, train=False)
    test_metrics = run_one_epoch(model, test_loader, optimizer=None, device=device, deep_supervision=False, train=False)

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

    parser.add_argument("--run-name", type=str, default="resunet_cbsa_cdfe_u6_e50_bs6")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")

    parser.add_argument("--cbsa-lambda", type=float, default=1e-4)
    parser.add_argument("--cbsa-alpha", type=float, default=1.0)
    parser.add_argument("--cbsa-beta", type=float, default=0.1)

    parser.add_argument("--cdfe-reduction", type=int, default=4)
    parser.add_argument("--cdfe-fg-thr", type=float, default=0.6)
    parser.add_argument("--cdfe-bg-thr", type=float, default=0.3)
    parser.add_argument("--cdfe-contrast-scale-init", type=float, default=0.1)
    parser.add_argument("--cdfe-uncertain-scale-init", type=float, default=0.1)
    parser.add_argument("--cdfe-detach-prob", action="store_true", default=True)

    args = parser.parse_args()
    main(args)
