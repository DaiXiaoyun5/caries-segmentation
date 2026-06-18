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
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, groups=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class LMSA2DPaperAdapt(nn.Module):
    """
    LMSA-2D-paper-adapt.

    This is NOT the official EFMS-Net implementation.
    It is a 2D adaptation based on the EFMS-Net paper:
    - directional pooling along height and width;
    - channel-spatial attention through coordinate-style interaction;
    - LoG-inspired high-frequency residual: x - upsample(downsample(Gaussian(x))).

    The original paper applies LMSA to skip connections. Here we apply this block
    to ResNet34-U-Net encoder skip features before they enter the SMP decoder.
    """
    def __init__(self, channels, reduction=16, ms_kernel=5, att_scale_init=0.1, log_scale_init=0.1):
        super().__init__()
        hidden = max(8, channels // reduction)
        padding = ms_kernel // 2

        self.pool_conv = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.att_h = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.att_w = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)

        # Local multi-scale texture branch. Kept lightweight with depthwise convolutions.
        self.ms_branch = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=ms_kernel, padding=padding, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.log_refine = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )

        self.out_act = nn.ReLU(inplace=True)
        self.att_scale = nn.Parameter(torch.tensor(float(att_scale_init)))
        self.log_scale = nn.Parameter(torch.tensor(float(log_scale_init)))

        gaussian_1d = torch.tensor([1., 4., 6., 4., 1.], dtype=torch.float32)
        gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
        gaussian_2d = gaussian_2d / gaussian_2d.sum()
        self.register_buffer("gaussian_kernel", gaussian_2d.view(1, 1, 5, 5), persistent=False)

    def gaussian_smooth(self, x):
        c = x.shape[1]
        weight = self.gaussian_kernel.expand(c, 1, 5, 5).contiguous()
        return F.conv2d(x, weight, padding=2, groups=c)

    def log_like_high_freq(self, x):
        smooth = self.gaussian_smooth(x)
        # Paper idea: diff = f - upsample(downsample(f * G)).
        low = F.avg_pool2d(smooth, kernel_size=2, stride=2, ceil_mode=True)
        low_up = F.interpolate(low, size=x.shape[-2:], mode="bilinear", align_corners=False)
        diff = x - low_up
        return diff

    def forward(self, x):
        identity = x
        b, c, h, w = x.shape

        # Directional pooling: height-aware and width-aware descriptors.
        feat_h = x.mean(dim=3, keepdim=True)                  # B,C,H,1
        feat_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # B,C,W,1

        pooled = torch.cat([feat_h, feat_w], dim=2)           # B,C,H+W,1
        pooled = self.pool_conv(pooled)

        att_h, att_w = torch.split(pooled, [h, w], dim=2)
        att_w = att_w.permute(0, 1, 3, 2)

        att_h = torch.sigmoid(self.att_h(att_h))
        att_w = torch.sigmoid(self.att_w(att_w))

        ms_feat = self.ms_branch(x)
        att_feat = ms_feat * att_h * att_w

        log_feat = self.log_refine(self.log_like_high_freq(x))

        out = identity + self.att_scale * att_feat + self.log_scale * log_feat
        out = self.out_act(out)
        return out


class ResNet34UNetCBSALMSA(nn.Module):
    """
    E1:
    ResNet34-U-Net + CBSA + LMSA-2D-paper-adapt.

    Fairness design:
    - SMP ResNet34-U-Net main body unchanged.
    - CBSA placement follows previous M3/M4: decoder output -> CBSA -> segmentation head.
    - LMSA-2D is applied to encoder skip features before decoder, matching EFMS-Net's skip usage.
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
        lmsa_reduction=16,
        lmsa_ms_kernel=5,
        lmsa_att_scale_init=0.1,
        lmsa_log_scale_init=0.1,
        lmsa_skip_start=1,
        lmsa_skip_end=-1,
    ):
        super().__init__()

        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
            activation=None,
        )

        encoder_channels = list(self.unet.encoder.out_channels)
        n_feats = len(encoder_channels)

        if lmsa_skip_end < 0:
            lmsa_skip_end = n_feats + lmsa_skip_end

        self.lmsa_indices = list(range(lmsa_skip_start, lmsa_skip_end))
        if len(self.lmsa_indices) == 0:
            raise ValueError("No LMSA skip levels selected.")

        self.lmsa_blocks = nn.ModuleDict({
            str(idx): LMSA2DPaperAdapt(
                channels=encoder_channels[idx],
                reduction=lmsa_reduction,
                ms_kernel=lmsa_ms_kernel,
                att_scale_init=lmsa_att_scale_init,
                log_scale_init=lmsa_log_scale_init,
            )
            for idx in self.lmsa_indices
        })

        self.cbsa = CBSABlock(
            e_lambda=cbsa_lambda,
            alpha=cbsa_alpha,
            beta=cbsa_beta,
        )

    def forward(self, x):
        features = list(self.unet.encoder(x))

        for idx in self.lmsa_indices:
            features[idx] = self.lmsa_blocks[str(idx)](features[idx])

        decoder_output = self.unet.decoder(*features)
        decoder_output = self.cbsa(decoder_output)
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
    config["model_class"] = "ResNet34UNetCBSALMSA"
    config["experiment"] = "E1: ResNet34-U-Net + CBSA + LMSA-2D-paper-adapt"
    config["note"] = "Not official EFMS-Net LMSA; 2D paper-adapted implementation."
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

    model = ResNet34UNetCBSALMSA(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cbsa_lambda=args.cbsa_lambda,
        cbsa_alpha=args.cbsa_alpha,
        cbsa_beta=args.cbsa_beta,
        lmsa_reduction=args.lmsa_reduction,
        lmsa_ms_kernel=args.lmsa_ms_kernel,
        lmsa_att_scale_init=args.lmsa_att_scale_init,
        lmsa_log_scale_init=args.lmsa_log_scale_init,
        lmsa_skip_start=args.lmsa_skip_start,
        lmsa_skip_end=args.lmsa_skip_end,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: E1 ResNet34-U-Net + CBSA + LMSA-2D-paper-adapt")
    print("run_name              =", args.run_name)
    print("encoder_name          =", args.encoder_name)
    print("encoder_weights       =", args.encoder_weights)
    print("cbsa_lambda           =", args.cbsa_lambda)
    print("cbsa_alpha            =", args.cbsa_alpha)
    print("cbsa_beta             =", args.cbsa_beta)
    print("lmsa_reduction        =", args.lmsa_reduction)
    print("lmsa_ms_kernel        =", args.lmsa_ms_kernel)
    print("lmsa_att_scale_init   =", args.lmsa_att_scale_init)
    print("lmsa_log_scale_init   =", args.lmsa_log_scale_init)
    print("lmsa_skip_start       =", args.lmsa_skip_start)
    print("lmsa_skip_end         =", args.lmsa_skip_end)
    print("batch_size            =", args.batch_size)
    print("epochs                =", args.epochs)
    print("lr                    =", args.lr)
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

    parser.add_argument("--run-name", type=str, default="resunet_cbsa_lmsa_e1_e50_bs6")
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

    parser.add_argument("--lmsa-reduction", type=int, default=16)
    parser.add_argument("--lmsa-ms-kernel", type=int, default=5)
    parser.add_argument("--lmsa-att-scale-init", type=float, default=0.1)
    parser.add_argument("--lmsa-log-scale-init", type=float, default=0.1)
    parser.add_argument("--lmsa-skip-start", type=int, default=1)
    parser.add_argument("--lmsa-skip-end", type=int, default=-1)

    args = parser.parse_args()
    main(args)
