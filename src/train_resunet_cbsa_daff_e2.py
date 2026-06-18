from pathlib import Path
import csv
import argparse
import math

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


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, groups=1, act=True):
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
        ]
        if act:
            layers.append(nn.GELU())
        super().__init__(*layers)


class MultiFrequencyDescriptor2D(nn.Module):
    """
    Lightweight 2D frequency descriptor.

    This is not exact DCT implementation from EFMS-Net.
    It creates a small set of fixed cosine basis maps on pooled features to
    approximate low/mid/high frequency responses in a stable, pure-PyTorch way.
    """
    def __init__(self, channels, pool_size=8, freq_indices=((0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (2, 2), (3, 1))):
        super().__init__()
        self.channels = channels
        self.pool_size = int(pool_size)
        self.freq_indices = list(freq_indices)
        self.num_freq = len(self.freq_indices)

        basis = []
        n = self.pool_size
        y = torch.arange(n, dtype=torch.float32).view(n, 1)
        x = torch.arange(n, dtype=torch.float32).view(1, n)

        for u, v in self.freq_indices:
            by = torch.cos(math.pi * (2 * y + 1) * u / (2 * n))
            bx = torch.cos(math.pi * (2 * x + 1) * v / (2 * n))
            b = by @ bx
            b = b / (b.abs().sum() + 1e-6)
            basis.append(b)

        basis = torch.stack(basis, dim=0)  # F,H,W
        self.register_buffer("basis", basis, persistent=False)

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        # B,C,F
        desc = torch.einsum("bchw,fhw->bcf", pooled, self.basis)
        return desc


class AFW2D(nn.Module):
    """
    Adaptive Frequency Weighting for 2D descriptors.

    Input : x feature map
    Output: channel-wise frequency attention, B,C,1,1
    """
    def __init__(self, channels, reduction=8, pool_size=8, num_freq=8):
        super().__init__()
        self.desc = MultiFrequencyDescriptor2D(channels, pool_size=pool_size)
        hidden = max(channels // reduction, 8)

        self.mlp = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
        )

        self.freq_logits = nn.Sequential(
            nn.Linear(num_freq, num_freq, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(num_freq, num_freq, bias=True),
        )

    def forward(self, x):
        d = self.desc(x)                       # B,C,F
        freq_w = torch.softmax(self.freq_logits(d), dim=-1)
        d = (d * freq_w).sum(dim=-1, keepdim=True)  # B,C,1
        att = torch.sigmoid(self.mlp(d)).unsqueeze(-1)  # B,C,1,1
        return att


class DAFF2DPaperAdapt(nn.Module):
    """
    DAFF-2D-paper-adapt.

    This is NOT the official EFMS-Net DAFF implementation.
    Original DAFF uses a 3D cascaded depthwise convolution branch, a bidirectional
    Mamba branch, DCT-based frequency representations and AFW.

    This 2D adaptation keeps the core motivation but removes Mamba dependency:
    - local branch: cascaded depthwise separable convolutions;
    - context branch: large-kernel/dilated depthwise convolutions as a stable
      no-Mamba global-context surrogate;
    - AFW2D: cosine-frequency descriptor based channel weighting;
    - soft branch fusion + residual output.

    Use this only as a paper-adapted exploratory experiment.
    """
    def __init__(
        self,
        channels,
        reduction=8,
        freq_pool_size=8,
        branch_scale_init=0.1,
        groups_shuffle=4,
    ):
        super().__init__()
        self.channels = channels
        self.groups_shuffle = int(groups_shuffle)

        self.local_branch = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0),
            ConvBNAct(channels, channels, kernel_size=3, padding=1, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0, act=False),
        )

        self.context_branch = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=5, padding=2, groups=channels),
            ConvBNAct(channels, channels, kernel_size=3, padding=2, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, padding=0, act=False),
        )
        # Set dilation for second depthwise conv in context branch.
        self.context_branch[1][0].dilation = (2, 2)

        self.local_afw = AFW2D(channels, reduction=reduction, pool_size=freq_pool_size)
        self.context_afw = AFW2D(channels, reduction=reduction, pool_size=freq_pool_size)

        self.branch_score = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, max(channels // reduction, 8), kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // reduction, 8), 2, kernel_size=1, bias=True),
        )

        self.fuse = ConvBNAct(channels, channels, kernel_size=1, padding=0, act=False)
        self.out_act = nn.GELU()
        self.branch_scale = nn.Parameter(torch.tensor(float(branch_scale_init)))

    def channel_shuffle(self, x):
        b, c, h, w = x.shape
        g = self.groups_shuffle
        if g <= 1 or c % g != 0:
            return x
        x = x.view(b, g, c // g, h, w)
        x = x.transpose(1, 2).contiguous()
        x = x.view(b, c, h, w)
        return x

    def forward(self, x):
        local = self.local_branch(x)
        context = self.context_branch(x)

        local = local * self.local_afw(local)
        context = context * self.context_afw(context)

        scores = self.branch_score(torch.cat([local, context], dim=1))
        weights = torch.softmax(scores, dim=1)
        fused = weights[:, 0:1] * local + weights[:, 1:2] * context
        fused = self.channel_shuffle(fused)
        fused = self.fuse(fused)

        out = x + self.branch_scale * fused
        out = self.out_act(out)
        return out


class ResNet34UNetCBSADAFF(nn.Module):
    """
    E2:
    ResNet34-U-Net + CBSA + DAFF-2D-paper-adapt.

    Fairness design:
    - SMP ResNet34-U-Net main body unchanged.
    - DAFF-2D is applied to bottleneck/deep encoder feature before decoder.
    - CBSA placement follows previous M3/M4: decoder output -> CBSA -> segmentation head.
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
        daff_reduction=8,
        daff_freq_pool_size=8,
        daff_scale_init=0.1,
        daff_feature_index=-1,
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
        if daff_feature_index < 0:
            daff_feature_index = n_feats + daff_feature_index

        if daff_feature_index < 0 or daff_feature_index >= n_feats:
            raise ValueError(f"Invalid daff_feature_index={daff_feature_index}, encoder channels={encoder_channels}")

        self.daff_feature_index = int(daff_feature_index)

        self.daff = DAFF2DPaperAdapt(
            channels=encoder_channels[self.daff_feature_index],
            reduction=daff_reduction,
            freq_pool_size=daff_freq_pool_size,
            branch_scale_init=daff_scale_init,
        )

        self.cbsa = CBSABlock(
            e_lambda=cbsa_lambda,
            alpha=cbsa_alpha,
            beta=cbsa_beta,
        )

    def forward(self, x):
        features = list(self.unet.encoder(x))
        features[self.daff_feature_index] = self.daff(features[self.daff_feature_index])

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
    config["model_class"] = "ResNet34UNetCBSADAFF"
    config["experiment"] = "E2: ResNet34-U-Net + CBSA + DAFF-2D-paper-adapt"
    config["note"] = "Not official EFMS-Net DAFF. No Mamba dependency; 2D no-Mamba paper-adapted exploratory implementation."
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

    model = ResNet34UNetCBSADAFF(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cbsa_lambda=args.cbsa_lambda,
        cbsa_alpha=args.cbsa_alpha,
        cbsa_beta=args.cbsa_beta,
        daff_reduction=args.daff_reduction,
        daff_freq_pool_size=args.daff_freq_pool_size,
        daff_scale_init=args.daff_scale_init,
        daff_feature_index=args.daff_feature_index,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: E2 ResNet34-U-Net + CBSA + DAFF-2D-paper-adapt")
    print("run_name             =", args.run_name)
    print("encoder_name         =", args.encoder_name)
    print("encoder_weights      =", args.encoder_weights)
    print("cbsa_lambda          =", args.cbsa_lambda)
    print("cbsa_alpha           =", args.cbsa_alpha)
    print("cbsa_beta            =", args.cbsa_beta)
    print("daff_reduction       =", args.daff_reduction)
    print("daff_freq_pool_size  =", args.daff_freq_pool_size)
    print("daff_scale_init      =", args.daff_scale_init)
    print("daff_feature_index   =", args.daff_feature_index)
    print("batch_size           =", args.batch_size)
    print("epochs               =", args.epochs)
    print("lr                   =", args.lr)
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

    parser.add_argument("--run-name", type=str, default="resunet_cbsa_daff_e2_e50_bs6")
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

    parser.add_argument("--daff-reduction", type=int, default=8)
    parser.add_argument("--daff-freq-pool-size", type=int, default=8)
    parser.add_argument("--daff-scale-init", type=float, default=0.1)
    parser.add_argument("--daff-feature-index", type=int, default=-1)

    args = parser.parse_args()
    main(args)
