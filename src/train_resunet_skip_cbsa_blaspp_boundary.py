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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class CBSABlock(nn.Module):
    """
    CBSA: SimAM + feature-gradient boundary cue + residual scaling.
    用在 skip connection 上，增强编码器传给解码器的弱边界细节。
    """
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


class LiteASPPBlock(nn.Module):
    """
    Lite-ASPP 放在 ResNet34 encoder 最深层 bottleneck 特征上。
    """
    def __init__(self, in_channels, out_channels=None, mid_channels=128, dropout=0.1):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=4, dilation=4, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels * 4, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        bp = self.image_pool(x)
        bp = F.interpolate(bp, size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat([b1, b2, b3, bp], dim=1))


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(skip_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class ResNet34UNetAdvancedBase(nn.Module):
    """
    自定义 ResNet34-U-Net，方便把模块放到正确位置：
    - skip_cbsa: CBSA 放在 skip connection 上
    - bottleneck_laspp: Lite-ASPP 放在 encoder bottleneck 上
    - boundary_aux: 额外输出边界辅助头
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        skip_cbsa=False,
        bottleneck_laspp=False,
        boundary_aux=False,
        cbsa_lambda=1e-4,
        cbsa_alpha=1.0,
        cbsa_beta=0.1,
        laspp_mid_channels=128,
        laspp_dropout=0.1,
    ):
        super().__init__()

        self.skip_cbsa = skip_cbsa
        self.bottleneck_laspp = bottleneck_laspp
        self.boundary_aux = boundary_aux

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )

        ch = list(self.encoder.out_channels)
        # SMP ResNet34 encoder channels usually: [3, 64, 64, 128, 256, 512]
        c1, c2, c3, c4, c5 = ch[1], ch[2], ch[3], ch[4], ch[5]

        if bottleneck_laspp:
            self.blaspp = LiteASPPBlock(
                in_channels=c5,
                out_channels=c5,
                mid_channels=laspp_mid_channels,
                dropout=laspp_dropout,
            )
        else:
            self.blaspp = nn.Identity()

        if skip_cbsa:
            self.cbsa1 = CBSABlock(cbsa_lambda, cbsa_alpha, cbsa_beta)
            self.cbsa2 = CBSABlock(cbsa_lambda, cbsa_alpha, cbsa_beta)
            self.cbsa3 = CBSABlock(cbsa_lambda, cbsa_alpha, cbsa_beta)
            self.cbsa4 = CBSABlock(cbsa_lambda, cbsa_alpha, cbsa_beta)
        else:
            self.cbsa1 = nn.Identity()
            self.cbsa2 = nn.Identity()
            self.cbsa3 = nn.Identity()
            self.cbsa4 = nn.Identity()

        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(c1, c1, kernel_size=2, stride=2),
            ConvBlock(c1, c1),
        )

        self.out_mask = nn.Conv2d(c1, num_classes, kernel_size=1)

        if boundary_aux:
            self.out_edge = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x, return_edge=False):
        features = self.encoder(x)
        f0 = features[0]
        f1, f2, f3, f4, f5 = features[1], features[2], features[3], features[4], features[5]

        f5 = self.blaspp(f5)

        s1 = self.cbsa1(f1)
        s2 = self.cbsa2(f2)
        s3 = self.cbsa3(f3)
        s4 = self.cbsa4(f4)

        x = self.up4(f5, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)
        x = self.final_up(x)

        if x.shape[-2:] != f0.shape[-2:]:
            x = F.interpolate(x, size=f0.shape[-2:], mode="bilinear", align_corners=False)

        mask_logits = self.out_mask(x)

        if self.boundary_aux and return_edge:
            edge_logits = self.out_edge(x)
            return mask_logits, edge_logits

        return mask_logits


class ResNet34UNetSkipCBSA(ResNet34UNetAdvancedBase):
    def __init__(self, **kwargs):
        super().__init__(skip_cbsa=True, bottleneck_laspp=False, boundary_aux=False, **kwargs)


class ResNet34UNetBottleneckLiteASPP(ResNet34UNetAdvancedBase):
    def __init__(self, **kwargs):
        super().__init__(skip_cbsa=False, bottleneck_laspp=True, boundary_aux=False, **kwargs)


class ResNet34UNetSkipCBSABottleneckLiteASPP(ResNet34UNetAdvancedBase):
    def __init__(self, **kwargs):
        super().__init__(skip_cbsa=True, bottleneck_laspp=True, boundary_aux=False, **kwargs)


class ResNet34UNetSkipCBSABottleneckLiteASPPBoundary(ResNet34UNetAdvancedBase):
    def __init__(self, **kwargs):
        super().__init__(skip_cbsa=True, bottleneck_laspp=True, boundary_aux=True, **kwargs)


def dice_loss_with_logits(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * inter + eps) / (union + eps)
    return 1.0 - dice.mean()


def seg_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dloss = dice_loss_with_logits(logits, targets)
    return bce + dloss


def mask_to_boundary(mask, kernel_size=3):
    """
    从 mask 自动生成边界监督图：
    boundary = dilation(mask) - erosion(mask)
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)

    pad = kernel_size // 2
    dilated = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded).clamp(0, 1)
    return boundary


def train_one_epoch_boundary(model, loader, optimizer, device, edge_weight=0.2):
    model.train()

    total_loss = 0.0
    total_seg_loss = 0.0
    total_edge_loss = 0.0

    tp = fp = fn = 0.0
    n_batches = 0

    for batch in loader:
        images, masks = batch[0], batch[1]
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)

        edges = mask_to_boundary(masks)

        optimizer.zero_grad(set_to_none=True)

        mask_logits, edge_logits = model(images, return_edge=True)

        loss_seg = seg_loss(mask_logits, masks)
        loss_edge = F.binary_cross_entropy_with_logits(edge_logits, edges)
        loss = loss_seg + edge_weight * loss_edge

        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        total_seg_loss += float(loss_seg.detach().cpu())
        total_edge_loss += float(loss_edge.detach().cpu())
        n_batches += 1

        with torch.no_grad():
            pred = (torch.sigmoid(mask_logits) > 0.5).float()
            tp += float((pred * masks).sum().detach().cpu())
            fp += float((pred * (1 - masks)).sum().detach().cpu())
            fn += float(((1 - pred) * masks).sum().detach().cpu())

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-7)
    iou = tp / (tp + fp + fn + 1e-7)

    return {
        "loss": total_loss / max(n_batches, 1),
        "seg_loss": total_seg_loss / max(n_batches, 1),
        "edge_loss": total_edge_loss / max(n_batches, 1),
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
    }


def build_model(args):
    common_kwargs = dict(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cbsa_lambda=args.cbsa_lambda,
        cbsa_alpha=args.cbsa_alpha,
        cbsa_beta=args.cbsa_beta,
        laspp_mid_channels=args.laspp_mid_channels,
        laspp_dropout=args.laspp_dropout,
    )

    if args.variant == "m8_skip_cbsa":
        return ResNet34UNetSkipCBSA(**common_kwargs)

    if args.variant == "m9_bottleneck_laspp":
        return ResNet34UNetBottleneckLiteASPP(**common_kwargs)

    if args.variant == "m10_skip_cbsa_bottleneck_laspp":
        return ResNet34UNetSkipCBSABottleneckLiteASPP(**common_kwargs)

    if args.variant == "m11_skip_cbsa_bottleneck_laspp_boundary":
        return ResNet34UNetSkipCBSABottleneckLiteASPPBoundary(**common_kwargs)

    raise ValueError(f"Unknown variant: {args.variant}")


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
    config["model_class"] = {
        "m8_skip_cbsa": "ResNet34UNetSkipCBSA",
        "m9_bottleneck_laspp": "ResNet34UNetBottleneckLiteASPP",
        "m10_skip_cbsa_bottleneck_laspp": "ResNet34UNetSkipCBSABottleneckLiteASPP",
        "m11_skip_cbsa_bottleneck_laspp_boundary": "ResNet34UNetSkipCBSABottleneckLiteASPPBoundary",
    }[args.variant]
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

    model = build_model(args).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    print("=" * 80)
    print("Model ready")
    print("variant       =", args.variant)
    print("run_name      =", args.run_name)
    print("batch_size    =", args.batch_size)
    print("epochs        =", args.epochs)
    print("lr            =", args.lr)
    print("edge_weight   =", args.edge_weight)
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

    is_boundary = args.variant == "m11_skip_cbsa_bottleneck_laspp_boundary"

    for epoch in range(1, args.epochs + 1):
        if is_boundary:
            train_metrics = train_one_epoch_boundary(
                model,
                train_loader,
                optimizer,
                device,
                edge_weight=args.edge_weight,
            )
        else:
            train_metrics = run_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                deep_supervision=False,
                train=True,
            )

        val_metrics = run_one_epoch(
            model,
            val_loader,
            optimizer=None,
            device=device,
            deep_supervision=False,
            train=False,
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

    best_train_metrics = run_one_epoch(
        model,
        train_loader,
        optimizer=None,
        device=device,
        deep_supervision=False,
        train=False,
    )
    best_val_metrics = run_one_epoch(
        model,
        val_loader,
        optimizer=None,
        device=device,
        deep_supervision=False,
        train=False,
    )
    test_metrics = run_one_epoch(
        model,
        test_loader,
        optimizer=None,
        device=device,
        deep_supervision=False,
        train=False,
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

    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--run-name", type=str, required=True)

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

    parser.add_argument("--laspp-mid-channels", type=int, default=128)
    parser.add_argument("--laspp-dropout", type=float, default=0.1)

    parser.add_argument("--edge-weight", type=float, default=0.2)

    args = parser.parse_args()
    main(args)
