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
from train_resnet34_unet_brr_b1_aux_e200 import run_one_epoch_b1
from train_resnet34_unet_brr_b2_rbsg_e200 import ResNet34UNetBRRB2RBSG


class ScaleModule(nn.Module):
    def __init__(self, channels, init_scale=1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1) * float(init_scale))

    def forward(self, x):
        return self.weight * x


class HaarWTConv2d(nn.Module):
    """
    Dependency-free WTConv-lite.

    The implementation follows the core idea of WTConv: apply depthwise
    convolution both on the original feature and on multi-level Haar wavelet
    sub-bands, then reconstruct. It avoids pywt/CUDA/Triton dependencies so the
    supercomputer training environment can run it as a plain PyTorch module.
    """
    def __init__(self, channels, kernel_size=5, wt_levels=2, wavelet_init_scale=0.10):
        super().__init__()
        self.channels = int(channels)
        self.wt_levels = int(wt_levels)

        self.register_buffer(
            "haar_filter",
            self._create_haar_filters(self.channels),
            persistent=False,
        )

        self.base_conv = nn.Conv2d(
            self.channels,
            self.channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=self.channels,
            bias=True,
        )
        self.base_scale = ScaleModule(self.channels, init_scale=1.0)

        self.wavelet_convs = nn.ModuleList([
            nn.Conv2d(
                self.channels * 4,
                self.channels * 4,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=self.channels * 4,
                bias=False,
            )
            for _ in range(self.wt_levels)
        ])
        self.wavelet_scales = nn.ModuleList([
            ScaleModule(self.channels * 4, init_scale=wavelet_init_scale)
            for _ in range(self.wt_levels)
        ])

    @staticmethod
    def _create_haar_filters(channels):
        filters = torch.tensor(
            [
                [[0.5, 0.5], [0.5, 0.5]],      # LL
                [[0.5, 0.5], [-0.5, -0.5]],    # LH
                [[0.5, -0.5], [0.5, -0.5]],    # HL
                [[0.5, -0.5], [-0.5, 0.5]],    # HH
            ],
            dtype=torch.float32,
        ).unsqueeze(1)
        return filters.repeat(int(channels), 1, 1, 1)

    def _wavelet_transform(self, x):
        b, c, h, w = x.shape
        y = F.conv2d(x, self.haar_filter, stride=2, groups=c)
        return y.reshape(b, c, 4, h // 2, w // 2)

    def _inverse_wavelet_transform(self, x):
        b, c, _, h, w = x.shape
        y = x.reshape(b, c * 4, h, w)
        return F.conv_transpose2d(y, self.haar_filter, stride=2, groups=c)

    def forward(self, x):
        ll_features = []
        high_features = []
        shapes = []

        current_ll = x
        for level in range(self.wt_levels):
            current_shape = current_ll.shape
            shapes.append(current_shape)

            if current_shape[2] % 2 or current_shape[3] % 2:
                current_ll = F.pad(
                    current_ll,
                    (0, current_shape[3] % 2, 0, current_shape[2] % 2),
                )

            current = self._wavelet_transform(current_ll)
            current_ll = current[:, :, 0]

            b, c, bands, h, w = current.shape
            current = current.reshape(b, c * bands, h, w)
            current = self.wavelet_scales[level](self.wavelet_convs[level](current))
            current = current.reshape(b, c, bands, h, w)

            ll_features.append(current[:, :, 0])
            high_features.append(current[:, :, 1:4])

        reconstructed_ll = 0
        for level in range(self.wt_levels - 1, -1, -1):
            current_ll = ll_features.pop() + reconstructed_ll
            current_high = high_features.pop()
            current_shape = shapes.pop()
            current = torch.cat([current_ll.unsqueeze(2), current_high], dim=2)
            reconstructed_ll = self._inverse_wavelet_transform(current)
            reconstructed_ll = reconstructed_ll[:, :, :current_shape[2], :current_shape[3]]

        base = self.base_scale(self.base_conv(x))
        return base + reconstructed_ll


class WTConvContextAdapter(nn.Module):
    """
    Wavelet large-receptive-field adapter for the deepest encoder feature.

    This is not a mask-logit residual and does not use boundary/rim maps. It
    enriches the bottleneck semantic feature before the original B2 decoder.
    """
    def __init__(
        self,
        channels,
        kernel_size=5,
        wt_levels=2,
        init_scale=0.05,
        wavelet_init_scale=0.10,
    ):
        super().__init__()
        self.wtconv = HaarWTConv2d(
            channels=channels,
            kernel_size=kernel_size,
            wt_levels=wt_levels,
            wavelet_init_scale=wavelet_init_scale,
        )
        self.mix = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.residual_scale = nn.Parameter(torch.tensor(float(init_scale), dtype=torch.float32))

    def forward(self, x):
        scale = torch.clamp(self.residual_scale, min=0.0, max=0.50)
        return x + scale * self.mix(self.wtconv(x))


class ResNet34UNetBRRB24WTConvContext(ResNet34UNetBRRB2RBSG):
    """
    B24: B2 + WTConv-lite bottleneck context adapter.

    It targets dental X-ray low-contrast context with a lightweight wavelet
    large-receptive-field module while preserving the complete B2 BRR/RBSG
    training and inference interface.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        wt_kernel_size=5,
        wt_levels=2,
        wt_init_scale=0.05,
        wt_wavelet_init_scale=0.10,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        encoder_channels = list(self.unet.encoder.out_channels)
        self.wt_context = WTConvContextAdapter(
            channels=int(encoder_channels[-1]),
            kernel_size=wt_kernel_size,
            wt_levels=wt_levels,
            init_scale=wt_init_scale,
            wavelet_init_scale=wt_wavelet_init_scale,
        )

    def forward(self, x, return_aux=False):
        features = list(self.unet.encoder(x))
        features[-1] = self.wt_context(features[-1])
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
        }

    @property
    def learned_wt_scale(self):
        return float(torch.clamp(self.wt_context.residual_scale, 0.0, 0.50).detach().cpu())


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
    config["model_class"] = "ResNet34UNetBRRB24WTConvContext"
    config["experiment"] = "B24: B2 + WTConv-lite bottleneck context adapter"
    config["note"] = (
        "B24 keeps B2 boundary/rim auxiliary supervision and RBSG-lite unchanged. "
        "It adds a dependency-free Haar WTConv-lite residual adapter on the "
        "deepest ResNet34 encoder feature to provide large receptive field context "
        "without changing the decoder, mask threshold, or inference interface."
    )
    config["design_reflection"] = {
        "why_not_more_boundary_rim": "previous pre-rim, consistency, and gate variants tended to increase conservativeness and reduce recall",
        "why_not_output_rescue": "B17 wavelet logit residual improved validation but did not beat B2 on test, suggesting direct mask residuals can overfit texture",
        "why_bottleneck_context": "context is injected before decoding so the original B2 decoder/RBSG can still decide how to use it",
    }
    config["design_references"] = [
        {
            "title": "Wavelet Convolutions for Large Receptive Fields",
            "venue": "ECCV 2024",
            "source": "WTConv-main.zip provided by user; simplified to plain PyTorch Haar filters",
        },
        {
            "title": "PRAD: A Benchmark Dataset for Periapical Radiograph Analysis and Diagnosis",
            "year": 2025,
            "adopted_idea": "same dental periapical radiograph domain motivates lightweight CNN-compatible context rather than heavy backbone replacement",
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
    config["wtconv_context"] = {
        "position": "deepest ResNet34 encoder feature before SMP decoder",
        "kernel_size": args.wt_kernel_size,
        "wt_levels": args.wt_levels,
        "init_scale": args.wt_init_scale,
        "wavelet_init_scale": args.wt_wavelet_init_scale,
        "dependency_free": True,
    }
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
    model = ResNet34UNetBRRB24WTConvContext(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        wt_kernel_size=args.wt_kernel_size,
        wt_levels=args.wt_levels,
        wt_init_scale=args.wt_init_scale,
        wt_wavelet_init_scale=args.wt_wavelet_init_scale,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B24 WTConv-context + B2 BRR RBSG-lite ResNet34-U-Net")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("wt_levels       =", args.wt_levels)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
            "learned_wt_scale",
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
            f"val_recall={val_metrics['recall']:.4f} wt_scale={model.learned_wt_scale:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                model.learned_wt_scale,
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
        "learned_wt_scale": model.learned_wt_scale,
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

    parser.add_argument("--run-name", type=str, default="b24_wtconv_context_auglite_e200_constlr_bs6")
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

    parser.add_argument("--wt-kernel-size", type=int, default=5)
    parser.add_argument("--wt-levels", type=int, default=2)
    parser.add_argument("--wt-init-scale", type=float, default=0.05)
    parser.add_argument("--wt-wavelet-init-scale", type=float, default=0.10)

    args = parser.parse_args()
    main(args)
