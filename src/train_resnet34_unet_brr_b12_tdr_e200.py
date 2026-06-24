import argparse
import csv
import math
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


def _scale_to_logit(scale: float, max_delta: float) -> float:
    ratio = max(min(float(scale) / max(float(max_delta), 1e-6), 0.999), 1e-4)
    return math.log(ratio / (1.0 - ratio))


class ConvGNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class TextureDetailRescueHead(nn.Module):
    """
    Texture Detail Rescue (TDR) head.

    The branch is intentionally asymmetric: it predicts a non-negative logit
    residual that can only add local evidence to the B2 mask logits. This is
    meant to counter the current experiment pattern where added gates/attention
    reduce false positives but increase missed caries pixels.

    Inputs:
      - raw dental X-ray image high-frequency residual: image - local average
      - pre-RBSG decoder feature: semantic context before rim suppression

    Output:
      - delta_logits >= 0, bounded by a learnable small scale
    """
    def __init__(
        self,
        decoder_channels,
        in_channels=3,
        hidden_channels=16,
        max_delta=0.75,
        init_scale=0.15,
        bias_init=-3.0,
        highpass_kernel=7,
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.highpass_kernel = int(highpass_kernel)

        self.texture_stem = nn.Sequential(
            ConvGNAct(in_channels, hidden_channels, kernel_size=3),
            ConvGNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                groups=hidden_channels,
            ),
        )
        self.context_stem = nn.Sequential(
            ConvGNAct(decoder_channels, hidden_channels, kernel_size=3),
            ConvGNAct(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                groups=hidden_channels,
            ),
        )
        self.fuse = nn.Sequential(
            ConvGNAct(hidden_channels * 2, hidden_channels, kernel_size=3),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

        final_conv = self.fuse[-1]
        nn.init.zeros_(final_conv.weight)
        nn.init.constant_(final_conv.bias, float(bias_init))

        self.scale_logit = nn.Parameter(
            torch.tensor(_scale_to_logit(init_scale, self.max_delta), dtype=torch.float32)
        )

    def high_frequency(self, x):
        k = self.highpass_kernel
        if k <= 1:
            return x
        pad = k // 2
        local_mean = F.avg_pool2d(
            x,
            kernel_size=k,
            stride=1,
            padding=pad,
            count_include_pad=False,
        )
        return x - local_mean

    def forward(self, image, decoder_feature):
        texture = self.texture_stem(self.high_frequency(image))
        context = self.context_stem(decoder_feature)
        raw_delta = self.fuse(torch.cat([texture, context], dim=1))

        scale = self.max_delta * torch.sigmoid(self.scale_logit)
        return scale * torch.sigmoid(raw_delta)


class ResNet34UNetBRRB12TDR(ResNet34UNetBRRB2RBSG):
    """
    B12: B2 + Texture Detail Rescue.

    B2 is kept intact:
      ResNet34-U-Net + boundary/rim auxiliary supervision + final RBSG-lite.

    B12 adds one non-boundary, non-rim branch:
      final_mask_logits = B2_mask_logits + non_negative_texture_detail_delta

    The added branch is deliberately recall-friendly. It is allowed to raise
    uncertain local lesion evidence, but it is not allowed to suppress B2 mask
    logits. This makes it different from previous gate/attention experiments
    that tended to increase precision while dropping recall.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        tdr_hidden_channels=16,
        tdr_max_delta=0.75,
        tdr_init_scale=0.15,
        tdr_bias_init=-3.0,
        tdr_highpass_kernel=7,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )

        seg_conv = self.unet.segmentation_head[0]
        dec_ch = seg_conv.in_channels
        self.tdr_head = TextureDetailRescueHead(
            decoder_channels=dec_ch,
            in_channels=in_channels,
            hidden_channels=tdr_hidden_channels,
            max_delta=tdr_max_delta,
            init_scale=tdr_init_scale,
            bias_init=tdr_bias_init,
            highpass_kernel=tdr_highpass_kernel,
        )

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)
        b2_mask_logits = self.mask_head(refined)
        rescue_delta = self.tdr_head(x, decoder_output)
        mask_logits = b2_mask_logits + rescue_delta

        if not return_aux:
            return mask_logits

        boundary_logits = self.boundary_head(refined)
        rim_logits = self.rim_head(refined)

        return {
            "mask": mask_logits,
            "boundary": boundary_logits,
            "rim": rim_logits,
        }

    @property
    def learned_tdr_scale(self):
        return float((self.tdr_head.max_delta * torch.sigmoid(self.tdr_head.scale_logit)).detach().cpu())


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
    config["model_class"] = "ResNet34UNetBRRB12TDR"
    config["experiment"] = "B12: B2 + Texture Detail Rescue non-negative logit residual"
    config["note"] = (
        "B12 keeps the complete B2 boundary/rim auxiliary supervision and final "
        "RBSG-lite. It adds one recall-friendly high-frequency texture/detail "
        "branch from the raw X-ray and pre-RBSG decoder feature. The branch "
        "predicts a non-negative bounded logit residual, so it can rescue weak "
        "small-lesion evidence but cannot suppress B2 mask logits."
    )
    config["design_references"] = [
        {
            "title": "nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation",
            "venue": "arXiv 2024 / nnU-Net v2 validation direction",
            "adopted_idea": "keep the strong baseline recipe stable and change one interpretable component at a time",
        },
        {
            "title": "PRAD: Periapical Radiograph Analysis Dataset and Benchmark Model Development",
            "venue": "arXiv 2025 periapical radiograph benchmark direction",
            "adopted_idea": "periapical radiographs contain detailed local lesions but suffer from resolution limits and artifacts",
        },
        {
            "title": "Instructive Feature Enhancement for Dichotomous Medical Image Segmentation",
            "venue": "arXiv / MICCAI-related medical segmentation direction",
            "adopted_idea": "texture/high-information channels can provide complementary cues for low-contrast medical segmentation",
        },
        {
            "title": "Spectral U-Net for medical image segmentation",
            "venue": "recent frequency-domain segmentation direction",
            "adopted_idea": "high-frequency/detail cues help recover fine local structures missed by ordinary encoder-decoder paths",
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
    config["tdr"] = {
        "position": "parallel non-negative residual added to final B2 mask logits",
        "inputs": [
            "raw X-ray high-frequency residual: image - avg_pool(image)",
            "pre-RBSG final decoder feature",
        ],
        "hidden_channels": args.tdr_hidden_channels,
        "max_delta_logit": args.tdr_max_delta,
        "initial_learned_scale": args.tdr_init_scale,
        "head_bias_init": args.tdr_bias_init,
        "highpass_kernel": args.tdr_highpass_kernel,
        "uses_boundary_or_rim": False,
        "extra_supervision": False,
        "non_negative_residual": True,
        "rationale": "prior B2 extensions mostly reduced FP but increased FN; this branch is designed to recover weak small-lesion evidence without suppressing existing mask logits",
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
            "vertical_flip_p": 0.0,
        },
        "photometric": {
            "brightness_factor": [0.85, 1.15],
            "contrast_factor": [0.85, 1.15],
            "gamma": [0.80, 1.25],
            "gaussian_noise_p": 0.10,
            "noise_std": [0.005, 0.020],
            "gaussian_blur_p": 0.10,
            "blur_radius": [0.2, 0.8],
        },
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

    model = ResNet34UNetBRRB12TDR(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        tdr_hidden_channels=args.tdr_hidden_channels,
        tdr_max_delta=args.tdr_max_delta,
        tdr_init_scale=args.tdr_init_scale,
        tdr_bias_init=args.tdr_bias_init,
        tdr_highpass_kernel=args.tdr_highpass_kernel,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B12 = B2 BRR-RBSG + Texture Detail Rescue")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("tdr_scale_init  =", args.tdr_init_scale)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
            "tdr_scale",
        ])

    best_val_dice = -1.0
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b1(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            epoch=epoch,
        )
        val_metrics = run_one_epoch_b1(
            model,
            val_loader,
            optimizer=None,
            device=device,
            train=False,
            epoch=epoch,
        )

        tdr_scale = model.learned_tdr_scale
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} tdr_scale={tdr_scale:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                tdr_scale,
            ])

        torch.save(model.state_dict(), ckpt_dir / "last.pth")

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_dir / "best.pth")
            print(f"  -> best model updated, epoch={best_epoch}, val_dice={best_val_dice:.4f}")

    model.load_state_dict(torch.load(ckpt_dir / "best.pth", map_location=device))

    best_train_metrics = run_one_epoch_b1(
        model,
        train_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
    )
    best_val_metrics = run_one_epoch_b1(
        model,
        val_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
    )
    test_metrics = run_one_epoch_b1(
        model,
        test_loader,
        optimizer=None,
        device=device,
        train=False,
        epoch=args.epochs,
    )

    summary = {
        "best_epoch": best_epoch,
        "best_val_dice_during_training": best_val_dice,
        "best_train_metrics": best_train_metrics,
        "best_val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "tdr_parameters": sum(p.numel() for p in model.tdr_head.parameters()),
        "learned_tdr_scale": model.learned_tdr_scale,
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

    parser.add_argument("--run-name", type=str, default="b12_tdr_rescue_auglite_e200_constlr_bs6")
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

    parser.add_argument("--tdr-hidden-channels", type=int, default=16)
    parser.add_argument("--tdr-max-delta", type=float, default=0.75)
    parser.add_argument("--tdr-init-scale", type=float, default=0.15)
    parser.add_argument("--tdr-bias-init", type=float, default=-3.0)
    parser.add_argument("--tdr-highpass-kernel", type=int, default=7)

    main(parser.parse_args())
