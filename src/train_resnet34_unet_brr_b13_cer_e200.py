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


def _valid_group_count(channels: int, max_groups: int = 4) -> int:
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


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
            nn.GroupNorm(
                num_groups=_valid_group_count(out_channels),
                num_channels=out_channels,
            ),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CandidateExpansionRefiner(nn.Module):
    """
    Candidate Expansion Refiner (CER).

    This is a lightweight, DoubleU-Net-inspired second-stage refiner, but it is
    deliberately smaller than a full second U-Net. B2 produces a coarse mask;
    CER re-reads:
      1) the raw X-ray,
      2) local high-frequency texture,
      3) B2 coarse probability,
      4) B2 uncertainty,
      5) a low-threshold candidate-expansion map,
      6) pre-RBSG decoder semantics.

    CER predicts a bounded non-negative logit residual. It can recover weak
    lesion pixels around low-confidence candidates, but it cannot suppress the
    original B2 mask logits. That matches the current empirical failure mode:
    B10/B11 mostly lost recall rather than suffering from excess false positives.
    """
    def __init__(
        self,
        decoder_channels,
        in_channels=3,
        hidden_channels=24,
        max_delta=0.80,
        init_scale=0.12,
        bias_init=-4.0,
        highpass_kernel=7,
        candidate_tau=0.15,
        candidate_sharpness=12.0,
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.highpass_kernel = int(highpass_kernel)
        self.candidate_tau = float(candidate_tau)
        self.candidate_sharpness = float(candidate_sharpness)

        spatial_in_channels = in_channels * 2 + 3
        self.spatial_stem = nn.Sequential(
            ConvGNAct(spatial_in_channels, hidden_channels, kernel_size=3),
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

    def candidate_maps(self, coarse_logits):
        prob = torch.sigmoid(coarse_logits.detach())
        uncertainty = 1.0 - torch.abs(2.0 * prob - 1.0)
        candidate = torch.sigmoid(
            (prob - self.candidate_tau) * self.candidate_sharpness
        )
        return prob, uncertainty, candidate

    def forward(self, image, decoder_feature, coarse_logits):
        high_freq = self.high_frequency(image)
        prob, uncertainty, candidate = self.candidate_maps(coarse_logits)

        spatial = self.spatial_stem(
            torch.cat([image, high_freq, prob, uncertainty, candidate], dim=1)
        )
        context = self.context_stem(decoder_feature)
        raw_delta = self.fuse(torch.cat([spatial, context], dim=1))

        scale = self.max_delta * torch.sigmoid(self.scale_logit)
        # Keep a soft floor so a completely missed but texture-supported lesion
        # still has a route to be rescued, while candidate regions get more gain.
        soft_candidate_gain = 0.25 + 0.75 * candidate
        return scale * torch.sigmoid(raw_delta) * soft_candidate_gain


class ResNet34UNetBRRB13CER(ResNet34UNetBRRB2RBSG):
    """
    B13: B2 + Candidate Expansion Refiner.

    B2 is unchanged:
      ResNet34-U-Net + boundary/rim auxiliary supervision + RBSG-lite.

    B13 adds a small second-stage CNN refiner:
      final_mask_logits = B2_mask_logits + CER_delta

    The refiner is lesion-candidate aware through B2 probability/uncertainty,
    but it is not a boundary/rim module and it does not introduce extra losses.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        cer_hidden_channels=24,
        cer_max_delta=0.80,
        cer_init_scale=0.12,
        cer_bias_init=-4.0,
        cer_highpass_kernel=7,
        cer_candidate_tau=0.15,
        cer_candidate_sharpness=12.0,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )

        seg_conv = self.unet.segmentation_head[0]
        dec_ch = seg_conv.in_channels
        self.cer = CandidateExpansionRefiner(
            decoder_channels=dec_ch,
            in_channels=in_channels,
            hidden_channels=cer_hidden_channels,
            max_delta=cer_max_delta,
            init_scale=cer_init_scale,
            bias_init=cer_bias_init,
            highpass_kernel=cer_highpass_kernel,
            candidate_tau=cer_candidate_tau,
            candidate_sharpness=cer_candidate_sharpness,
        )

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)

        refined = self.rbsg(decoder_output, b_pre, r_pre)
        b2_mask_logits = self.mask_head(refined)
        cer_delta = self.cer(x, decoder_output, b2_mask_logits)
        mask_logits = b2_mask_logits + cer_delta

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
    def learned_cer_scale(self):
        return float((self.cer.max_delta * torch.sigmoid(self.cer.scale_logit)).detach().cpu())


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
    config["model_class"] = "ResNet34UNetBRRB13CER"
    config["experiment"] = "B13: B2 + Candidate Expansion Refiner"
    config["note"] = (
        "B13 keeps B2 unchanged and adds a lightweight DoubleU-Net-inspired "
        "second-stage refiner. The refiner uses B2 coarse probability, "
        "uncertainty, low-threshold candidate expansion, raw X-ray texture, "
        "and pre-RBSG decoder semantics to predict a bounded non-negative "
        "logit residual. It targets missed low-confidence caries pixels rather "
        "than further suppressing hard negatives."
    )
    config["design_references"] = [
        {
            "title": "When CNNs Outperform Transformers and Mambas: Revisiting Deep Architectures for Dental Caries Segmentation",
            "year": 2025,
            "adopted_idea": "dental caries segmentation can favor CNN/two-stage spatial priors over heavier Transformer/Mamba modules under limited data",
        },
        {
            "title": "DoubleU-Net: A Deep Convolutional Neural Network for Medical Image Segmentation",
            "year": 2020,
            "adopted_idea": "stacked coarse-to-refined U-shaped segmentation can improve small/flat lesion masks",
        },
        {
            "title": "PRAD: Periapical Radiograph Analysis Dataset and Benchmark Model Development",
            "year": 2025,
            "adopted_idea": "periapical radiographs preserve detailed local lesions but are affected by resolution limits and artifacts",
        },
        {
            "title": "FreqU-FNet: Frequency-Aware U-Net for Imbalanced Medical Image Segmentation",
            "year": 2025,
            "adopted_idea": "frequency/detail cues are useful for minority-class medical segmentation",
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
    config["cer"] = {
        "position": "parallel second-stage residual added to final B2 mask logits",
        "inputs": [
            "raw X-ray image",
            "raw X-ray high-frequency residual",
            "B2 coarse probability detached from mask logits",
            "B2 uncertainty map",
            "low-threshold candidate expansion map",
            "pre-RBSG final decoder feature",
        ],
        "hidden_channels": args.cer_hidden_channels,
        "max_delta_logit": args.cer_max_delta,
        "initial_learned_scale": args.cer_init_scale,
        "head_bias_init": args.cer_bias_init,
        "highpass_kernel": args.cer_highpass_kernel,
        "candidate_tau": args.cer_candidate_tau,
        "candidate_sharpness": args.cer_candidate_sharpness,
        "uses_boundary_or_rim": False,
        "extra_supervision": False,
        "non_negative_residual": True,
        "rationale": "B10/B11 reduced recall; CER tests whether a lesion-candidate-aware second-stage positive residual can recover low-confidence caries pixels",
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

    model = ResNet34UNetBRRB13CER(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        cer_hidden_channels=args.cer_hidden_channels,
        cer_max_delta=args.cer_max_delta,
        cer_init_scale=args.cer_init_scale,
        cer_bias_init=args.cer_bias_init,
        cer_highpass_kernel=args.cer_highpass_kernel,
        cer_candidate_tau=args.cer_candidate_tau,
        cer_candidate_sharpness=args.cer_candidate_sharpness,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B13 = B2 BRR-RBSG + Candidate Expansion Refiner")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("augment_profile =", args.augment_profile)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("lr              =", args.lr)
    print("cer_scale_init  =", args.cer_init_scale)
    print("cer_tau         =", args.cer_candidate_tau)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
            "cer_scale",
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

        cer_scale = model.learned_cer_scale
        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} cer_scale={cer_scale:.4f}"
        )

        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                cer_scale,
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
        "cer_parameters": sum(p.numel() for p in model.cer.parameters()),
        "learned_cer_scale": model.learned_cer_scale,
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

    parser.add_argument("--run-name", type=str, default="b13_cer_auglite_e200_constlr_bs6")
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

    parser.add_argument("--cer-hidden-channels", type=int, default=24)
    parser.add_argument("--cer-max-delta", type=float, default=0.80)
    parser.add_argument("--cer-init-scale", type=float, default=0.12)
    parser.add_argument("--cer-bias-init", type=float, default=-4.0)
    parser.add_argument("--cer-highpass-kernel", type=int, default=7)
    parser.add_argument("--cer-candidate-tau", type=float, default=0.15)
    parser.add_argument("--cer-candidate-sharpness", type=float, default=12.0)

    main(parser.parse_args())
