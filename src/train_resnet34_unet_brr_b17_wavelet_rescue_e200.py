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
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.GroupNorm(_valid_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class WaveletFrequencyPromptRescue(nn.Module):
    """
    Wavelet Frequency Prompt Rescue (WFPR).

    Dental caries in periapical radiographs are often low-contrast and small.
    Fixed Haar wavelet high-frequency bands provide a cheap, deterministic
    texture prompt. The branch is kept conservative by predicting only a
    bounded non-negative logit residual around low-threshold B2 candidates.
    """
    def __init__(
        self,
        decoder_channels,
        hidden_channels=16,
        max_delta=0.55,
        init_scale=0.08,
        bias_init=-4.0,
        candidate_tau=0.12,
        candidate_sharpness=12.0,
    ):
        super().__init__()
        self.max_delta = float(max_delta)
        self.candidate_tau = float(candidate_tau)
        self.candidate_sharpness = float(candidate_sharpness)

        filters = torch.tensor(
            [
                [[1.0, 1.0], [-1.0, -1.0]],   # LH
                [[1.0, -1.0], [1.0, -1.0]],   # HL
                [[1.0, -1.0], [-1.0, 1.0]],   # HH
            ],
            dtype=torch.float32,
        ).unsqueeze(1) * 0.5
        self.register_buffer("haar_detail_filters", filters, persistent=False)

        self.wavelet_stem = nn.Sequential(
            ConvGNAct(4, hidden_channels, kernel_size=3),
            ConvGNAct(hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels),
        )
        self.context_stem = nn.Sequential(
            ConvGNAct(decoder_channels, hidden_channels, kernel_size=3),
            ConvGNAct(hidden_channels, hidden_channels, kernel_size=3, groups=hidden_channels),
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

    def normalize_map(self, x):
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        return (x - x_min) / (x_max - x_min + 1e-6)

    def wavelet_prompt(self, image):
        gray = image.mean(dim=1, keepdim=True)
        detail = F.conv2d(gray, self.haar_detail_filters, stride=2)
        detail = torch.abs(detail)
        detail = F.interpolate(
            detail,
            size=gray.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        detail = self.normalize_map(detail)
        return torch.cat([gray, detail], dim=1)

    def candidate_support(self, coarse_logits):
        prob = torch.sigmoid(coarse_logits.detach())
        candidate = torch.sigmoid(
            (prob - self.candidate_tau) * self.candidate_sharpness
        )
        return 0.20 + 0.80 * candidate

    def forward(self, image, decoder_feature, coarse_logits):
        wavelet_feature = self.wavelet_stem(self.wavelet_prompt(image))
        context_feature = self.context_stem(decoder_feature)
        raw_delta = self.fuse(torch.cat([wavelet_feature, context_feature], dim=1))
        scale = self.max_delta * torch.sigmoid(self.scale_logit)
        return scale * torch.sigmoid(raw_delta) * self.candidate_support(coarse_logits)


class ResNet34UNetBRRB17WaveletRescue(ResNet34UNetBRRB2RBSG):
    """
    B17: B2 + Wavelet Frequency Prompt Rescue.

    It is a frequency-domain complement to B15's Sobel/local-contrast rescue.
    The branch uses no boundary/rim supervision and cannot suppress B2 logits.
    """
    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        wfpr_hidden_channels=16,
        wfpr_max_delta=0.55,
        wfpr_init_scale=0.08,
        wfpr_bias_init=-4.0,
        wfpr_candidate_tau=0.12,
        wfpr_candidate_sharpness=12.0,
    ):
        super().__init__(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            num_classes=num_classes,
        )
        seg_conv = self.unet.segmentation_head[0]
        dec_ch = seg_conv.in_channels
        self.wfpr = WaveletFrequencyPromptRescue(
            decoder_channels=dec_ch,
            hidden_channels=wfpr_hidden_channels,
            max_delta=wfpr_max_delta,
            init_scale=wfpr_init_scale,
            bias_init=wfpr_bias_init,
            candidate_tau=wfpr_candidate_tau,
            candidate_sharpness=wfpr_candidate_sharpness,
        )

    def forward(self, x, return_aux=False):
        features = self.unet.encoder(x)
        decoder_output = self.unet.decoder(*features)

        b_pre = self.boundary_head_pre(decoder_output)
        r_pre = self.rim_head_pre(decoder_output)
        refined = self.rbsg(decoder_output, b_pre, r_pre)

        b2_mask_logits = self.mask_head(refined)
        mask_logits = b2_mask_logits + self.wfpr(x, decoder_output, b2_mask_logits)

        if not return_aux:
            return mask_logits

        return {
            "mask": mask_logits,
            "boundary": self.boundary_head(refined),
            "rim": self.rim_head(refined),
        }

    @property
    def learned_wfpr_scale(self):
        return float((self.wfpr.max_delta * torch.sigmoid(self.wfpr.scale_logit)).detach().cpu())


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
    config["model_class"] = "ResNet34UNetBRRB17WaveletRescue"
    config["experiment"] = "B17: B2 + Wavelet Frequency Prompt Rescue"
    config["note"] = (
        "B17 keeps B2 unchanged and adds a fixed Haar-wavelet high-frequency "
        "prompt branch. The branch predicts a bounded non-negative logit residual "
        "around low-threshold B2 candidates, targeting weak low-contrast caries "
        "pixels without adding more boundary/rim suppression."
    )
    config["dataset_observation"] = {
        "test_cases": 358,
        "small_group_cases": 119,
        "small_area_ratio_mean_512": 0.011539875960149685,
        "implication": "sparse low-contrast lesions may benefit from deterministic frequency prompts more than heavy global attention",
    }
    config["design_references"] = [
        {
            "title": "FreqU-FNet: Frequency-Aware U-Net for Imbalanced Medical Image Segmentation",
            "year": 2025,
            "adopted_idea": "frequency-aware prompts can help minority foreground segmentation",
        },
        {
            "title": "Spectral U-Net for medical image segmentation",
            "year": 2024,
            "adopted_idea": "frequency/detail information can compensate for information loss in ordinary encoder-decoder paths",
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
    config["wfpr"] = {
        "position": "parallel non-negative residual added to final B2 mask logits",
        "frequency_prompt": "fixed Haar LH/HL/HH detail bands from grayscale X-ray",
        "hidden_channels": args.wfpr_hidden_channels,
        "max_delta_logit": args.wfpr_max_delta,
        "initial_learned_scale": args.wfpr_init_scale,
        "candidate_tau": args.wfpr_candidate_tau,
        "candidate_sharpness": args.wfpr_candidate_sharpness,
        "uses_boundary_or_rim": False,
        "extra_supervision": False,
        "non_negative_residual": True,
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
    model = ResNet34UNetBRRB17WaveletRescue(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        wfpr_hidden_channels=args.wfpr_hidden_channels,
        wfpr_max_delta=args.wfpr_max_delta,
        wfpr_init_scale=args.wfpr_init_scale,
        wfpr_bias_init=args.wfpr_bias_init,
        wfpr_candidate_tau=args.wfpr_candidate_tau,
        wfpr_candidate_sharpness=args.wfpr_candidate_sharpness,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("=" * 80)
    print("Model ready: B17 = B2 BRR-RBSG + Wavelet Frequency Prompt Rescue")
    print("run_name        =", args.run_name)
    print("device          =", device)
    print("batch_size      =", args.batch_size)
    print("epochs          =", args.epochs)
    print("wfpr_scale_init =", args.wfpr_init_scale)
    print("=" * 80)

    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss", "train_dice", "train_iou", "train_precision", "train_recall",
            "val_loss", "val_dice", "val_iou", "val_precision", "val_recall",
            "wfpr_scale",
        ])

    best_val_dice = -1.0
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch_b1(model, train_loader, optimizer, device, train=True, epoch=epoch)
        val_metrics = run_one_epoch_b1(model, val_loader, optimizer=None, device=device, train=False, epoch=epoch)
        wfpr_scale = model.learned_wfpr_scale

        print(
            f"[Epoch {epoch:03d}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_dice={train_metrics['dice']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_dice={val_metrics['dice']:.4f} "
            f"val_iou={val_metrics['iou']:.4f} val_precision={val_metrics['precision']:.4f} "
            f"val_recall={val_metrics['recall']:.4f} wfpr_scale={wfpr_scale:.4f}"
        )
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                train_metrics["loss"], train_metrics["dice"], train_metrics["iou"],
                train_metrics["precision"], train_metrics["recall"],
                val_metrics["loss"], val_metrics["dice"], val_metrics["iou"],
                val_metrics["precision"], val_metrics["recall"],
                wfpr_scale,
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
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "wfpr_parameters": sum(p.numel() for p in model.wfpr.parameters()),
        "learned_wfpr_scale": model.learned_wfpr_scale,
    }
    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(summary, run_dir / "summary_metrics.json")
    save_prediction_panel(model, test_loader, device, pred_dir / "test_preview.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", type=str, default="b17_wavelet_rescue_auglite_e200_constlr_bs6")
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
    parser.add_argument("--wfpr-hidden-channels", type=int, default=16)
    parser.add_argument("--wfpr-max-delta", type=float, default=0.55)
    parser.add_argument("--wfpr-init-scale", type=float, default=0.08)
    parser.add_argument("--wfpr-bias-init", type=float, default=-4.0)
    parser.add_argument("--wfpr-candidate-tau", type=float, default=0.12)
    parser.add_argument("--wfpr-candidate-sharpness", type=float, default=12.0)
    main(parser.parse_args())
