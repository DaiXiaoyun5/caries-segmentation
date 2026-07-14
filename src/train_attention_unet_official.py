#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Faithful 2D Attention U-Net baseline for CariXray.

This is a dimensional adaptation of Oktay et al.'s official medical U-Net,
not a ResNet encoder with generic attention attached.  It keeps the native
five-level U-Net, additive grid gates on the three semantic skip levels, an
ungated first low-level skip, and multi-scale deep supervision fusion.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from official_baseline_common import (
    binary_dice_loss,
    make_dataset_bundle,
    run_official_experiment,
    set_seed,
)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AdditiveGridAttentionGate2D(nn.Module):
    """2D form of the official concatenation-based grid attention block."""

    def __init__(
        self,
        skip_channels: int,
        gating_channels: int,
        inter_channels: int,
    ) -> None:
        super().__init__()
        self.theta = nn.Conv2d(
            skip_channels,
            inter_channels,
            kernel_size=2,
            stride=2,
            bias=False,
        )
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1)
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.output_transform = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
        )

    def forward(
        self,
        skip: torch.Tensor,
        gating: torch.Tensor,
    ) -> torch.Tensor:
        theta_x = self.theta(skip)
        phi_g = self.phi(gating)
        if phi_g.shape[-2:] != theta_x.shape[-2:]:
            phi_g = F.interpolate(
                phi_g,
                size=theta_x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        attention = torch.sigmoid(self.psi(F.relu(theta_x + phi_g, inplace=True)))
        attention = F.interpolate(
            attention,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.output_transform(skip * attention)


class AttentionUNetOfficial2D(nn.Module):
    """Native Attention U-Net with official feature_scale=4 channel widths."""

    FILTERS = (16, 32, 64, 128, 256)

    def __init__(self, in_channels: int = 1, num_classes: int = 1) -> None:
        super().__init__()
        f1, f2, f3, f4, f5 = self.FILTERS
        self.pool = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(in_channels, f1)
        self.conv2 = ConvBlock(f1, f2)
        self.conv3 = ConvBlock(f2, f3)
        self.conv4 = ConvBlock(f3, f4)
        self.center = ConvBlock(f4, f5)

        # The original paper explicitly leaves the first low-level skip ungated.
        self.att4 = AdditiveGridAttentionGate2D(f4, f5, f4)
        self.att3 = AdditiveGridAttentionGate2D(f3, f4, f3)
        self.att2 = AdditiveGridAttentionGate2D(f2, f3, f2)

        self.up4 = nn.ConvTranspose2d(f5, f4, 2, stride=2)
        self.dec4 = ConvBlock(f4 + f4, f4)
        self.up3 = nn.ConvTranspose2d(f4, f3, 2, stride=2)
        self.dec3 = ConvBlock(f3 + f3, f3)
        self.up2 = nn.ConvTranspose2d(f3, f2, 2, stride=2)
        self.dec2 = ConvBlock(f2 + f2, f2)
        self.up1 = nn.ConvTranspose2d(f2, f1, 2, stride=2)
        self.dec1 = ConvBlock(f1 + f1, f1)

        self.dsv4 = nn.Conv2d(f4, num_classes, 1)
        self.dsv3 = nn.Conv2d(f3, num_classes, 1)
        self.dsv2 = nn.Conv2d(f2, num_classes, 1)
        self.dsv1 = nn.Conv2d(f1, num_classes, 1)
        self.final = nn.Conv2d(num_classes * 4, num_classes, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # The paper initializes AGs close to pass-through rather than suppressing
        # all locations at the start of training.
        for gate in (self.att4, self.att3, self.att2):
            nn.init.zeros_(gate.psi.weight)
            nn.init.constant_(gate.psi.bias, 2.0)

    @staticmethod
    def _upsample_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == reference.shape[-2:]:
            return x
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        center = self.center(self.pool(conv4))

        gated4 = self.att4(conv4, center)
        up4 = self._upsample_like(self.up4(center), gated4)
        dec4 = self.dec4(torch.cat([gated4, up4], dim=1))

        gated3 = self.att3(conv3, dec4)
        up3 = self._upsample_like(self.up3(dec4), gated3)
        dec3 = self.dec3(torch.cat([gated3, up3], dim=1))

        gated2 = self.att2(conv2, dec3)
        up2 = self._upsample_like(self.up2(dec3), gated2)
        dec2 = self.dec2(torch.cat([gated2, up2], dim=1))

        up1 = self._upsample_like(self.up1(dec2), conv1)
        dec1 = self.dec1(torch.cat([conv1, up1], dim=1))

        output_size = x.shape[-2:]
        supervised = []
        for head, feature in (
            (self.dsv1, dec1),
            (self.dsv2, dec2),
            (self.dsv3, dec3),
            (self.dsv4, dec4),
        ):
            logits = head(feature)
            if logits.shape[-2:] != output_size:
                logits = F.interpolate(
                    logits,
                    size=output_size,
                    mode="bilinear",
                    align_corners=False,
                )
            supervised.append(logits)
        return self.final(torch.cat(supervised, dim=1))


def attention_dice_loss(output, masks):
    return binary_dice_loss(output, masks)


def main(args) -> None:
    set_seed(args.seed)
    datasets = make_dataset_bundle(
        image_size=args.image_size,
        channels=1,
        train_augmentation="attention_unet_official",
        normalization="per_image_zscore",
    )
    model = AttentionUNetOfficial2D(in_channels=1, num_classes=1)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_decay_epochs,
        gamma=args.lr_decay_gamma,
    )
    metadata = {
        "script": "src/train_attention_unet_official.py",
        "model_name": "Attention U-Net (official native U-Net, 2D adaptation)",
        "model_class": "AttentionUNetOfficial2D",
        "architecture": {
            "encoder": "native five-level convolutional U-Net, trained from scratch",
            "filters": list(AttentionUNetOfficial2D.FILTERS),
            "feature_scale": 4,
            "input_channels": 1,
            "attention": "additive grid attention on skip levels 2-4",
            "first_low_level_skip": "ungated, as stated in the paper",
            "deep_supervision": "four decoder scales fused by final 1x1 convolution",
            "resnet_encoder": False,
        },
        "training_recipe": {
            "source": "official Attention-Gated-Networks CT configuration and paper",
            "input_size": args.image_size,
            "normalization": "per-image zero mean / unit variance",
            "augmentation": "affine rotation/scale/translation and axial(horizontal) flip",
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "loss": "Sorensen-Dice loss",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "scheduler": {
                "type": "StepLR",
                "step_size_epochs": args.lr_decay_epochs,
                "gamma": args.lr_decay_gamma,
            },
            "checkpoint_selection": "minimum validation Dice loss",
        },
        "reference": {
            "paper": "https://arxiv.org/abs/1804.03999",
            "official_code": "https://github.com/ozan-oktay/Attention-Gated-Networks",
        },
        "fidelity_note": (
            "The source method is 3D CT. Only Conv3d/Pool3d/interpolation are "
            "dimensionally adapted to 2D; the native encoder, gate placement, "
            "deep supervision, feature scale, optimizer, loss, and augmentation "
            "principles are retained."
        ),
    }
    run_official_experiment(
        args,
        model,
        datasets,
        optimizer,
        attention_dice_loss,
        metadata,
        scheduler=scheduler,
        scheduler_step="epoch",
        val_every_epochs=args.val_every_epochs,
        selection_metric="loss",
        selection_mode="min",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="attention_unet_official_2d_e1000_bs2",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--lr-decay-epochs", type=int, default=250)
    parser.add_argument("--lr-decay-gamma", type=float, default=0.1)
    parser.add_argument("--val-every-epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
