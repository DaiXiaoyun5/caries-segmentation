#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Controlled Attention U-Net comparison for CariXray.

The model uses the same ImageNet-pretrained ResNet34 encoder as the project
baseline and canonical additive attention gates on the four encoder-decoder
skip connections. It deliberately has no boundary/rim targets and no RBSG.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from train_comparison_baseline_common import run_comparison_experiment


def parse_decoder_channels(value):
    if isinstance(value, str):
        value = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    return tuple(int(x) for x in value)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class AdditiveAttentionGate(nn.Module):
    """Oktay-style additive gate applied to one encoder skip feature."""

    def __init__(self, skip_channels, gating_channels, inter_channels):
        super().__init__()
        self.skip_projection = nn.Sequential(
            nn.Conv2d(skip_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.gating_projection = nn.Sequential(
            nn.Conv2d(gating_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, 1, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, skip, gating):
        if gating.shape[-2:] != skip.shape[-2:]:
            gating = F.interpolate(
                gating,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        alpha = self.psi(
            self.skip_projection(skip) + self.gating_projection(gating)
        )
        return skip * alpha


class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.skip_channels = int(skip_channels)
        if self.skip_channels > 0:
            inter_channels = max(min(self.skip_channels, out_channels) // 2, 1)
            self.attention = AdditiveAttentionGate(
                skip_channels=self.skip_channels,
                gating_channels=in_channels,
                inter_channels=inter_channels,
            )
        else:
            self.attention = None
        self.conv = DoubleConv(
            in_channels + self.skip_channels,
            out_channels,
        )

    def forward(self, x, skip=None):
        if skip is None:
            x = F.interpolate(
                x,
                scale_factor=2.0,
                mode="bilinear",
                align_corners=False,
            )
        else:
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            skip = self.attention(skip, x)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class AttentionUNetResNet34(nn.Module):
    """ResNet34 U-Net with ordinary attention-gated skip connections."""

    def __init__(
        self,
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        decoder_channels=(256, 128, 64, 32, 16),
    ):
        super().__init__()
        decoder_channels = parse_decoder_channels(decoder_channels)
        if len(decoder_channels) != 5:
            raise ValueError(
                "decoder_channels must contain five values, got "
                f"{decoder_channels}"
            )

        self.encoder = smp.encoders.get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        encoder_channels = list(self.encoder.out_channels)
        head_channels = encoder_channels[-1]
        skip_channels = list(encoder_channels[1:-1])[::-1] + [0]
        in_decoder_channels = [head_channels] + list(decoder_channels[:-1])

        self.decoder_blocks = nn.ModuleList([
            AttentionDecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(
                in_decoder_channels,
                skip_channels,
                decoder_channels,
            )
        ])
        self.segmentation_head = nn.Conv2d(
            decoder_channels[-1],
            num_classes,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        output_size = x.shape[-2:]
        features = self.encoder(x)
        decoder = features[-1]
        skips = list(features[1:-1])[::-1] + [None]

        for block, skip in zip(self.decoder_blocks, skips):
            decoder = block(decoder, skip)

        logits = self.segmentation_head(decoder)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits


def build_model(args):
    return AttentionUNetResNet34(
        encoder_name=args.encoder_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        decoder_channels=args.decoder_channels,
    )


def main(args):
    metadata = {
        "script": "train_attention_unet_r34_auglite_e260.py",
        "model_class": "AttentionUNetResNet34",
        "model_name": "Attention U-Net (ResNet34 encoder)",
        "experiment": "Controlled external baseline: ResNet34 Attention U-Net",
        "note": (
            "Canonical additive attention gates filter the four ResNet34 skip "
            "features. No boundary/rim supervision, RBSG, or SSGCB is used."
        ),
        "architecture": {
            "encoder": "ResNet34, ImageNet pretrained",
            "decoder_channels": list(parse_decoder_channels(args.decoder_channels)),
            "attention": "additive attention gate before each skip concatenation",
            "attention_skip_levels": 4,
            "output_channels": 1,
        },
        "reference": {
            "title": "Attention U-Net: Learning Where to Look for the Pancreas",
            "authors": "Oktay et al.",
            "year": 2018,
            "url": "https://arxiv.org/abs/1804.03999",
        },
    }
    run_comparison_experiment(args, build_model, metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        type=str,
        default="attention_unet_r34_auglite_e260_constlr_bs6",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument(
        "--augment-profile",
        type=str,
        default="lite",
        choices=["lite", "strong", "none"],
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encoder-name", type=str, default="resnet34")
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument(
        "--decoder-channels",
        type=str,
        default="256,128,64,32,16",
    )
    main(parser.parse_args())
