#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Native UNet++ with the architecture and loss described by Zhou et al."""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from official_baseline_common import (
    binary_bce_dice_loss,
    make_dataset_bundle,
    run_official_experiment,
    set_seed,
)


class UNetPPConvBlock(nn.Module):
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


class UNetPlusPlusOfficial2D(nn.Module):
    """Original nested dense skip topology with four deep-supervision heads."""

    FILTERS = (32, 64, 128, 256, 512)

    def __init__(self, in_channels: int = 1, num_classes: int = 1) -> None:
        super().__init__()
        f0, f1, f2, f3, f4 = self.FILTERS
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = UNetPPConvBlock(in_channels, f0)
        self.conv1_0 = UNetPPConvBlock(f0, f1)
        self.conv2_0 = UNetPPConvBlock(f1, f2)
        self.conv3_0 = UNetPPConvBlock(f2, f3)
        self.conv4_0 = UNetPPConvBlock(f3, f4)

        self.conv0_1 = UNetPPConvBlock(f0 + f1, f0)
        self.conv1_1 = UNetPPConvBlock(f1 + f2, f1)
        self.conv2_1 = UNetPPConvBlock(f2 + f3, f2)
        self.conv3_1 = UNetPPConvBlock(f3 + f4, f3)

        self.conv0_2 = UNetPPConvBlock(f0 * 2 + f1, f0)
        self.conv1_2 = UNetPPConvBlock(f1 * 2 + f2, f1)
        self.conv2_2 = UNetPPConvBlock(f2 * 2 + f3, f2)

        self.conv0_3 = UNetPPConvBlock(f0 * 3 + f1, f0)
        self.conv1_3 = UNetPPConvBlock(f1 * 3 + f2, f1)

        self.conv0_4 = UNetPPConvBlock(f0 * 4 + f1, f0)

        self.final1 = nn.Conv2d(f0, num_classes, 1)
        self.final2 = nn.Conv2d(f0, num_classes, 1)
        self.final3 = nn.Conv2d(f0, num_classes, 1)
        self.final4 = nn.Conv2d(f0, num_classes, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    @staticmethod
    def upsample(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=reference.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

    def forward(self, x: torch.Tensor):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.upsample(x1_0, x0_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.upsample(x2_0, x1_0)], 1))
        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self.upsample(x1_1, x0_0)], 1)
        )

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.upsample(x3_0, x2_0)], 1))
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, self.upsample(x2_1, x1_0)], 1)
        )
        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.upsample(x1_2, x0_0)], 1)
        )

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.upsample(x4_0, x3_0)], 1))
        x2_2 = self.conv2_2(
            torch.cat([x2_0, x2_1, self.upsample(x3_1, x2_0)], 1)
        )
        x1_3 = self.conv1_3(
            torch.cat([x1_0, x1_1, x1_2, self.upsample(x2_2, x1_0)], 1)
        )
        x0_4 = self.conv0_4(
            torch.cat(
                [x0_0, x0_1, x0_2, x0_3, self.upsample(x1_3, x0_0)],
                1,
            )
        )

        # Accurate mode in the paper averages all four segmentation branches.
        return [
            self.final1(x0_1),
            self.final2(x0_2),
            self.final3(x0_3),
            self.final4(x0_4),
        ]


def unetpp_deep_supervision_loss(outputs, masks):
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 4:
        raise RuntimeError("UNet++ accurate mode requires four supervised outputs")
    return torch.stack([binary_bce_dice_loss(output, masks) for output in outputs]).mean()


def main(args) -> None:
    set_seed(args.seed)
    datasets = make_dataset_bundle(
        image_size=args.image_size,
        channels=1,
        train_augmentation="unetpp_medical",
        normalization="zero_one",
    )
    model = UNetPlusPlusOfficial2D(in_channels=1, num_classes=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    metadata = {
        "script": "src/train_unetpp_official.py",
        "model_name": "UNet++ (native nested architecture with deep supervision)",
        "model_class": "UNetPlusPlusOfficial2D",
        "architecture": {
            "encoder": "native UNet++ convolutional encoder trained from scratch",
            "filters": list(UNetPlusPlusOfficial2D.FILTERS),
            "nested_dense_skip_pathways": True,
            "deep_supervision_heads": 4,
            "inference_mode": "accurate mode; mean probability of all four heads",
            "resnet_encoder": False,
        },
        "training_recipe": {
            "source": "UNet++ paper implementation details",
            "input_size": args.image_size,
            "augmentation": "conservative affine transforms and horizontal flip",
            "optimizer": "Adam",
            "learning_rate": args.lr,
            "loss": "equal mean of BCE + Dice at four semantic levels",
            "maximum_epochs": args.epochs,
            "early_stopping_patience": args.early_stopping_patience,
            "batch_size": args.batch_size,
            "checkpoint_selection": "minimum validation deep-supervision loss",
        },
        "reference": {
            "paper": "https://arxiv.org/abs/1807.10165",
            "official_code": "https://github.com/MrGiovanni/UNetPlusPlus",
        },
        "fidelity_note": (
            "The 32/64/128/256/512 widths, nested dense nodes, four heads, "
            "accurate-mode averaging, Adam 3e-4, and BCE+Dice objective follow "
            "the original paper. No ResNet34 encoder is substituted."
        ),
    }
    run_official_experiment(
        args,
        model,
        datasets,
        optimizer,
        unetpp_deep_supervision_loss,
        metadata,
        val_every_epochs=1,
        selection_metric="loss",
        selection_mode="min",
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="unetpp_official_ds_e200_bs4")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
