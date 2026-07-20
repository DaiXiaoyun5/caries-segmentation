#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""CariXray-adapted training recipe for the unchanged VainF DeepLabV3+."""

import argparse

import torch

from deeplabv3plus_vainf_common import (
    VainFDeepLabV3PlusResNet50,
    build_vainf_deeplabv3plus,
    deeplab_ce_foreground_dice,
    make_auglite_dataset_bundle,
)
from official_baseline_common import (
    run_cli_with_failure_report,
    run_official_experiment,
    set_seed,
)


def main(args) -> None:
    set_seed(args.seed)
    datasets = make_auglite_dataset_bundle(args.image_size)
    model = build_vainf_deeplabv3plus(pretrained=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metadata = {
        "script": "src/train_deeplabv3plus_vainf_medical_e260.py",
        "model_name": "DeepLabV3+ (VainF ResNet50, OS16; CariXray recipe)",
        "model_class": VainFDeepLabV3PlusResNet50.__name__,
        "protocol_family": "CariXray-adapted",
        "architecture": {
            "source": "VainF/DeepLabV3Plus-Pytorch vendored snapshot",
            "source_manifest": (
                "third_party/vainf_deeplabv3plus/source_manifest.json"
            ),
            "encoder": "ResNet50, ImageNet pretrained from local cache",
            "output_stride": 16,
            "aspp_atrous_rates": [6, 12, 18],
            "aspp_channels": 256,
            "low_level_projection_channels": 48,
            "optional_separable_conversion": False,
            "num_classes": 2,
            "input_normalization": "ImageNet mean/std",
            "identical_to_strict_network": True,
        },
        "training_recipe": {
            "source": "CariXray-adapted controlled medical recipe",
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": args.weight_decay,
            "scheduler_name": "constant",
            "epochs": args.epochs,
            "loss": "two-class cross entropy + foreground soft Dice",
            "batch_size": args.batch_size,
            "augmentation": "project AugLite, whole image at 512x512",
            "checkpoint_selection": "maximum validation Dice",
        },
        "reference": {
            "paper": "https://arxiv.org/abs/1802.02611",
            "implementation": "https://github.com/VainF/DeepLabV3Plus-Pytorch",
        },
        "comparison_note": (
            "The network is byte-for-byte the same vendored VainF model used by "
            "the strict run. Only the loss, optimizer, augmentation, batch size, "
            "seed, budget, and checkpoint criterion follow the CariXray protocol."
        ),
    }

    run_official_experiment(
        args,
        model,
        datasets,
        optimizer,
        deeplab_ce_foreground_dice,
        metadata,
        scheduler=None,
        scheduler_step="none",
        max_iterations=None,
        val_every_epochs=1,
        selection_metric="dice",
        selection_mode="max",
        drop_last_train=False,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="deeplabv3plus_vainf_r50_os16_medical_auglite_e260_bs6",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=260)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    run_cli_with_failure_report(main, parser.parse_args())
