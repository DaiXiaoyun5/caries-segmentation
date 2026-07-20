#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""VainF-faithful DeepLabV3+-ResNet50 baseline for CariXray."""

import argparse
import math

from deeplabv3plus_vainf_common import (
    VainFDeepLabV3PlusResNet50,
    build_vainf_deeplabv3plus,
    build_vainf_official_optimizer,
    deeplab_cross_entropy,
)
from official_baseline_common import (
    IterationPolynomialLR,
    make_dataset_bundle,
    run_cli_with_failure_report,
    run_official_experiment,
    set_seed,
)


def main(args) -> None:
    set_seed(args.seed)
    datasets = make_dataset_bundle(
        image_size=args.image_size,
        channels=3,
        train_augmentation="deeplab_official",
        normalization="zero_one",
    )
    model = build_vainf_deeplabv3plus(pretrained=True)
    optimizer = build_vainf_official_optimizer(
        model,
        classifier_lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = IterationPolynomialLR(
        optimizer,
        total_steps=args.total_iters,
        power=0.9,
    )

    steps_per_epoch = len(datasets["train"]) // args.batch_size
    if steps_per_epoch < 1:
        raise RuntimeError("Training split is smaller than one DeepLab batch")
    args.epochs = math.ceil(args.total_iters / steps_per_epoch)

    metadata = {
        "script": "src/train_deeplabv3plus_vainf_official.py",
        "model_name": "DeepLabV3+ (VainF ResNet50, output stride 16)",
        "model_class": VainFDeepLabV3PlusResNet50.__name__,
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
        },
        "training_recipe": {
            "source": "VainF DeepLabV3Plus-Pytorch reference recipe",
            "optimizer": "SGD",
            "momentum": 0.9,
            "classifier_learning_rate": args.lr,
            "backbone_learning_rate": args.lr * 0.1,
            "lr_scaling_note": "0.01 at batch 16 scaled linearly to batch 4",
            "weight_decay": args.weight_decay,
            "scheduler": "iteration-wise polynomial, power=0.9",
            "total_iters": args.total_iters,
            "loss": "unweighted two-class cross entropy",
            "batch_size": args.batch_size,
            "drop_last": True,
            "augmentation": "random scale 0.5-2.0, crop/pad, horizontal flip",
            "backbone_batch_norm_momentum": 0.01,
            "checkpoint_selection": "maximum validation mean IoU",
        },
        "reference": {
            "paper": "https://arxiv.org/abs/1802.02611",
            "implementation": (
                "https://github.com/VainF/DeepLabV3Plus-Pytorch"
            ),
            "modeling": (
                "https://github.com/VainF/DeepLabV3Plus-Pytorch/"
                "blob/master/network/modeling.py"
            ),
            "training": (
                "https://github.com/VainF/DeepLabV3Plus-Pytorch/"
                "blob/master/main.py"
            ),
        },
        "fidelity_note": (
            "This is the strict VainF-style rerun. Unlike the superseded SMP "
            "run, OS16 uses ASPP rates 6/12/18 and the default VainF decoder "
            "without optional separable-convolution conversion."
        ),
        "supersedes_for_main_table": "deeplabv3plus_r50_os16_30k_bs4",
    }

    run_official_experiment(
        args,
        model,
        datasets,
        optimizer,
        deeplab_cross_entropy,
        metadata,
        scheduler=scheduler,
        scheduler_step="batch",
        max_iterations=args.total_iters,
        val_interval_iterations=args.val_interval_iters,
        selection_metric="mean_iou",
        selection_mode="max",
        drop_last_train=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default="deeplabv3plus_vainf_r50_os16_official_30k_bs4",
    )
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=4)
    parser.add_argument("--total-iters", type=int, default=30000)
    parser.add_argument("--val-interval-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.0025)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.set_defaults(epochs=1)
    run_cli_with_failure_report(main, parser.parse_args())
