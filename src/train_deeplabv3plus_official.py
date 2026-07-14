#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Standard DeepLabV3+-ResNet50 baseline with the accepted SGD/poly recipe."""

import argparse
import math

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from official_baseline_common import (
    IterationPolynomialLR,
    make_dataset_bundle,
    run_official_experiment,
    set_seed,
)


class DeepLabV3PlusResNet50(nn.Module):
    """ResNet50 OS16 + ASPP + low-level decoder, without project modules."""

    def __init__(self) -> None:
        super().__init__()
        self.network = smp.DeepLabV3Plus(
            encoder_name="resnet50",
            encoder_depth=5,
            encoder_weights="imagenet",
            encoder_output_stride=16,
            decoder_channels=256,
            decoder_atrous_rates=(12, 24, 36),
            in_channels=3,
            classes=2,
            activation=None,
            upsampling=4,
        )
        for module in self.network.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0.01
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network((x - self.image_mean) / self.image_std)


def deeplab_cross_entropy(output, masks):
    return F.cross_entropy(output, masks[:, 0].long())


def build_deeplab_optimizer(model, classifier_lr, weight_decay):
    return torch.optim.SGD(
        [
            {
                "params": model.network.encoder.parameters(),
                "lr": classifier_lr * 0.1,
            },
            {
                "params": model.network.decoder.parameters(),
                "lr": classifier_lr,
            },
            {
                "params": model.network.segmentation_head.parameters(),
                "lr": classifier_lr,
            },
        ],
        lr=classifier_lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )


def main(args) -> None:
    set_seed(args.seed)
    datasets = make_dataset_bundle(
        image_size=args.image_size,
        channels=3,
        train_augmentation="deeplab_official",
        normalization="zero_one",
    )
    model = DeepLabV3PlusResNet50()
    optimizer = build_deeplab_optimizer(model, args.lr, args.weight_decay)
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
        "script": "src/train_deeplabv3plus_official.py",
        "model_name": "DeepLabV3+ (ResNet50, output stride 16)",
        "model_class": "DeepLabV3PlusResNet50",
        "architecture": {
            "encoder": "ResNet50, ImageNet pretrained",
            "output_stride": 16,
            "aspp_atrous_rates": [12, 24, 36],
            "decoder_channels": 256,
            "low_level_feature_decoder": True,
            "num_classes": 2,
            "input_normalization": "ImageNet mean/std",
            "implementation": "segmentation-models-pytorch standard DeepLabV3Plus",
        },
        "training_recipe": {
            "source": "widely used VainF DeepLabV3Plus PyTorch recipe",
            "optimizer": "SGD",
            "momentum": 0.9,
            "classifier_learning_rate": args.lr,
            "encoder_learning_rate": args.lr * 0.1,
            "lr_scaling_note": "0.01 at batch 16 scaled linearly to batch 4",
            "weight_decay": args.weight_decay,
            "scheduler": "iteration-wise polynomial, power=0.9",
            "total_iters": args.total_iters,
            "loss": "two-class cross entropy",
            "batch_size": args.batch_size,
            "drop_last": True,
            "augmentation": "random scale 0.5-2.0, crop/pad, horizontal flip",
            "batch_norm_momentum": 0.01,
            "checkpoint_selection": "maximum validation mean IoU",
        },
        "reference": {
            "paper": "https://arxiv.org/abs/1802.02611",
            "reference_implementation": "https://github.com/VainF/DeepLabV3Plus-Pytorch",
            "maintained_pytorch_implementation": "https://github.com/qubvel-org/segmentation_models.pytorch",
        },
        "fidelity_note": (
            "This replaces the old ResNet34/e50 exploratory run. It uses the "
            "standard ResNet50 OS16 architecture, ASPP rates, CE objective, "
            "backbone 0.1x LR, SGD, and poly schedule."
        ),
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
    parser.add_argument("--run-name", default="deeplabv3plus_r50_os16_30k_bs4")
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
    main(parser.parse_args())
