#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SegFormer-B2 using the official MiT-B2 architecture and optimizer recipe."""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from official_baseline_common import (
    IterationPolynomialLR,
    make_dataset_bundle,
    run_official_experiment,
    set_seed,
)

try:
    from transformers import SegformerForSemanticSegmentation
except ImportError:
    SegformerForSemanticSegmentation = None


class SegFormerB2Official(nn.Module):
    """ImageNet-pretrained MiT-B2 plus the standard all-MLP SegFormer head."""

    def __init__(
        self,
        pretrained_model_name: str = "nvidia/mit-b2",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        if SegformerForSemanticSegmentation is None:
            raise RuntimeError(
                "SegFormer-B2 requires transformers==4.50.0; install requirements.txt first."
            )
        self.network = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=2,
            id2label={0: "background", 1: "caries"},
            label2id={"background": 0, "caries": 1},
            ignore_mismatched_sizes=True,
            local_files_only=local_files_only,
        )
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
        output_size = x.shape[-2:]
        x = (x - self.image_mean) / self.image_std
        logits = self.network(pixel_values=x).logits
        return F.interpolate(
            logits,
            size=output_size,
            mode="bilinear",
            align_corners=False,
        )


def segformer_cross_entropy(output, masks):
    return F.cross_entropy(output, masks[:, 0].long())


def build_official_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    head_lr_multiplier: float,
):
    grouped = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        is_head = "decode_head" in name
        no_decay = (
            parameter.ndim == 1
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "pos_block" in name
        )
        lr = base_lr * (head_lr_multiplier if is_head else 1.0)
        decay = 0.0 if no_decay else weight_decay
        grouped.setdefault((lr, decay), []).append(parameter)
    parameter_groups = [
        {"params": params, "lr": lr, "weight_decay": decay}
        for (lr, decay), params in grouped.items()
    ]
    return torch.optim.AdamW(
        parameter_groups,
        lr=base_lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )


def main(args) -> None:
    set_seed(args.seed)
    datasets = make_dataset_bundle(
        image_size=args.image_size,
        channels=3,
        train_augmentation="segformer_official",
        normalization="zero_one",
    )
    model = SegFormerB2Official(
        pretrained_model_name=args.pretrained_model_name,
        local_files_only=args.local_files_only,
    )
    optimizer = build_official_optimizer(
        model,
        base_lr=args.lr,
        weight_decay=args.weight_decay,
        head_lr_multiplier=args.head_lr_multiplier,
    )
    scheduler = IterationPolynomialLR(
        optimizer,
        total_steps=args.total_iters,
        power=1.0,
        warmup_steps=args.warmup_iters,
        warmup_ratio=args.warmup_ratio,
    )
    steps_per_epoch = math.ceil(len(datasets["train"]) / args.batch_size)
    args.epochs = math.ceil(args.total_iters / steps_per_epoch)
    metadata = {
        "script": "src/train_segformer_b2_official.py",
        "model_name": "SegFormer-B2 (official MiT-B2 + all-MLP decoder)",
        "model_class": "SegFormerB2Official",
        "architecture": {
            "encoder": "MiT-B2 hierarchical Transformer",
            "encoder_pretraining": "ImageNet-1K",
            "checkpoint": args.pretrained_model_name,
            "decoder": "standard SegFormer all-MLP decode head",
            "num_classes": 2,
            "input_size": args.image_size,
            "input_normalization": "ImageNet mean/std",
        },
        "training_recipe": {
            "source": "official NVlabs B2 512x512 ADE20K configuration",
            "optimizer": "AdamW",
            "base_learning_rate": args.lr,
            "decode_head_lr_multiplier": args.head_lr_multiplier,
            "betas": [0.9, 0.999],
            "weight_decay": args.weight_decay,
            "no_decay": "normalization, bias, and positional-block parameters",
            "scheduler": "iteration-wise polynomial, power=1.0",
            "warmup_iters": args.warmup_iters,
            "warmup_ratio": args.warmup_ratio,
            "total_iters": args.total_iters,
            "loss": "two-class cross entropy",
            "batch_size": args.batch_size,
            "augmentation": "random scale 0.5-2.0, crop/pad, flip, photometric distortion",
            "checkpoint_selection": "maximum validation mean IoU",
        },
        "reference": {
            "paper": "https://arxiv.org/abs/2105.15203",
            "official_code": "https://github.com/NVlabs/SegFormer",
            "encoder_checkpoint": "https://huggingface.co/nvidia/mit-b2",
        },
        "fidelity_note": (
            "Architecture, ImageNet MiT-B2 initialization, AdamW parameter groups, "
            "6e-5 base LR, 10x decode-head LR, 0.01 decay, 1500-step warmup, "
            "and polynomial schedule follow the official recipe. The 40k budget "
            "is a documented small-dataset fine-tuning adaptation of ADE20K's 160k."
        ),
    }
    run_official_experiment(
        args,
        model,
        datasets,
        optimizer,
        segformer_cross_entropy,
        metadata,
        scheduler=scheduler,
        scheduler_step="batch",
        max_iterations=args.total_iters,
        val_interval_iterations=args.val_interval_iters,
        selection_metric="mean_iou",
        selection_mode="max",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="segformer_b2_official_40k_bs2")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--total-iters", type=int, default=40000)
    parser.add_argument("--val-interval-iters", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=6e-5)
    parser.add_argument("--head-lr-multiplier", type=float, default=10.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-iters", type=int, default=1500)
    parser.add_argument("--warmup-ratio", type=float, default=1e-6)
    parser.add_argument("--pretrained-model-name", default="nvidia/mit-b2")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    # Filled from total_iters and split size before the common runner starts.
    parser.set_defaults(epochs=1)
    main(parser.parse_args())
