#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""DEPRECATED B30-controlled SegFormer experiment.

This historical E260/constant-lr/BCE variant is retained for auditability but
does not use SegFormer's official optimizer, two-class CE, or polynomial
schedule. Use ``train_segformer_b2_official.py`` for the paper comparison.
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_comparison_baseline_common import run_comparison_experiment

try:
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
except ImportError:
    SegformerConfig = None
    SegformerForSemanticSegmentation = None


class SegFormerB2Binary(nn.Module):
    """NVIDIA MiT-B2 encoder plus the standard SegFormer MLP decoder."""

    def __init__(
        self,
        pretrained_model_name="nvidia/mit-b2",
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
        normalize_inputs=True,
        local_files_only=False,
    ):
        super().__init__()
        if SegformerForSemanticSegmentation is None:
            raise RuntimeError(
                "SegFormer-B2 requires transformers==4.49.0. "
                "Install the comparison dependency before submitting this job."
            )
        if in_channels != 3:
            raise ValueError("The ImageNet-pretrained MiT-B2 checkpoint expects 3 channels.")

        self.normalize_inputs = bool(normalize_inputs)
        pretrained = str(encoder_weights).lower() not in {
            "none", "null", "false", "random", "scratch",
        }
        common_config = {
            "num_labels": int(num_classes),
            "id2label": {0: "caries"},
            "label2id": {"caries": 0},
        }

        if pretrained:
            self.segformer = SegformerForSemanticSegmentation.from_pretrained(
                pretrained_model_name,
                ignore_mismatched_sizes=True,
                local_files_only=bool(local_files_only),
                **common_config,
            )
        else:
            config = SegformerConfig.from_pretrained(
                pretrained_model_name,
                local_files_only=bool(local_files_only),
                **common_config,
            )
            self.segformer = SegformerForSemanticSegmentation(config)

        self.register_buffer(
            "input_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "input_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, x):
        output_size = x.shape[-2:]
        if self.normalize_inputs:
            x = (x - self.input_mean) / self.input_std
        logits = self.segformer(pixel_values=x).logits
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(
                logits,
                size=output_size,
                mode="bilinear",
                align_corners=False,
            )
        return logits


def build_model(args):
    return SegFormerB2Binary(
        pretrained_model_name=args.pretrained_model_name,
        encoder_weights=args.encoder_weights,
        in_channels=3,
        num_classes=1,
        normalize_inputs=bool(args.normalize_inputs),
        local_files_only=bool(args.local_files_only),
    )


def main(args):
    metadata = {
        "script": "train_segformer_b2_auglite_e260.py",
        "model_class": "SegFormerB2Binary",
        "model_name": "SegFormer-B2",
        "experiment": "Controlled external baseline: ImageNet-pretrained SegFormer-B2",
        "note": (
            "Uses the official NVIDIA MiT-B2 ImageNet encoder checkpoint and "
            "the standard SegFormer all-MLP decode head with one binary logit."
        ),
        "architecture": {
            "encoder": "MiT-B2 hierarchical Transformer",
            "decoder": "standard SegFormer all-MLP decode head",
            "pretrained_model_name": args.pretrained_model_name,
            "input_channels": 3,
            "xray_channel_handling": "RGB conversion/repetition by the shared dataset",
            "imagenet_normalization": bool(args.normalize_inputs),
            "output_channels": 1,
            "output_resize": "bilinear to input resolution before loss/metrics",
        },
        "reference": {
            "title": "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers",
            "authors": "Xie et al.",
            "venue": "NeurIPS",
            "year": 2021,
            "url": "https://arxiv.org/abs/2105.15203",
            "official_code": "https://github.com/NVlabs/SegFormer",
            "checkpoint": "https://huggingface.co/nvidia/mit-b2",
        },
    }
    run_comparison_experiment(args, build_model, metadata)


if __name__ == "__main__":
    raise SystemExit(
        "Deprecated comparison recipe. Submit "
        "src/train_segformer_b2_official.py instead."
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        type=str,
        default="segformer_b2_auglite_e260_constlr_bs6",
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
    parser.add_argument("--encoder-weights", type=str, default="imagenet")
    parser.add_argument(
        "--pretrained-model-name",
        type=str,
        default="nvidia/mit-b2",
    )
    parser.add_argument("--normalize-inputs", type=int, default=1, choices=[0, 1])
    parser.add_argument("--local-files-only", action="store_true")
    main(parser.parse_args())
