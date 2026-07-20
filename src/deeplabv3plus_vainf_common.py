#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared VainF DeepLabV3+ model and CariXray recipe helpers.

The network source is vendored under ``third_party/vainf_deeplabv3plus`` so
compute jobs do not need GitHub access.  ImageNet weights are loaded only from
the project's existing torchvision cache.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Dict


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from official_baseline_common import CariesOfficialDataset
from third_party.vainf_deeplabv3plus.network import modeling as vainf_modeling
from train_resnet34_unet_auglite_e200_constlr import AugmentedCariesMaskDataset


DEFAULT_TORCH_HOME = PROJECT_ROOT / ".cache/torch"
EXPECTED_UNUSED_RESNET_KEYS = {"fc.weight", "fc.bias"}


def find_resnet50_checkpoint(torch_home: Path = DEFAULT_TORCH_HOME) -> Path:
    """Return one deterministic cached torchvision ResNet50 state dictionary."""

    checkpoint_dir = Path(torch_home) / "hub/checkpoints"
    candidates = {path.name: path for path in checkpoint_dir.glob("resnet50-*.pth")}
    preferred_names = (
        "resnet50-0676ba61.pth",
        "resnet50-11ad3fa6.pth",
        "resnet50-19c8e357.pth",
    )
    for name in preferred_names:
        if name in candidates:
            return candidates[name]
    if len(candidates) == 1:
        return next(iter(candidates.values()))
    if not candidates:
        raise RuntimeError(
            "No cached ResNet50 ImageNet checkpoint was found under "
            f"{checkpoint_dir}. Run scripts/setup/"
            "setup_official_baselines_env.sh on the login node first."
        )
    raise RuntimeError(
        "Multiple unrecognized ResNet50 checkpoints were found under "
        f"{checkpoint_dir}: {', '.join(sorted(candidates))}. Keep one known "
        "torchvision checkpoint or extend the explicit preference list."
    )


def _unwrap_state_dict(payload) -> Dict[str, torch.Tensor]:
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise RuntimeError("The cached ResNet50 checkpoint is not a state dictionary")
    state_dict = {}
    for key, value in payload.items():
        normalized = str(key)
        if normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        state_dict[normalized] = value
    return state_dict


def _load_cached_resnet50_backbone(
    network: nn.Module,
    checkpoint_path: Path,
) -> None:
    payload = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = _unwrap_state_dict(payload)
    incompatible = network.backbone.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing or unexpected != EXPECTED_UNUSED_RESNET_KEYS:
        raise RuntimeError(
            "Cached ResNet50 weights do not match the VainF backbone: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )


class VainFDeepLabV3PlusResNet50(nn.Module):
    """Unmodified VainF ResNet50 DeepLabV3+ network with offline weights."""

    output_stride = 16
    aspp_atrous_rates = (6, 12, 18)

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.network = vainf_modeling.deeplabv3plus_resnet50(
            num_classes=2,
            output_stride=self.output_stride,
            pretrained_backbone=False,
        )
        self.pretrained_checkpoint = None
        if pretrained:
            checkpoint_path = find_resnet50_checkpoint()
            _load_cached_resnet50_backbone(self.network, checkpoint_path)
            self.pretrained_checkpoint = str(checkpoint_path)

        # VainF's training entry point applies momentum 0.01 to the backbone.
        for module in self.network.backbone.modules():
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network((images - self.image_mean) / self.image_std)


def build_vainf_deeplabv3plus(pretrained: bool = True) -> nn.Module:
    return VainFDeepLabV3PlusResNet50(pretrained=pretrained)


def build_vainf_official_optimizer(
    model: nn.Module,
    classifier_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    network = model.network if hasattr(model, "network") else model
    return torch.optim.SGD(
        [
            {
                "params": network.backbone.parameters(),
                "lr": float(classifier_lr) * 0.1,
                "name": "backbone",
            },
            {
                "params": network.classifier.parameters(),
                "lr": float(classifier_lr),
                "name": "classifier",
            },
        ],
        lr=float(classifier_lr),
        momentum=0.9,
        weight_decay=float(weight_decay),
    )


def deeplab_cross_entropy(
    output: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    return F.cross_entropy(output, masks[:, 0].long())


def _foreground_soft_dice_loss(
    output: torch.Tensor,
    masks: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    foreground = torch.softmax(output, dim=1)[:, 1:2]
    targets = masks.float()
    dims = (1, 2, 3)
    intersection = (foreground * targets).sum(dims)
    denominator = foreground.sum(dims) + targets.sum(dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def deeplab_ce_foreground_dice(
    output: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    return deeplab_cross_entropy(output, masks) + _foreground_soft_dice_loss(
        output,
        masks,
    )


def make_auglite_dataset_bundle(image_size: int):
    """Use the project's exact whole-image AugLite only for training."""

    split_dir = PROJECT_ROOT / "data/splits"
    evaluation_common = {
        "image_size": int(image_size),
        "channels": 3,
        "train": False,
        "augmentation": "none",
        "normalization": "zero_one",
    }
    return {
        "train": AugmentedCariesMaskDataset(
            split_dir / "train.txt",
            target_size=(int(image_size), int(image_size)),
            profile="lite",
        ),
        "train_eval": CariesOfficialDataset(
            split_dir / "train.txt",
            **evaluation_common,
        ),
        "val": CariesOfficialDataset(
            split_dir / "val.txt",
            **evaluation_common,
        ),
        "test": CariesOfficialDataset(
            split_dir / "test.txt",
            **evaluation_common,
        ),
    }
