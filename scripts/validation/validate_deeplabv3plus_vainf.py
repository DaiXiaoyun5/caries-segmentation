#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Runtime preflight for both VainF DeepLabV3+ experiment entry points."""

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch

from deeplabv3plus_vainf_common import (
    VainFDeepLabV3PlusResNet50,
    build_vainf_deeplabv3plus,
    build_vainf_official_optimizer,
    deeplab_ce_foreground_dice,
    deeplab_cross_entropy,
)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA validation was requested but torch cannot see a GPU")
    return torch.device(requested)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-pretrained", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model = build_vainf_deeplabv3plus(
        pretrained=not args.skip_pretrained
    ).to(device).eval()
    twin = build_vainf_deeplabv3plus(pretrained=False).eval()
    if set(model.state_dict()) != set(twin.state_dict()):
        raise RuntimeError("Official and medical builders do not share one architecture")

    rates = tuple(
        branch[0].dilation[0]
        for branch in model.network.classifier.aspp.convs[1:4]
    )
    if rates != VainFDeepLabV3PlusResNet50.aspp_atrous_rates:
        raise RuntimeError(f"Unexpected ASPP rates: {rates}")

    images = torch.zeros(1, 3, 64, 64, device=device)
    masks = torch.zeros(1, 1, 64, 64, device=device)
    masks[:, :, 24:40, 24:40] = 1.0
    with torch.no_grad():
        output = model(images)
        ce = deeplab_cross_entropy(output, masks)
        medical = deeplab_ce_foreground_dice(output, masks)
    if tuple(output.shape) != (1, 2, 64, 64):
        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(ce) or not torch.isfinite(medical):
        raise RuntimeError(f"Non-finite preflight loss: CE={ce}, medical={medical}")

    optimizer = build_vainf_official_optimizer(model, 0.0025, 1e-4)
    lrs = [float(group["lr"]) for group in optimizer.param_groups]
    if lrs != [0.00025, 0.0025]:
        raise RuntimeError(f"Unexpected optimizer learning rates: {lrs}")

    print("torch:", torch.__version__)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    print("pretrained checkpoint:", model.pretrained_checkpoint)
    print("ASPP rates:", rates)
    print("output:", tuple(output.shape))
    print("losses:", {"cross_entropy": float(ce), "medical": float(medical)})
    print("optimizer learning rates:", lrs)
    print("VainF DeepLabV3+ validation: OK")


if __name__ == "__main__":
    main()
