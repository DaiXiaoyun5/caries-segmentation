#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Cache and verify pretrained weights used by the official 2D baselines.

The cluster compute nodes are not assumed to have Internet access.  Run this
utility from the login node through ``setup_official_baselines_env.sh``.  The
submit jobs call it again with ``--verify-only`` so a missing cache fails with
an actionable message before a GPU training process is started.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"
TORCH_HOME = PROJECT_ROOT / ".cache" / "torch"
MANIFEST_PATH = (
    PROJECT_ROOT / ".cache" / "official_baselines" / "pretrained_assets.json"
)


def configure_cache(verify_only: bool) -> None:
    HF_HOME.mkdir(parents=True, exist_ok=True)
    (TORCH_HOME / "hub" / "checkpoints").mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    # These variables must be set before importing Transformers.
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["TORCH_HOME"] = str(TORCH_HOME)
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    if verify_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def describe_files(paths: List[Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in sorted(set(paths)):
        if not path.is_file():
            continue
        records.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "size_bytes": path.stat().st_size,
            }
        )
    return records


def prepare_segformer(verify_only: bool) -> Dict[str, object]:
    from transformers import SegformerForSemanticSegmentation

    try:
        model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=2,
            id2label={0: "background", 1: "caries"},
            label2id={"background": 0, "caries": 1},
            ignore_mismatched_sizes=True,
            local_files_only=verify_only,
        )
    except Exception as exc:
        mode = "offline verification" if verify_only else "login-node download"
        raise RuntimeError(
            "SegFormer MiT-B2 pretrained weights failed during "
            f"{mode}. Run bash scripts/setup/setup_official_baselines_env.sh "
            "on the login node before submitting SegFormer."
        ) from exc

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    gc.collect()

    repo_cache = HF_HOME / "hub" / "models--nvidia--mit-b2"
    cached_files = list((repo_cache / "snapshots").glob("*/*"))
    if not cached_files:
        raise RuntimeError(
            f"MiT-B2 loaded but no Hugging Face snapshot was found under {repo_cache}"
        )
    return {
        "source": "https://huggingface.co/nvidia/mit-b2",
        "loader": "SegformerForSemanticSegmentation.from_pretrained",
        "local_files_only": verify_only,
        "parameter_count_after_binary_head_adaptation": parameter_count,
        "cached_files": describe_files(cached_files),
    }


def prepare_deeplab(verify_only: bool) -> Dict[str, object]:
    import segmentation_models_pytorch as smp

    checkpoint_dir = TORCH_HOME / "hub" / "checkpoints"
    candidates = list(checkpoint_dir.glob("resnet50-*.pth"))
    if verify_only and not candidates:
        raise RuntimeError(
            "No cached ResNet50 ImageNet checkpoint was found under "
            f"{checkpoint_dir}. Run bash "
            "scripts/setup/setup_official_baselines_env.sh on the login node "
            "before submitting DeepLabV3+."
        )

    try:
        model = smp.DeepLabV3Plus(
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
    except Exception as exc:
        mode = "offline verification" if verify_only else "login-node download"
        raise RuntimeError(
            "DeepLabV3+ ResNet50 ImageNet weights failed during "
            f"{mode}. Run bash scripts/setup/setup_official_baselines_env.sh "
            "on the login node before submitting DeepLabV3+."
        ) from exc

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    del model
    gc.collect()

    candidates = list(checkpoint_dir.glob("resnet50-*.pth"))
    if not candidates:
        raise RuntimeError(
            f"ResNet50 loaded but no checkpoint was found under {checkpoint_dir}"
        )
    return {
        "source": "https://download.pytorch.org/models/",
        "loader": "segmentation_models_pytorch.DeepLabV3Plus",
        "encoder_weights": "imagenet",
        "parameter_count": parameter_count,
        "cached_files": describe_files(candidates),
    }


def load_manifest() -> Dict[str, object]:
    if not MANIFEST_PATH.exists():
        return {"models": {}}
    try:
        payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"models": {}}
    if not isinstance(payload.get("models"), dict):
        payload["models"] = {}
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=("all", "segformer", "deeplab"),
        default="all",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Forbid Hugging Face downloads and verify the existing caches.",
    )
    args = parser.parse_args()

    configure_cache(args.verify_only)
    manifest = load_manifest()
    models = manifest["models"]

    if args.model in ("all", "segformer"):
        print("Checking SegFormer MiT-B2 pretrained assets...", flush=True)
        models["segformer_mit_b2"] = prepare_segformer(args.verify_only)
        print("SegFormer MiT-B2 cache: OK", flush=True)

    if args.model in ("all", "deeplab"):
        print("Checking DeepLabV3+ ResNet50 pretrained assets...", flush=True)
        models["deeplabv3plus_resnet50"] = prepare_deeplab(args.verify_only)
        print("DeepLabV3+ ResNet50 cache: OK", flush=True)

    manifest.update(
        {
            "schema_version": 1,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "verification_mode": "offline" if args.verify_only else "download_allowed",
            "hf_home": str(HF_HOME),
            "torch_home": str(TORCH_HOME),
        }
    )
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Asset manifest: {MANIFEST_PATH}", flush=True)


if __name__ == "__main__":
    main()
