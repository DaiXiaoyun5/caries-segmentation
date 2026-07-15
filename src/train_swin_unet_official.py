#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Adapter that trains the supplied official Swin-Unet without rewriting it."""

import argparse
import hashlib
import importlib
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from official_baseline_common import (
    IterationPolynomialLR,
    make_dataset_bundle,
    multiclass_dice_loss_official_swin,
    run_cli_with_failure_report,
    run_official_experiment,
    set_seed,
)


OFFICIAL_ZIP_SHA256 = "0dff0004b71514fb3509b4fabc6b5ffd76d33aee283ba9fca8e0b1ceeb6215dc"
OFFICIAL_PRETRAIN_SHA256 = "9f71c168d837d1b99dd1dc29e14990a7a9e8bdc5f673d46b04fe36fe15590ad3"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_swin_model(args):
    source_dir = Path(args.swin_source_dir).expanduser().resolve()
    checkpoint = Path(args.pretrained_checkpoint).expanduser().resolve()
    cfg_path = source_dir / "configs/swin_tiny_patch4_window7_224_lite.yaml"
    required = (
        source_dir / "config.py",
        source_dir / "networks/vision_transformer.py",
        source_dir / "networks/swin_transformer_unet_skip_expand_decoder_sys.py",
        cfg_path,
        checkpoint,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Official Swin-Unet assets are incomplete:\n  " + "\n  ".join(missing)
        )
    actual_hash = sha256_file(checkpoint)
    if actual_hash.lower() != OFFICIAL_PRETRAIN_SHA256:
        raise RuntimeError(
            "Unexpected Swin-T checkpoint SHA256: "
            f"{actual_hash}; expected {OFFICIAL_PRETRAIN_SHA256}"
        )

    sys.path.insert(0, str(source_dir))
    config_module = importlib.import_module("config")
    network_module = importlib.import_module("networks.vision_transformer")
    config_args = SimpleNamespace(
        cfg=str(cfg_path),
        opts=None,
        batch_size=args.batch_size,
        zip=False,
        cache_mode="part",
        resume=None,
        accumulation_steps=None,
        use_checkpoint=False,
        amp_opt_level="O0",
        tag=None,
        eval=False,
        throughput=False,
    )
    config = config_module.get_config(config_args)
    config.defrost()
    config.DATA.IMG_SIZE = 224
    config.MODEL.PRETRAIN_CKPT = str(checkpoint)
    config.freeze()
    model = network_module.SwinUnet(
        config,
        img_size=224,
        num_classes=2,
        zero_head=False,
        vis=False,
    )
    # Official load_from mirrors encoder weights into the symmetric decoder.
    model.load_from(config)
    return model, source_dir, checkpoint, actual_hash


def swin_official_loss(output, masks):
    labels = masks[:, 0].long()
    loss_ce = F.cross_entropy(output, labels)
    loss_dice = multiclass_dice_loss_official_swin(output, masks, num_classes=2)
    return 0.4 * loss_ce + 0.6 * loss_dice


def main(args) -> None:
    set_seed(args.seed)
    if args.image_size != 224:
        raise ValueError(
            "The supplied official Swin-Unet configuration is fixed at 224; "
            "changing it would no longer be a faithful comparison."
        )
    datasets = make_dataset_bundle(
        image_size=224,
        channels=1,
        train_augmentation="swin_unet_official",
        normalization="zero_one",
    )
    model, source_dir, checkpoint, checkpoint_hash = load_official_swin_model(args)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    steps_per_epoch = math.ceil(len(datasets["train"]) / args.batch_size)
    total_steps = args.epochs * steps_per_epoch
    scheduler = IterationPolynomialLR(
        optimizer,
        total_steps=total_steps,
        power=0.9,
    )
    metadata = {
        "script": "src/train_swin_unet_official.py",
        "model_name": "Swin-Unet (official Swin-T 224 configuration)",
        "model_class": "official networks.vision_transformer.SwinUnet",
        "architecture": {
            "source_dir": str(source_dir),
            "configuration": "swin_tiny_patch4_window7_224_lite.yaml",
            "input_size": 224,
            "patch_size": 4,
            "window_size": 7,
            "embed_dim": 96,
            "depths": [2, 2, 2, 2],
            "decoder_depths": [2, 2, 2, 1],
            "num_heads": [3, 6, 12, 24],
            "drop_path_rate": 0.2,
            "num_classes": 2,
            "pretrained_checkpoint": str(checkpoint),
            "pretrained_checkpoint_sha256": checkpoint_hash,
            "pretrained_initialization": "official load_from encoder-to-decoder mirroring",
        },
        "training_recipe": {
            "source": "supplied official Swin-Unet repository trainer.py/train.sh",
            "optimizer": "SGD",
            "momentum": 0.9,
            "weight_decay": 1e-4,
            "base_lr_batch24": 0.05,
            "learning_rate_batch6": args.lr,
            "lr_scaling": "0.05 * batch_size / 24",
            "scheduler": "iteration-wise polynomial, power=0.9",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "loss": "0.4 cross entropy + 0.6 official multiclass Dice",
            "augmentation": "official random rot90/flip or random rotation, bicubic resize",
            "seed": args.seed,
            "checkpoint_selection": "minimum validation combined loss",
        },
        "reference": {
            "paper": "https://arxiv.org/abs/2105.05537",
            "official_code": "https://github.com/HuCaoFighting/Swin-Unet",
        },
        "asset_provenance": {
            "supplied_zip_sha256": OFFICIAL_ZIP_SHA256,
            "supplied_checkpoint_sha256": OFFICIAL_PRETRAIN_SHA256,
            "third_party_source_or_weights_committed_to_git": False,
        },
        "fidelity_note": (
            "The model is imported directly from the user-supplied official source. "
            "Input resolution, architecture YAML, encoder/decoder initialization, "
            "augmentation, optimizer, loss, schedule, epochs, and seed are retained."
        ),
    }
    run_official_experiment(
        args,
        model,
        datasets,
        optimizer,
        swin_official_loss,
        metadata,
        scheduler=scheduler,
        scheduler_step="batch",
        val_every_epochs=1,
        selection_metric="loss",
        selection_mode="min",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="swin_unet_official_224_e150_bs6")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--val-batch-size", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=0.0125)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--swin-source-dir",
        default="/share/home/u2515283028/caries_project/external_models/Swin-Unet-main",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        default=(
            "/share/home/u2515283028/caries_project/external_assets/"
            "swin_tiny_patch4_window7_224.pth"
        ),
    )
    run_cli_with_failure_report(main, parser.parse_args())
