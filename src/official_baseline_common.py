#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Shared data, metric, and result-export code for faithful external baselines.

Only the CariXray split and reporting convention are shared here.  Model
architecture, input size, augmentation, optimizer, loss, scheduler, training
budget, and checkpoint rule remain under the control of each method-specific
script.  This is intentionally different from ``train_comparison_baseline_common``
which applies the project's controlled B30 protocol to every model.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import random
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TVF


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
METRIC_KEYS = (
    "loss",
    "dice",
    "iou",
    "mean_iou",
    "precision",
    "recall",
    "specificity",
    "accuracy",
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def save_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def _runtime_snapshot() -> Dict[str, Any]:
    """Small, non-secret runtime snapshot for reproducible failure reports."""
    cuda_error = None
    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception as exc:
        cuda_available = False
        cuda_error = f"{type(exc).__name__}: {exc}"
    gpu_name = None
    if cuda_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unavailable"
    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_NODELIST",
        "CUDA_VISIBLE_DEVICES",
        "CONDA_DEFAULT_ENV",
    )
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_probe_error": cuda_error,
        "gpu_name": gpu_name,
        "slurm": {key: os.environ.get(key) for key in slurm_keys},
    }


def save_run_status(
    run_name: str,
    status: str,
    stage: str,
    **details: Any,
) -> None:
    payload = {
        "run_name": str(run_name),
        "status": str(status),
        "stage": str(stage),
        "runtime": _runtime_snapshot(),
    }
    payload.update(details)
    save_json(payload, PROJECT_ROOT / "runs" / str(run_name) / "run_status.json")


def run_cli_with_failure_report(
    entrypoint: Callable[[Any], None],
    args: Any,
) -> None:
    """Run a CLI entry point and preserve its traceback beside config.json."""
    run_name = str(getattr(args, "run_name", "unknown_official_baseline"))
    run_dir = PROJECT_ROOT / "runs" / run_name
    diagnostic_name = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in run_name
    )
    diagnostic_path = (
        PROJECT_ROOT
        / "diagnostics"
        / "official_baselines"
        / f"{diagnostic_name}_failure.json"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in ("failure_report.json", "failure_traceback.txt"):
        stale_path = run_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    if diagnostic_path.exists():
        diagnostic_path.unlink()
    try:
        save_run_status(run_name, "running", "entrypoint_started")
        entrypoint(args)
    except BaseException as exc:
        traceback_text = traceback.format_exc()
        status_path = run_dir / "run_status.json"
        last_completed_stage = None
        if status_path.exists():
            try:
                last_completed_stage = json.loads(
                    status_path.read_text(encoding="utf-8")
                ).get("stage")
            except Exception:
                last_completed_stage = "status_file_unreadable"
        report = {
            "run_name": run_name,
            "status": "failed",
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "last_completed_stage": last_completed_stage,
            "traceback": traceback_text,
            "runtime": _runtime_snapshot(),
        }
        save_json(report, run_dir / "failure_report.json")
        save_json(report, diagnostic_path)
        (run_dir / "failure_traceback.txt").write_text(
            traceback_text,
            encoding="utf-8",
        )
        save_run_status(
            run_name,
            "failed",
            "exception_caught",
            exception_type=type(exc).__name__,
            exception_message=str(exc),
            last_completed_stage=last_completed_stage,
        )
        raise


def _resize_pair(
    image: Image.Image,
    mask: Image.Image,
    size: int,
    image_interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> Tuple[Image.Image, Image.Image]:
    output_size = [int(size), int(size)]
    image = TVF.resize(image, output_size, interpolation=image_interpolation)
    mask = TVF.resize(mask, output_size, interpolation=InterpolationMode.NEAREST)
    return image, mask


def _random_scale_crop_pair(
    image: Image.Image,
    mask: Image.Image,
    size: int,
    scale_range: Tuple[float, float] = (0.5, 2.0),
) -> Tuple[Image.Image, Image.Image]:
    """Random resize followed by crop/pad, matching common dense-prediction recipes."""
    image, mask = _resize_pair(image, mask, size)
    scale = random.uniform(*scale_range)
    scaled_size = max(1, int(round(size * scale)))
    image, mask = _resize_pair(image, mask, scaled_size)

    if scaled_size < size:
        pad_w = size - scaled_size
        pad_h = size - scaled_size
        left = random.randint(0, pad_w)
        top = random.randint(0, pad_h)
        border = (left, top, pad_w - left, pad_h - top)
        image = ImageOps.expand(image, border=border, fill=0)
        mask = ImageOps.expand(mask, border=border, fill=0)
    elif scaled_size > size:
        left = random.randint(0, scaled_size - size)
        top = random.randint(0, scaled_size - size)
        box = (left, top, left + size, top + size)
        image = image.crop(box)
        mask = mask.crop(box)

    return image, mask


def _augment_attention_unet(
    image: Image.Image,
    mask: Image.Image,
    size: int,
) -> Tuple[Image.Image, Image.Image]:
    """2D counterpart of the official affine/flip/random-crop CT recipe."""
    image, mask = _resize_pair(image, mask, size)
    angle = random.uniform(-15.0, 15.0)
    scale = random.uniform(0.7, 1.3)
    max_translate = int(round(0.10 * size))
    translate = (
        random.randint(-max_translate, max_translate),
        random.randint(-max_translate, max_translate),
    )
    image = TVF.affine(
        image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )
    mask = TVF.affine(
        mask,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0,
    )
    if random.random() < 0.5:
        image = TVF.hflip(image)
        mask = TVF.hflip(mask)
    return image, mask


def _augment_unetpp(
    image: Image.Image,
    mask: Image.Image,
    size: int,
) -> Tuple[Image.Image, Image.Image]:
    """Conservative, widely used 2D medical augmentation for native UNet++."""
    image, mask = _resize_pair(image, mask, size)
    angle = random.uniform(-15.0, 15.0)
    scale = random.uniform(0.85, 1.15)
    max_translate = int(round(0.05 * size))
    translate = (
        random.randint(-max_translate, max_translate),
        random.randint(-max_translate, max_translate),
    )
    image = TVF.affine(
        image,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.BILINEAR,
        fill=0,
    )
    mask = TVF.affine(
        mask,
        angle=angle,
        translate=translate,
        scale=scale,
        shear=[0.0, 0.0],
        interpolation=InterpolationMode.NEAREST,
        fill=0,
    )
    if random.random() < 0.5:
        image = TVF.hflip(image)
        mask = TVF.hflip(mask)
    return image, mask


def _augment_scale_crop_flip(
    image: Image.Image,
    mask: Image.Image,
    size: int,
    photometric: bool,
) -> Tuple[Image.Image, Image.Image]:
    image, mask = _random_scale_crop_pair(image, mask, size, (0.5, 2.0))
    if random.random() < 0.5:
        image = TVF.hflip(image)
        mask = TVF.hflip(mask)
    if photometric:
        if random.random() < 0.5:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.15))
    return image, mask


def _augment_swin_unet(
    image: Image.Image,
    mask: Image.Image,
    size: int,
) -> Tuple[Image.Image, Image.Image]:
    """Faithful PIL equivalent of Swin-Unet RandomGenerator."""
    transpose = getattr(Image, "Transpose", Image)
    if random.random() > 0.5:
        k = random.randint(0, 3)
        if k == 1:
            image = image.transpose(transpose.ROTATE_90)
            mask = mask.transpose(transpose.ROTATE_90)
        elif k == 2:
            image = image.transpose(transpose.ROTATE_180)
            mask = mask.transpose(transpose.ROTATE_180)
        elif k == 3:
            image = image.transpose(transpose.ROTATE_270)
            mask = mask.transpose(transpose.ROTATE_270)
        if random.randint(0, 1) == 0:
            image = TVF.vflip(image)
            mask = TVF.vflip(mask)
        else:
            image = TVF.hflip(image)
            mask = TVF.hflip(mask)
    elif random.random() > 0.5:
        angle = random.randrange(-20, 20)
        # The official scipy implementation deliberately uses nearest order for
        # both image and label in this branch.
        image = TVF.rotate(
            image,
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        mask = TVF.rotate(
            mask,
            angle,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
    return _resize_pair(image, mask, size, InterpolationMode.BICUBIC)


class CariesOfficialDataset(Dataset):
    """CariXray split adapter with an explicitly named method recipe."""

    def __init__(
        self,
        split_file: Path,
        image_size: int,
        channels: int,
        train: bool,
        augmentation: str,
        normalization: str = "zero_one",
    ) -> None:
        self.split_file = Path(split_file)
        self.image_size = int(image_size)
        self.channels = int(channels)
        self.train = bool(train)
        self.augmentation = str(augmentation)
        self.normalization = str(normalization)
        self.items = []

        if self.channels not in (1, 3):
            raise ValueError(f"channels must be 1 or 3, got {self.channels}")
        if self.normalization not in ("zero_one", "per_image_zscore"):
            raise ValueError(f"Unsupported normalization: {self.normalization}")

        with open(self.split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                image_path, mask_path = line.split("\t")
                self.items.append((Path(image_path), Path(mask_path)))
        if not self.items:
            raise RuntimeError(f"No samples found in {self.split_file}")

    def __len__(self) -> int:
        return len(self.items)

    def _transform_pair(
        self,
        image: Image.Image,
        mask: Image.Image,
    ) -> Tuple[Image.Image, Image.Image]:
        if not self.train or self.augmentation == "none":
            return _resize_pair(image, mask, self.image_size)
        if self.augmentation == "attention_unet_official":
            return _augment_attention_unet(image, mask, self.image_size)
        if self.augmentation == "unetpp_medical":
            return _augment_unetpp(image, mask, self.image_size)
        if self.augmentation == "segformer_official":
            return _augment_scale_crop_flip(
                image, mask, self.image_size, photometric=True
            )
        if self.augmentation == "deeplab_official":
            return _augment_scale_crop_flip(
                image, mask, self.image_size, photometric=False
            )
        if self.augmentation == "swin_unet_official":
            return _augment_swin_unet(image, mask, self.image_size)
        raise ValueError(f"Unsupported augmentation recipe: {self.augmentation}")

    def __getitem__(self, index: int):
        image_path, mask_path = self.items[index]
        image_mode = "L" if self.channels == 1 else "RGB"
        image = Image.open(image_path).convert(image_mode)
        mask = Image.open(mask_path).convert("L")
        image, mask = self._transform_pair(image, mask)

        image_np = np.asarray(image, dtype=np.float32) / 255.0
        if self.channels == 1:
            image_np = image_np[..., None]
        if self.normalization == "per_image_zscore":
            mean = image_np.mean(axis=(0, 1), keepdims=True)
            std = image_np.std(axis=(0, 1), keepdims=True)
            image_np = (image_np - mean) / np.maximum(std, 1e-6)

        mask_np = (np.asarray(mask, dtype=np.float32) > 0).astype(np.float32)
        image_tensor = torch.from_numpy(image_np.copy()).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_np.copy()).unsqueeze(0).float()
        return image_tensor, mask_tensor, str(image_path), str(mask_path)


def make_dataset_bundle(
    image_size: int,
    channels: int,
    train_augmentation: str,
    normalization: str = "zero_one",
) -> Dict[str, Dataset]:
    split_dir = PROJECT_ROOT / "data/splits"
    shared = {
        "image_size": image_size,
        "channels": channels,
        "normalization": normalization,
    }
    return {
        "train": CariesOfficialDataset(
            split_dir / "train.txt",
            train=True,
            augmentation=train_augmentation,
            **shared,
        ),
        "train_eval": CariesOfficialDataset(
            split_dir / "train.txt",
            train=False,
            augmentation="none",
            **shared,
        ),
        "val": CariesOfficialDataset(
            split_dir / "val.txt",
            train=False,
            augmentation="none",
            **shared,
        ),
        "test": CariesOfficialDataset(
            split_dir / "test.txt",
            train=False,
            augmentation="none",
            **shared,
        ),
    }


def binary_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dims)
    denominator = probs.sum(dims) + targets.sum(dims)
    score = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    return 1.0 - score.mean()


def binary_bce_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, targets) + binary_dice_loss(
        logits, targets
    )


def multiclass_dice_loss_official_swin(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """The DiceLoss formula from the supplied official Swin-Unet repository."""
    probs = torch.softmax(logits, dim=1)
    target_labels = targets[:, 0].long()
    one_hot = F.one_hot(target_labels, num_classes=num_classes).permute(0, 3, 1, 2)
    one_hot = one_hot.float()
    loss = logits.new_tensor(0.0)
    for class_index in range(num_classes):
        score = probs[:, class_index]
        target = one_hot[:, class_index]
        intersect = torch.sum(score * target)
        score_sum = torch.sum(score * score)
        target_sum = torch.sum(target * target)
        dice = (2.0 * intersect + 1e-5) / (score_sum + target_sum + 1e-5)
        loss = loss + (1.0 - dice)
    return loss / float(num_classes)


def merge_output_logits(output: Any) -> torch.Tensor:
    """Return evaluation logits while preserving UNet++ accurate-mode averaging."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("logits", "out"):
            if key in output:
                return merge_output_logits(output[key])
    if isinstance(output, (list, tuple)) and output:
        tensors = [merge_output_logits(item) for item in output]
        if tensors[0].shape[1] == 1:
            mean_prob = torch.stack([torch.sigmoid(item) for item in tensors]).mean(0)
            mean_prob = mean_prob.clamp(1e-6, 1.0 - 1e-6)
            return torch.logit(mean_prob)
        mean_prob = torch.stack([torch.softmax(item, dim=1) for item in tensors]).mean(0)
        return torch.log(mean_prob.clamp_min(1e-8))
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


def logits_to_foreground(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[1] == 1:
        return (torch.sigmoid(logits) > 0.5).float()
    return (torch.argmax(logits, dim=1, keepdim=True) == 1).float()


def metrics_from_confusion(
    loss_sum: float,
    batches: int,
    tp: float,
    fp: float,
    fn: float,
    tn: float,
) -> Dict[str, float]:
    eps = 1e-6
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    fg_iou = (tp + eps) / (tp + fp + fn + eps)
    bg_iou = (tn + eps) / (tn + fp + fn + eps)
    return {
        "loss": loss_sum / max(batches, 1),
        "dice": dice,
        "iou": fg_iou,
        "mean_iou": 0.5 * (fg_iou + bg_iou),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall": (tp + eps) / (tp + fn + eps),
        "specificity": (tn + eps) / (tn + fp + eps),
        "accuracy": (tp + tn + eps) / (tp + tn + fp + fn + eps),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def load_masks_for_metric(
    mask_paths: Sequence[str],
    metric_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resize GT directly from source files, avoiding an input-size round trip."""
    targets = []
    for mask_path in mask_paths:
        with Image.open(mask_path) as mask_image:
            mask_image = mask_image.convert("L").resize(
                (int(metric_size), int(metric_size)),
                resample=Image.Resampling.NEAREST
                if hasattr(Image, "Resampling")
                else Image.NEAREST,
            )
            target = (np.asarray(mask_image, dtype=np.float32) > 0).astype(np.float32)
        targets.append(torch.from_numpy(target).unsqueeze(0))
    return torch.stack(targets, dim=0).to(device=device, dtype=torch.float32)


class IterationPolynomialLR:
    """Small dependency-free poly scheduler with optional linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_steps: int,
        power: float,
        warmup_steps: int = 0,
        warmup_ratio: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.total_steps = max(int(total_steps), 1)
        self.power = float(power)
        self.warmup_steps = max(int(warmup_steps), 0)
        self.warmup_ratio = float(warmup_ratio)
        self.base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        self.step_num = 0
        self._apply()

    def _factor(self) -> float:
        if self.warmup_steps > 0 and self.step_num < self.warmup_steps:
            alpha = self.step_num / float(self.warmup_steps)
            return self.warmup_ratio + alpha * (1.0 - self.warmup_ratio)
        progress = min(self.step_num / float(self.total_steps), 1.0)
        return max(0.0, (1.0 - progress) ** self.power)

    def _apply(self) -> None:
        factor = self._factor()
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * factor

    def step(self) -> None:
        self.step_num += 1
        self._apply()

    def get_last_lr(self) -> Sequence[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "power": self.power,
            "warmup_steps": self.warmup_steps,
            "warmup_ratio": self.warmup_ratio,
            "base_lrs": self.base_lrs,
            "step_num": self.step_num,
        }


def _build_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    drop_last: bool = False,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        num_workers=int(num_workers),
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        generator=generator,
    )


def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scheduler_step: str = "none",
    max_updates: Optional[int] = None,
    metric_size: Optional[int] = None,
) -> Tuple[Dict[str, float], int]:
    train = optimizer is not None
    model.train(train)
    loss_sum = 0.0
    batches = 0
    updates = 0
    tp = fp = fn = tn = 0.0

    for images, masks, _, mask_paths in loader:
        if max_updates is not None and updates >= max_updates:
            break
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True).float()
        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            output = model(images)
            loss = loss_fn(output, masks)
            if train:
                loss.backward()
                optimizer.step()
                updates += 1
                if scheduler is not None and scheduler_step == "batch":
                    scheduler.step()

        with torch.no_grad():
            logits = merge_output_logits(output)
            if metric_size is not None and logits.shape[-2:] != (
                int(metric_size),
                int(metric_size),
            ):
                logits = F.interpolate(
                    logits,
                    size=(int(metric_size), int(metric_size)),
                    mode="bilinear",
                    align_corners=False,
                )
                gt = load_masks_for_metric(mask_paths, metric_size, device)
            else:
                gt = (masks > 0.5).float()
            preds = logits_to_foreground(logits)
            tp += float((preds * gt).sum().cpu())
            fp += float((preds * (1.0 - gt)).sum().cpu())
            fn += float(((1.0 - preds) * gt).sum().cpu())
            tn += float(((1.0 - preds) * (1.0 - gt)).sum().cpu())
        loss_sum += float(loss.detach().cpu())
        batches += 1

    return metrics_from_confusion(loss_sum, batches, tp, fp, fn, tn), updates


def _checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    global_step: int,
    selection_value: float,
) -> Dict[str, Any]:
    value = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "selection_value": float(selection_value),
    }
    if scheduler is not None and hasattr(scheduler, "state_dict"):
        value["scheduler_state_dict"] = scheduler.state_dict()
    return value


def _load_model_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)


@torch.no_grad()
def save_prediction_panel(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    path: Path,
    max_items: int = 4,
) -> None:
    model.eval()
    images, masks, _, _ = next(iter(loader))
    logits = merge_output_logits(model(images.to(device)))
    preds = logits_to_foreground(logits).cpu()
    images = images.cpu()
    masks = masks.cpu()
    count = min(int(images.shape[0]), int(max_items))
    fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)
    for index in range(count):
        image = images[index]
        if image.shape[0] == 1:
            display = image[0].numpy()
            display = (display - display.min()) / max(
                float(display.max() - display.min()), 1e-6
            )
            rgb = np.repeat(display[..., None], 3, axis=2)
        else:
            rgb = image.permute(1, 2, 0).numpy()
            rgb = np.clip(rgb, 0.0, 1.0)
        gt = masks[index, 0].numpy()
        pred = preds[index, 0].numpy()
        overlay = rgb.copy()
        overlay[..., 0] = np.where(pred > 0, 1.0, overlay[..., 0])
        overlay[..., 1] = np.where(pred > 0, 0.0, overlay[..., 1])
        overlay[..., 2] = np.where(pred > 0, 0.0, overlay[..., 2])
        overlay = 0.6 * rgb + 0.4 * overlay
        for column, (value, title, cmap) in enumerate(
            (
                (rgb, "Image", None),
                (gt, "GT", "gray"),
                (pred, "Prediction", "gray"),
                (overlay, "Overlay", None),
            )
        ):
            axes[index, column].imshow(value, cmap=cmap)
            axes[index, column].set_title(title)
            axes[index, column].axis("off")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


@torch.no_grad()
def save_per_case_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    csv_path: Path,
    summary_path: Path,
    metric_size: Optional[int] = None,
) -> None:
    model.eval()
    rows = []
    for images, masks, image_paths, mask_paths in loader:
        logits = merge_output_logits(model(images.to(device)))
        if metric_size is not None and logits.shape[-2:] != (
            int(metric_size),
            int(metric_size),
        ):
            logits = F.interpolate(
                logits,
                size=(int(metric_size), int(metric_size)),
                mode="bilinear",
                align_corners=False,
            )
            gt = load_masks_for_metric(mask_paths, metric_size, torch.device("cpu"))
        else:
            gt = (masks > 0.5).float()
        preds = logits_to_foreground(logits).cpu()
        for index in range(images.shape[0]):
            pred = preds[index : index + 1]
            target = gt[index : index + 1]
            tp = float((pred * target).sum())
            fp = float((pred * (1.0 - target)).sum())
            fn = float(((1.0 - pred) * target).sum())
            tn = float(((1.0 - pred) * (1.0 - target)).sum())
            metrics = metrics_from_confusion(0.0, 1, tp, fp, fn, tn)
            rows.append(
                {
                    "image_path": image_paths[index],
                    "mask_path": mask_paths[index],
                    "foreground_pixels": int(target.sum()),
                    **{key: metrics[key] for key in METRIC_KEYS if key != "loss"},
                }
            )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    metric_names = [key for key in METRIC_KEYS if key not in ("loss",)]
    summary = {"num_cases": len(rows)}
    for key in metric_names:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "median": float(np.median(values)),
        }
    save_json(summary, summary_path)


def run_official_experiment(
    args: Any,
    model: nn.Module,
    datasets: Dict[str, Dataset],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Any, torch.Tensor], torch.Tensor],
    metadata: Dict[str, Any],
    scheduler: Optional[Any] = None,
    scheduler_step: str = "none",
    max_iterations: Optional[int] = None,
    val_interval_iterations: Optional[int] = None,
    val_every_epochs: int = 1,
    selection_metric: str = "dice",
    selection_mode: str = "max",
    early_stopping_patience: int = 0,
    drop_last_train: bool = False,
    final_metric_size: Optional[int] = 512,
) -> None:
    """Train one method and export the project-standard evidence files."""
    if selection_mode not in ("max", "min"):
        raise ValueError("selection_mode must be 'max' or 'min'")
    if scheduler_step not in ("none", "batch", "epoch"):
        raise ValueError("scheduler_step must be none, batch, or epoch")

    set_seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    run_dir = PROJECT_ROOT / "runs" / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    prediction_dir = run_dir / "preds"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    save_run_status(args.run_name, "running", "runner_initialized", device=str(device))

    val_batch_size = int(getattr(args, "val_batch_size", args.batch_size))
    train_loader = _build_loader(
        datasets["train"],
        args.batch_size,
        True,
        args.num_workers,
        args.seed,
        drop_last_train,
    )
    train_eval_loader = _build_loader(
        datasets["train_eval"],
        val_batch_size,
        False,
        args.num_workers,
        args.seed,
    )
    val_loader = _build_loader(
        datasets["val"],
        val_batch_size,
        False,
        args.num_workers,
        args.seed,
    )
    test_loader = _build_loader(
        datasets["test"],
        val_batch_size,
        False,
        args.num_workers,
        args.seed,
    )

    # Keep CLI values at the top level for compatibility with the project's
    # existing analysis scripts, while also retaining an explicit snapshot.
    config = vars(args).copy()
    config.update(metadata)
    config["arguments"] = vars(args).copy()
    config["data_and_evaluation"] = {
        "split_files": "data/splits/{train,val,test}.txt",
        "split_sizes": {
            key: len(datasets[key]) for key in ("train", "val", "test")
        },
        "shared_across_methods": [
            "train/validation/test case split",
            "foreground threshold or class argmax",
            "global TP/FP/FN/TN metric aggregation",
            "best-checkpoint test evaluation",
        ],
        "method_specific_training": metadata.get("training_recipe", {}),
        "final_metric_resolution": (
            None
            if final_metric_size is None
            else [int(final_metric_size), int(final_metric_size)]
        ),
        "metric_resolution_note": (
            "Final best-train/validation/test and per-case metrics use GT resized "
            "directly from source masks to this common resolution. Training-time "
            "history and checkpoint loss remain at each method's native input size."
        ),
    }
    save_json(config, run_dir / "config.json")
    save_run_status(args.run_name, "running", "config_saved", device=str(device))

    sample_images, _, sample_image_paths, sample_mask_paths = next(iter(train_loader))
    save_run_status(
        args.run_name,
        "running",
        "first_batch_loaded",
        device=str(device),
        batch_shape=list(sample_images.shape),
        first_image=sample_image_paths[0],
        first_mask=sample_mask_paths[0],
    )
    model.eval()
    with torch.no_grad():
        sample_logits = merge_output_logits(model(sample_images[:1].to(device)))
    save_run_status(
        args.run_name,
        "running",
        "forward_check_completed",
        device=str(device),
        input_shape=list(sample_images[:1].shape),
        output_shape=list(sample_logits.shape),
    )
    if sample_logits.shape[-2:] != sample_images.shape[-2:]:
        raise RuntimeError(
            f"Model output {tuple(sample_logits.shape)} does not match input "
            f"{tuple(sample_images[:1].shape)}"
        )
    if sample_logits.shape[1] not in (1, 2):
        raise RuntimeError(
            f"Binary task expects one foreground logit or two class logits, got "
            f"{sample_logits.shape[1]} channels"
        )

    print("=" * 80)
    print("run_name:", args.run_name)
    print("model:", metadata.get("model_name"))
    print("device:", device)
    print("split sizes:", len(datasets["train"]), len(datasets["val"]), len(datasets["test"]))
    print("input/output:", tuple(sample_images[:1].shape), tuple(sample_logits.shape))
    print("checkpoint rule:", selection_mode, selection_metric)
    print("=" * 80)

    fieldnames = ["epoch", "global_step", "lr"]
    fieldnames += [f"train_{key}" for key in METRIC_KEYS]
    fieldnames += [f"val_{key}" for key in METRIC_KEYS]
    history_path = run_dir / "history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()
    save_run_status(args.run_name, "running", "training_started", device=str(device))

    best_value = -math.inf if selection_mode == "max" else math.inf
    best_epoch = -1
    best_global_step = -1
    validation_events_without_improvement = 0
    global_step = 0
    next_val_step = int(val_interval_iterations or 0)
    stopped_early = False

    for epoch in range(1, int(args.epochs) + 1):
        remaining = None
        if max_iterations is not None:
            remaining = int(max_iterations) - global_step
            if remaining <= 0:
                break
        train_metrics, updates = run_one_epoch(
            model,
            train_loader,
            loss_fn,
            device,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_step=scheduler_step,
            max_updates=remaining,
        )
        global_step += updates
        if scheduler is not None and scheduler_step == "epoch":
            scheduler.step()

        reached_end = max_iterations is not None and global_step >= max_iterations
        if val_interval_iterations:
            should_validate = global_step >= next_val_step or reached_end
        else:
            should_validate = epoch % max(int(val_every_epochs), 1) == 0
        should_validate = bool(should_validate or epoch == int(args.epochs))

        val_metrics = None
        if should_validate:
            val_metrics, _ = run_one_epoch(
                model, val_loader, loss_fn, device, optimizer=None
            )
            current_value = float(val_metrics[selection_metric])
            improved = (
                current_value > best_value
                if selection_mode == "max"
                else current_value < best_value
            )
            if improved:
                best_value = current_value
                best_epoch = epoch
                best_global_step = global_step
                validation_events_without_improvement = 0
                torch.save(
                    _checkpoint_payload(
                        model,
                        optimizer,
                        scheduler,
                        epoch,
                        global_step,
                        current_value,
                    ),
                    checkpoint_dir / "best.pth",
                )
                print(
                    f"  -> best updated: epoch={epoch}, step={global_step}, "
                    f"{selection_metric}={current_value:.6f}"
                )
            else:
                validation_events_without_improvement += 1
            if val_interval_iterations:
                while next_val_step <= global_step:
                    next_val_step += int(val_interval_iterations)

        last_selection = (
            float(val_metrics[selection_metric]) if val_metrics is not None else best_value
        )
        torch.save(
            _checkpoint_payload(
                model,
                optimizer,
                scheduler,
                epoch,
                global_step,
                last_selection,
            ),
            checkpoint_dir / "last.pth",
        )

        lr = float(optimizer.param_groups[0]["lr"])
        row: Dict[str, Any] = {"epoch": epoch, "global_step": global_step, "lr": lr}
        for key in METRIC_KEYS:
            row[f"train_{key}"] = train_metrics[key]
            row[f"val_{key}"] = "" if val_metrics is None else val_metrics[key]
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writerow(row)

        val_text = "not scheduled"
        if val_metrics is not None:
            val_text = (
                f"loss={val_metrics['loss']:.4f}, dice={val_metrics['dice']:.4f}, "
                f"mIoU={val_metrics['mean_iou']:.4f}"
            )
        print(
            f"[epoch {epoch:04d}/{int(args.epochs):04d} | step {global_step}] "
            f"lr={lr:.6g} train_loss={train_metrics['loss']:.4f} "
            f"train_dice={train_metrics['dice']:.4f} val={val_text}"
        )

        if (
            early_stopping_patience > 0
            and val_metrics is not None
            and validation_events_without_improvement >= early_stopping_patience
        ):
            stopped_early = True
            print(
                f"Early stopping after {early_stopping_patience} validation events "
                "without improvement."
            )
            break
        if reached_end:
            break

    best_path = checkpoint_dir / "best.pth"
    if not best_path.exists():
        raise RuntimeError("No best checkpoint was produced; validation was never run.")
    _load_model_checkpoint(model, best_path, device)
    best_train_metrics, _ = run_one_epoch(
        model,
        train_eval_loader,
        loss_fn,
        device,
        optimizer=None,
        metric_size=final_metric_size,
    )
    best_val_metrics, _ = run_one_epoch(
        model,
        val_loader,
        loss_fn,
        device,
        optimizer=None,
        metric_size=final_metric_size,
    )
    test_metrics, _ = run_one_epoch(
        model,
        test_loader,
        loss_fn,
        device,
        optimizer=None,
        metric_size=final_metric_size,
    )

    save_json(best_train_metrics, run_dir / "best_train_metrics.json")
    save_json(best_val_metrics, run_dir / "best_val_metrics.json")
    save_json(test_metrics, run_dir / "test_metrics.json")
    save_json(
        {
            "best_epoch": best_epoch,
            "best_global_step": best_global_step,
            "checkpoint_selection": {
                "metric": selection_metric,
                "mode": selection_mode,
                "best_value_during_training": best_value,
            },
            "stopped_early": stopped_early,
            "completed_global_steps": global_step,
            "best_train_metrics": best_train_metrics,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
        },
        run_dir / "summary_metrics.json",
    )
    save_prediction_panel(
        model, test_loader, device, prediction_dir / "test_preview.png"
    )
    save_per_case_metrics(
        model,
        test_loader,
        device,
        run_dir / "per_case_metrics.csv",
        run_dir / "per_case_metrics_summary.json",
        metric_size=final_metric_size,
    )
    print("=" * 80)
    print("finished:", args.run_name)
    print("best epoch/step:", best_epoch, best_global_step)
    print("test metrics:", test_metrics)
    print("=" * 80)
    save_run_status(
        args.run_name,
        "completed",
        "result_export_completed",
        device=str(device),
        best_epoch=best_epoch,
        best_global_step=best_global_step,
    )
