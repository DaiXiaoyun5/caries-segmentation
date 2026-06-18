#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import re
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")

NNUNET_BASE = PROJECT_ROOT / "nnunet_caries"
NNUNET_RAW = NNUNET_BASE / "nnUNet_raw"
NNUNET_PREPROCESSED = NNUNET_BASE / "nnUNet_preprocessed"
NNUNET_RESULTS = NNUNET_BASE / "nnUNet_results"

DATASET_ID = 701
DATASET_NAME = "CariXray"
DATASET_FOLDER_NAME = f"Dataset{DATASET_ID:03d}_{DATASET_NAME}"
DATASET_DIR = NNUNET_RAW / DATASET_FOLDER_NAME

SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
TRAIN_SPLIT = SPLIT_DIR / "train.txt"
VAL_SPLIT = SPLIT_DIR / "val.txt"
TEST_SPLIT = SPLIT_DIR / "test.txt"

TEST_GT_DIR = NNUNET_BASE / "test_gt" / DATASET_FOLDER_NAME


def sanitize_case_id(stem: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", stem)
    stem = re.sub(r"_+", "_", stem)
    return stem.strip("_")


def read_split(path: Path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_path, mask_path = line.split("\t")
            items.append((Path(img_path), Path(mask_path)))
    return items


def save_image_png_gray(src_img: Path, dst_png: Path):
    """
    nnU-Net 当前 dataset.json 只声明 1 个 channel。
    所以这里必须保存为灰度 L 图，而不是 RGB。
    """
    img = Image.open(src_img).convert("L")
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst_png)


def save_mask_png(src_mask: Path, dst_png: Path):
    mask = Image.open(src_mask).convert("L")
    arr = np.array(mask)
    arr = (arr > 0).astype(np.uint8)
    dst_png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(dst_png)


def copy_items(items, split_name, images_dir, labels_dir=None, test_gt_dir=None):
    ids = []
    mapping = []
    used = set()

    for idx, (img_path, mask_path) in enumerate(items):
        base_id = sanitize_case_id(img_path.stem)
        case_id = base_id
        if case_id in used:
            case_id = f"{base_id}_{idx:05d}"
        used.add(case_id)

        img_out = images_dir / f"{case_id}_0000.png"
        save_image_png_gray(img_path, img_out)

        if labels_dir is not None:
            mask_out = labels_dir / f"{case_id}.png"
            save_mask_png(mask_path, mask_out)

        if test_gt_dir is not None:
            gt_out = test_gt_dir / f"{case_id}.png"
            save_mask_png(mask_path, gt_out)

        ids.append(case_id)
        mapping.append({
            "case_id": case_id,
            "split": split_name,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        })

        if (idx + 1) % 500 == 0:
            print(f"[{split_name}] converted {idx + 1}/{len(items)}")

    return ids, mapping


def main():
    print("=" * 80)
    print("Preparing nnU-Net v2 dataset as GRAYSCALE single-channel PNG")
    print("Dataset:", DATASET_FOLDER_NAME)
    print("=" * 80)

    for d in [NNUNET_RAW, NNUNET_PREPROCESSED, NNUNET_RESULTS, TEST_GT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    if DATASET_DIR.exists():
        print("Removing old raw dataset:", DATASET_DIR)
        shutil.rmtree(DATASET_DIR)

    # 预处理也清掉，避免上次 RGB 失败残留
    old_preprocessed = NNUNET_PREPROCESSED / DATASET_FOLDER_NAME
    if old_preprocessed.exists():
        print("Removing old preprocessed dataset:", old_preprocessed)
        shutil.rmtree(old_preprocessed)

    imagesTr = DATASET_DIR / "imagesTr"
    labelsTr = DATASET_DIR / "labelsTr"
    imagesTs = DATASET_DIR / "imagesTs"

    imagesTr.mkdir(parents=True, exist_ok=True)
    labelsTr.mkdir(parents=True, exist_ok=True)
    imagesTs.mkdir(parents=True, exist_ok=True)

    if TEST_GT_DIR.exists():
        shutil.rmtree(TEST_GT_DIR)
    TEST_GT_DIR.mkdir(parents=True, exist_ok=True)

    train_items = read_split(TRAIN_SPLIT)
    val_items = read_split(VAL_SPLIT)
    test_items = read_split(TEST_SPLIT)

    print(f"train = {len(train_items)}")
    print(f"val   = {len(val_items)}")
    print(f"test  = {len(test_items)}")

    train_ids, train_mapping = copy_items(train_items, "train", imagesTr, labelsTr)
    val_ids, val_mapping = copy_items(val_items, "val", imagesTr, labelsTr)
    test_ids, test_mapping = copy_items(test_items, "test", imagesTs, labels_dir=None, test_gt_dir=TEST_GT_DIR)

    dataset_json = {
        "channel_names": {
            "0": "grayscale"
        },
        "labels": {
            "background": 0,
            "caries": 1
        },
        "numTraining": len(train_ids) + len(val_ids),
        "file_ending": ".png"
    }

    with open(DATASET_DIR / "dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, ensure_ascii=False, indent=2)

    meta = {
        "dataset_id": DATASET_ID,
        "dataset_name": DATASET_NAME,
        "dataset_folder_name": DATASET_FOLDER_NAME,
        "nnUNet_raw": str(NNUNET_RAW),
        "nnUNet_preprocessed": str(NNUNET_PREPROCESSED),
        "nnUNet_results": str(NNUNET_RESULTS),
        "raw_dataset_dir": str(DATASET_DIR),
        "test_gt_dir": str(TEST_GT_DIR),
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "mapping": train_mapping + val_mapping + test_mapping,
    }

    with open(NNUNET_BASE / "caries_nnunet_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("nnU-Net dataset prepared.")
    print("imagesTr:", len(list(imagesTr.glob("*_0000.png"))))
    print("labelsTr:", len(list(labelsTr.glob("*.png"))))
    print("imagesTs:", len(list(imagesTs.glob("*_0000.png"))))
    print("dataset.json:", DATASET_DIR / "dataset.json")
    print("meta:", NNUNET_BASE / "caries_nnunet_meta.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
