from pathlib import Path

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
IMAGES_DIR = PROJECT_ROOT / "data" / "unzip" / "images"
MASKS_DIR = PROJECT_ROOT / "data" / "masks" / "all"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"

IMAGE_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"

def decide_split(stem: str):
    if stem.startswith("train_crawling_") or stem.startswith("train2_crawling_"):
        return "train"
    elif stem.startswith("val_crawling_"):
        return "val"
    elif stem.startswith("test_crawling_"):
        return "test"
    else:
        return None

def main():
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    train_list, val_list, test_list = [], [], []
    unknown = []
    missing_mask = []

    image_files = sorted(IMAGES_DIR.glob(f"*{IMAGE_SUFFIX}"))

    for img_path in image_files:
        stem = img_path.stem
        split = decide_split(stem)
        mask_path = MASKS_DIR / f"{stem}{MASK_SUFFIX}"

        if split is None:
            unknown.append(stem)
            continue

        if not mask_path.exists():
            missing_mask.append(stem)
            continue

        line = f"{img_path}\t{mask_path}\n"

        if split == "train":
            train_list.append(line)
        elif split == "val":
            val_list.append(line)
        elif split == "test":
            test_list.append(line)

    (SPLITS_DIR / "train.txt").write_text("".join(train_list), encoding="utf-8")
    (SPLITS_DIR / "val.txt").write_text("".join(val_list), encoding="utf-8")
    (SPLITS_DIR / "test.txt").write_text("".join(test_list), encoding="utf-8")

    print("=" * 60)
    print(f"train: {len(train_list)}")
    print(f"val  : {len(val_list)}")
    print(f"test : {len(test_list)}")
    print(f"unknown stems    : {len(unknown)}")
    print(f"missing masks    : {len(missing_mask)}")
    print("=" * 60)

    if unknown:
        print("[Unknown examples]")
        for x in unknown[:10]:
            print(x)

    if missing_mask:
        print("[Missing mask examples]")
        for x in missing_mask[:10]:
            print(x)

    print("\nSaved:")
    print(SPLITS_DIR / "train.txt")
    print(SPLITS_DIR / "val.txt")
    print(SPLITS_DIR / "test.txt")

if __name__ == "__main__":
    main()
