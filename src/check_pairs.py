from pathlib import Path

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
IMAGES_DIR = PROJECT_ROOT / "data" / "unzip" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "unzip" / "labels"
OUT_DIR = PROJECT_ROOT / "data"

IMAGE_SUFFIX = ".jpg"
LABEL_SUFFIX = ".txt"

def main():
    image_files = sorted(IMAGES_DIR.glob(f"*{IMAGE_SUFFIX}"))
    label_files = sorted(LABELS_DIR.glob(f"*{LABEL_SUFFIX}"))

    image_stems = {p.stem for p in image_files}
    label_stems = {p.stem for p in label_files}

    common = sorted(image_stems & label_stems)
    only_images = sorted(image_stems - label_stems)
    only_labels = sorted(label_stems - image_stems)

    print("=" * 60)
    print(f"Images total : {len(image_files)}")
    print(f"Labels total : {len(label_files)}")
    print(f"Matched pairs: {len(common)}")
    print(f"Only images  : {len(only_images)}")
    print(f"Only labels  : {len(only_labels)}")
    print("=" * 60)

    if only_images:
        print("\n[Only images] first 10:")
        for x in only_images[:10]:
            print(x)

    if only_labels:
        print("\n[Only labels] first 10:")
        for x in only_labels[:10]:
            print(x)

    # 写出配对清单
    pair_file = OUT_DIR / "matched_pairs.txt"
    with open(pair_file, "w", encoding="utf-8") as f:
        for stem in common:
            img_path = IMAGES_DIR / f"{stem}{IMAGE_SUFFIX}"
            lbl_path = LABELS_DIR / f"{stem}{LABEL_SUFFIX}"
            f.write(f"{img_path}\t{lbl_path}\n")

    print(f"\nMatched pair list saved to: {pair_file}")

if __name__ == "__main__":
    main()
