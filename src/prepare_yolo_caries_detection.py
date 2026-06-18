import argparse
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")


def read_split(split_path):
    items = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 1:
                continue
            image_path = Path(parts[0])
            items.append(image_path)
    return items


def polygon_label_to_bboxes(src_label_path, dst_label_path):
    """
    CariXray label appears to be YOLO segmentation polygon:
        class x1 y1 x2 y2 ... xn yn

    Convert every polygon to one YOLO detection bbox:
        class x_center y_center width height
    """
    dst_label_path.parent.mkdir(parents=True, exist_ok=True)

    if not src_label_path.exists():
        dst_label_path.write_text("", encoding="utf-8")
        return 0

    lines_out = []
    raw = src_label_path.read_text(encoding="utf-8").strip()
    if not raw:
        dst_label_path.write_text("", encoding="utf-8")
        return 0

    for line in raw.splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls = parts[0]
        vals = [float(x) for x in parts[1:]]

        # segmentation polygon should have x,y pairs
        if len(vals) % 2 != 0:
            vals = vals[:-1]
        if len(vals) < 4:
            continue

        xs = vals[0::2]
        ys = vals[1::2]

        x1 = max(0.0, min(xs))
        y1 = max(0.0, min(ys))
        x2 = min(1.0, max(xs))
        y2 = min(1.0, max(ys))

        w = x2 - x1
        h = y2 - y1
        if w <= 0 or h <= 0:
            continue

        xc = x1 + w / 2.0
        yc = y1 + h / 2.0

        lines_out.append(f"{cls} {xc:.8f} {yc:.8f} {w:.8f} {h:.8f}")

    dst_label_path.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")
    return len(lines_out)


def safe_link_or_copy(src, dst, use_symlink=True):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if use_symlink:
        try:
            os.symlink(src, dst)
            return
        except Exception:
            pass

    shutil.copy2(src, dst)


def main(args):
    out_root = PROJECT_ROOT / args.out_dir
    src_label_dir = PROJECT_ROOT / "data" / "unzip" / "labels"
    split_dir = PROJECT_ROOT / "data" / "splits"

    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("out_root:", out_root)
    print("src_label_dir:", src_label_dir)

    if out_root.exists() and args.overwrite:
        shutil.rmtree(out_root)

    out_root.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_boxes = 0

    for split in ["train", "val", "test"]:
        split_file = split_dir / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(split_file)

        images = read_split(split_file)
        print(f"{split}: {len(images)} images")

        list_file = out_root / f"{split}.txt"
        with open(list_file, "w", encoding="utf-8") as lf:
            for img_path in images:
                if not img_path.exists():
                    raise FileNotFoundError(img_path)

                stem = img_path.stem
                src_label_path = src_label_dir / f"{stem}.txt"

                dst_img_path = out_root / "images" / split / img_path.name
                dst_label_path = out_root / "labels" / split / f"{stem}.txt"

                safe_link_or_copy(img_path, dst_img_path, use_symlink=not args.copy_images)
                n_box = polygon_label_to_bboxes(src_label_path, dst_label_path)

                lf.write(str(dst_img_path.resolve()) + "\n")
                total_images += 1
                total_boxes += n_box

        # write label count sanity file
        n_label_files = len(list((out_root / "labels" / split).glob("*.txt")))
        print(f"{split}: label files written = {n_label_files}")

    yaml_text = f"""# Auto-generated YOLO detection dataset for CariXray caries detection
path: {out_root.resolve()}
train: train.txt
val: val.txt
test: test.txt

names:
  0: caries
"""
    yaml_path = out_root / "data.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    print("Total images:", total_images)
    print("Total boxes :", total_boxes)
    print("Saved YAML  :", yaml_path)

    # show a few converted labels
    print("\n===== sample converted labels =====")
    for p in sorted((out_root / "labels" / "train").glob("*.txt"))[:5]:
        print("----", p)
        print(p.read_text(encoding="utf-8").strip()[:500])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=str, default="data/yolo_caries_detect")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-images", action="store_true", help="copy images instead of symlink")
    args = parser.parse_args()
    main(args)
