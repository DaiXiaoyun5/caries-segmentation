from pathlib import Path
from PIL import Image, ImageDraw
import argparse

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
IMAGES_DIR = PROJECT_ROOT / "data" / "unzip" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "unzip" / "labels"
MASK_DIR = PROJECT_ROOT / "data" / "masks" / "debug"
OVERLAY_DIR = PROJECT_ROOT / "data" / "overlay" / "debug"

IMAGE_SUFFIX = ".jpg"
LABEL_SUFFIX = ".txt"

def polygon_txt_to_mask(image_path: Path, label_path: Path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line_idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 7:
            print(f"[Warning] line {line_idx} too short, skipped.")
            continue

        # parts[0] 是类别号，这里单类任务先忽略类别
        coords = parts[1:]

        if len(coords) % 2 != 0:
            print(f"[Warning] line {line_idx} has odd number of coordinates, skipped.")
            continue

        coords = [float(x) for x in coords]
        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * (w - 1)
            y = coords[i + 1] * (h - 1)
            points.append((x, y))

        # 填充多边形
        draw.polygon(points, outline=255, fill=255)

    return image, mask

def make_overlay(image: Image.Image, mask: Image.Image):
    base = image.convert("RGBA")
    red_layer = Image.new("RGBA", base.size, (255, 0, 0, 0))

    alpha = mask.point(lambda p: 120 if p > 0 else 0)
    red_layer.putalpha(alpha)

    overlay = Image.alpha_composite(base, red_layer).convert("RGB")
    return overlay

def main(stem: str):
    image_path = IMAGES_DIR / f"{stem}{IMAGE_SUFFIX}"
    label_path = LABELS_DIR / f"{stem}{LABEL_SUFFIX}"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label not found: {label_path}")

    MASK_DIR.mkdir(parents=True, exist_ok=True)
    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    image, mask = polygon_txt_to_mask(image_path, label_path)
    overlay = make_overlay(image, mask)

    mask_path = MASK_DIR / f"{stem}.png"
    overlay_path = OVERLAY_DIR / f"{stem}_overlay.jpg"

    mask.save(mask_path)
    overlay.save(overlay_path)

    print(f"Image size   : {image.size}")
    print(f"Mask saved   : {mask_path}")
    print(f"Overlay saved: {overlay_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stem", type=str, required=True, help="file stem, e.g. train_crawling_1632")
    args = parser.parse_args()
    main(args.stem)
