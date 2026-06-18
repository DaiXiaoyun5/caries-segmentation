from pathlib import Path
from PIL import Image, ImageDraw

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
IMAGES_DIR = PROJECT_ROOT / "data" / "unzip" / "images"
LABELS_DIR = PROJECT_ROOT / "data" / "unzip" / "labels"
MASK_DIR = PROJECT_ROOT / "data" / "masks" / "all"

IMAGE_SUFFIX = ".jpg"
LABEL_SUFFIX = ".txt"

def make_mask(image_path: Path, label_path: Path, mask_path: Path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size

    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line_idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) < 7:
            continue

        coords = parts[1:]
        if len(coords) % 2 != 0:
            continue

        coords = [float(x) for x in coords]
        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * (w - 1)
            y = coords[i + 1] * (h - 1)
            points.append((x, y))

        draw.polygon(points, outline=255, fill=255)

    mask.save(mask_path)

def main():
    MASK_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(LABELS_DIR.glob(f"*{LABEL_SUFFIX}"))
    total = len(label_files)
    ok = 0
    missing = 0

    for idx, label_path in enumerate(label_files, start=1):
        stem = label_path.stem
        image_path = IMAGES_DIR / f"{stem}{IMAGE_SUFFIX}"
        mask_path = MASK_DIR / f"{stem}.png"

        if not image_path.exists():
            print(f"[Missing image] {image_path}")
            missing += 1
            continue

        make_mask(image_path, label_path, mask_path)
        ok += 1

        if idx % 200 == 0 or idx == total:
            print(f"[{idx}/{total}] processed, success={ok}, missing={missing}")

    print("=" * 60)
    print(f"Done. success={ok}, missing={missing}")
    print(f"Masks saved to: {MASK_DIR}")
    print("=" * 60)

if __name__ == "__main__":
    main()
