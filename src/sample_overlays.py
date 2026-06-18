from pathlib import Path
from PIL import Image
import random

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
IMAGES_DIR = PROJECT_ROOT / "data" / "unzip" / "images"
MASK_DIR = PROJECT_ROOT / "data" / "masks" / "all"
OUT_DIR = PROJECT_ROOT / "data" / "overlay" / "sample"

IMAGE_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"

def make_overlay(image: Image.Image, mask: Image.Image):
    base = image.convert("RGBA")
    red_layer = Image.new("RGBA", base.size, (255, 0, 0, 0))
    alpha = mask.point(lambda p: 120 if p > 0 else 0)
    red_layer.putalpha(alpha)
    overlay = Image.alpha_composite(base, red_layer).convert("RGB")
    return overlay

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGES_DIR.glob(f"*{IMAGE_SUFFIX}"))
    stems = [p.stem for p in image_files]
    random.seed(42)
    picks = random.sample(stems, min(20, len(stems)))

    for stem in picks:
        img_path = IMAGES_DIR / f"{stem}{IMAGE_SUFFIX}"
        mask_path = MASK_DIR / f"{stem}{MASK_SUFFIX}"

        if not mask_path.exists():
            print(f"[Missing mask] {stem}")
            continue

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        overlay = make_overlay(image, mask)
        overlay.save(OUT_DIR / f"{stem}_overlay.jpg")

    print(f"Saved sample overlays to: {OUT_DIR}")

if __name__ == "__main__":
    main()
