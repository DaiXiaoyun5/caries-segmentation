from pathlib import Path
import random

import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
IMAGES_DIR = PROJECT_ROOT / "data" / "unzip" / "images"
MASKS_DIR = PROJECT_ROOT / "data" / "masks" / "all"
OUT_DIR = PROJECT_ROOT / "runs" / "debug" / "edge_gt_preview"

IMAGE_SUFFIX = ".jpg"
MASK_SUFFIX = ".png"

# 你可以改这个数字，表示抽多少张图做预览
NUM_SAMPLES = 12

# 形态学核大小，先用 3x3，第一版最稳
KERNEL_SIZE = 3


def make_edge_from_mask(mask_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    输入:
        mask_array: [H, W], 取值 0/1 或 0/255
    输出:
        edge_array: [H, W], 取值 0/1
    """
    mask_bin = (mask_array > 0).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 形态学梯度 = 膨胀 - 腐蚀
    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    eroded = cv2.erode(mask_bin, kernel, iterations=1)
    edge = dilated - eroded

    edge = (edge > 0).astype(np.uint8)
    return edge


def overlay_edge_on_image(image: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """
    image: [H, W, 3], range 0~1
    edge : [H, W], 0/1
    """
    out = image.copy()
    out[..., 0] = np.where(edge > 0, 1.0, out[..., 0])  # 红色
    out[..., 1] = np.where(edge > 0, 0.0, out[..., 1])
    out[..., 2] = np.where(edge > 0, 0.0, out[..., 2])
    return out


def save_preview(stem: str):
    img_path = IMAGES_DIR / f"{stem}{IMAGE_SUFFIX}"
    mask_path = MASKS_DIR / f"{stem}{MASK_SUFFIX}"

    if not img_path.exists():
        print(f"[Missing image] {img_path}")
        return False
    if not mask_path.exists():
        print(f"[Missing mask] {mask_path}")
        return False

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    image_np = np.array(image).astype(np.float32) / 255.0
    mask_np = np.array(mask).astype(np.uint8)

    edge_np = make_edge_from_mask(mask_np, kernel_size=KERNEL_SIZE)
    overlay_np = overlay_edge_on_image(image_np, edge_np)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))

    axes[0].imshow(image_np)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(mask_np, cmap="gray")
    axes[1].set_title("Mask")
    axes[1].axis("off")

    axes[2].imshow(edge_np, cmap="gray")
    axes[2].set_title("Edge GT")
    axes[2].axis("off")

    axes[3].imshow(overlay_np)
    axes[3].set_title("Image + Edge")
    axes[3].axis("off")

    plt.tight_layout()
    save_path = OUT_DIR / f"{stem}_edge_preview.png"
    plt.savefig(save_path, dpi=150)
    plt.close()

    return True


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(IMAGES_DIR.glob(f"*{IMAGE_SUFFIX}"))
    stems = [p.stem for p in image_files]

    if len(stems) == 0:
        print("No image files found.")
        return

    random.seed(42)
    chosen = random.sample(stems, min(NUM_SAMPLES, len(stems)))

    print("=" * 60)
    print(f"Total images found : {len(stems)}")
    print(f"Samples to preview : {len(chosen)}")
    print(f"Kernel size        : {KERNEL_SIZE}")
    print(f"Output directory   : {OUT_DIR}")
    print("=" * 60)

    ok = 0
    for stem in chosen:
        success = save_preview(stem)
        if success:
            ok += 1
            print(f"[OK] {stem}")

    print("=" * 60)
    print(f"Done. Saved {ok} preview images.")
    print("=" * 60)


if __name__ == "__main__":
    main()
