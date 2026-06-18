from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader


PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
TRAIN_SPLIT = PROJECT_ROOT / "data" / "splits" / "train.txt"
OUT_DIR = PROJECT_ROOT / "runs" / "debug"

TARGET_SIZE = (512, 512)   # (width, height)
EDGE_KERNEL_SIZE = 3


def make_edge_from_mask(mask_array: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    输入:
        mask_array: [H, W], 值为 0/1 或 0/255
    输出:
        edge_array: [H, W], 值为 0/1
    """
    mask_bin = (mask_array > 0).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated = cv2.dilate(mask_bin, kernel, iterations=1)
    eroded = cv2.erode(mask_bin, kernel, iterations=1)
    edge = dilated - eroded
    edge = (edge > 0).astype(np.float32)

    return edge


class CariesEdgeDataset(Dataset):
    def __init__(self, split_file, target_size=(512, 512), edge_kernel_size=3):
        self.items = []
        self.target_size = target_size
        self.edge_kernel_size = edge_kernel_size

        with open(split_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_path, mask_path = line.split("\t")
                self.items.append((Path(img_path), Path(mask_path)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mask_path = self.items[idx]

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 统一 resize
        image = image.resize(self.target_size, resample=Image.BILINEAR)
        mask = mask.resize(self.target_size, resample=Image.NEAREST)

        image_np = np.array(image).astype(np.float32) / 255.0
        mask_np = np.array(mask).astype(np.float32)
        mask_np = (mask_np > 0).astype(np.float32)

        edge_np = make_edge_from_mask(mask_np, kernel_size=self.edge_kernel_size)

        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)       # [3, H, W]
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)             # [1, H, W]
        edge_tensor = torch.from_numpy(edge_np).unsqueeze(0)             # [1, H, W]

        return image_tensor, mask_tensor, edge_tensor, str(img_path), str(mask_path)


def make_overlay(image_np: np.ndarray, binary_np: np.ndarray, color="red") -> np.ndarray:
    """
    image_np: [H, W, 3], 0~1
    binary_np: [H, W], 0/1
    """
    out = image_np.copy()

    if color == "red":
        out[..., 0] = np.where(binary_np > 0, 1.0, out[..., 0])
        out[..., 1] = np.where(binary_np > 0, 0.0, out[..., 1])
        out[..., 2] = np.where(binary_np > 0, 0.0, out[..., 2])
    elif color == "green":
        out[..., 0] = np.where(binary_np > 0, 0.0, out[..., 0])
        out[..., 1] = np.where(binary_np > 0, 1.0, out[..., 1])
        out[..., 2] = np.where(binary_np > 0, 0.0, out[..., 2])

    out = 0.6 * image_np + 0.4 * out
    return out


def save_batch_preview(images, masks, edges, save_path):
    """
    images: [B, 3, H, W]
    masks : [B, 1, H, W]
    edges : [B, 1, H, W]
    """
    b = min(images.shape[0], 4)

    fig, axes = plt.subplots(b, 5, figsize=(15, 3 * b))
    if b == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(b):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mask = masks[i, 0].cpu().numpy()
        edge = edges[i, 0].cpu().numpy()

        overlay_mask = make_overlay(img, mask, color="red")
        overlay_edge = make_overlay(img, edge, color="green")

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(edge, cmap="gray")
        axes[i, 2].set_title("Edge GT")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(overlay_mask)
        axes[i, 3].set_title("Image + Mask")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(overlay_edge)
        axes[i, 4].set_title("Image + Edge")
        axes[i, 4].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = CariesEdgeDataset(
        TRAIN_SPLIT,
        target_size=TARGET_SIZE,
        edge_kernel_size=EDGE_KERNEL_SIZE,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    print("=" * 60)
    print(f"dataset size      = {len(dataset)}")
    print(f"target size       = {TARGET_SIZE}")
    print(f"edge kernel size  = {EDGE_KERNEL_SIZE}")
    print("=" * 60)

    batch = next(iter(loader))
    images, masks, edges, img_paths, mask_paths = batch

    print("images.shape =", images.shape)
    print("masks.shape  =", masks.shape)
    print("edges.shape  =", edges.shape)

    print("images.min/max =", images.min().item(), images.max().item())
    print("masks unique   =", torch.unique(masks))
    print("edges unique   =", torch.unique(edges))

    print("\nSample paths:")
    for i in range(min(4, len(img_paths))):
        print("img :", img_paths[i])
        print("mask:", mask_paths[i])

    save_path = OUT_DIR / "edge_batch_preview.png"
    save_batch_preview(images, masks, edges, save_path)
    print(f"\nPreview saved to: {save_path}")


if __name__ == "__main__":
    main()
