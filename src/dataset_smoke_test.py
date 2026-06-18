from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
TRAIN_SPLIT = PROJECT_ROOT / "data" / "splits" / "train.txt"
OUT_DIR = PROJECT_ROOT / "runs" / "debug"

# 统一输入尺寸
TARGET_SIZE = (512, 512)  # (width, height)

class CariesDataset(Dataset):
    def __init__(self, split_file):
        self.items = []
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

        # 读图
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 统一 resize
        image = image.resize(TARGET_SIZE, resample=Image.BILINEAR)
        mask = mask.resize(TARGET_SIZE, resample=Image.NEAREST)

        image = np.array(image).astype(np.float32) / 255.0
        mask = np.array(mask).astype(np.float32)

        # 二值化
        mask = (mask > 0).astype(np.float32)

        # HWC -> CHW
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask, str(img_path), str(mask_path)

def make_preview(images, masks, save_path):
    b = min(images.shape[0], 4)

    fig, axes = plt.subplots(b, 3, figsize=(9, 3 * b))
    if b == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(b):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mask = masks[i, 0].cpu().numpy()

        overlay = img.copy()
        overlay[..., 0] = np.where(mask > 0, 1.0, overlay[..., 0])
        overlay[..., 1] = np.where(mask > 0, 0.0, overlay[..., 1])
        overlay[..., 2] = np.where(mask > 0, 0.0, overlay[..., 2])
        overlay = 0.6 * img + 0.4 * overlay

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(mask, cmap="gray")
        axes[i, 1].set_title("Mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Overlay")
        axes[i, 2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = CariesDataset(TRAIN_SPLIT)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    print(f"dataset size = {len(dataset)}")
    print(f"target size  = {TARGET_SIZE}")

    batch = next(iter(loader))
    images, masks, img_paths, mask_paths = batch

    print("images.shape =", images.shape)
    print("masks.shape  =", masks.shape)
    print("images.min/max =", images.min().item(), images.max().item())
    print("masks unique  =", torch.unique(masks))

    print("\nSample paths:")
    for i in range(min(4, len(img_paths))):
        print("img :", img_paths[i])
        print("mask:", mask_paths[i])

    save_path = OUT_DIR / "first_batch_preview.png"
    make_preview(images, masks, save_path)
    print(f"\nPreview saved to: {save_path}")

if __name__ == "__main__":
    main()
