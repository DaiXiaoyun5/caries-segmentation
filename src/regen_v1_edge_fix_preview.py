from pathlib import Path
import sys
import json

import matplotlib
matplotlib.use("Agg")

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path("/share/home/u2515283028/caries_project")
SRC_DIR = PROJECT_ROOT / "src"
sys.path.append(str(SRC_DIR))

from train_method_v1_edge_fix import CariesEdgeDataset, UnetPPWithEdge, save_prediction_panel

RUN_DIR = PROJECT_ROOT / "runs" / "unetpp_v1_edge_fix"
CONFIG_PATH = RUN_DIR / "config.json"
CKPT_PATH = RUN_DIR / "checkpoints" / "best.pth"
PRED_DIR = RUN_DIR / "preds"
OUT_PATH = PRED_DIR / "test_preview.png"

TEST_SPLIT = PROJECT_ROOT / "data" / "splits" / "test.txt"


def main():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    image_size = config.get("image_size", 512)
    batch_size = config.get("batch_size", 6)
    encoder_name = config.get("encoder_name", "resnet34")
    encoder_weights = config.get("encoder_weights", "imagenet")
    edge_vis_threshold = config.get("edge_vis_threshold", 0.2)

    dataset = CariesEdgeDataset(
        TEST_SPLIT,
        target_size=(image_size, image_size),
        edge_kernel_size=3,
    )
    loader = DataLoader(
        dataset,
        batch_size=min(batch_size, 4),
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    model = UnetPPWithEdge(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
    ).to(device)

    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    PRED_DIR.mkdir(parents=True, exist_ok=True)

    save_prediction_panel(
        model,
        loader,
        device,
        OUT_PATH,
        edge_vis_threshold=edge_vis_threshold,
    )

    print(f"Preview saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
