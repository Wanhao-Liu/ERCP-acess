"""
train_yolo.py — Two-phase YOLOv8-nano fine-tuning for papilla detection.

Phase 1 (epochs 0-49):  backbone frozen (first `freeze` layers), head-only training.
Phase 2 (epoch 50+):    all layers unfrozen, lr reduced to lr0/10.

Usage:
    python scripts/perception/train_yolo.py [--config configs/yolo/train.yaml]
    python scripts/perception/train_yolo.py --unfreeze_epoch 50
"""
from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = "/data/ERCP/ercp_access"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, "/root/.local/lib/python3.10/site-packages")

import yaml
from ultralytics import YOLO
from ultralytics.utils import callbacks as yolo_callbacks


# ── Callback factory ─────────────────────────────────────────────────────────

def build_unfreeze_callback(unfreeze_epoch: int = 50):
    """
    Build a YOLO on_epoch_start callback that:
    - At epoch == unfreeze_epoch: unfreezes all model layers and halves lr.

    Returns the callback function.
    """
    _unfrozen = [False]  # mutable state via closure

    def on_epoch_start(trainer):
        epoch = trainer.epoch  # 0-based current epoch

        if epoch == unfreeze_epoch and not _unfrozen[0]:
            print(f"\n[train_yolo] Epoch {epoch}: Unfreezing all layers and reducing lr.")

            # Unfreeze all parameters
            for param in trainer.model.parameters():
                param.requires_grad = True

            # Reduce learning rate
            for pg in trainer.optimizer.param_groups:
                pg["lr"] = pg["initial_lr"] * 0.1

            _unfrozen[0] = True
            print(f"[train_yolo] All layers unfrozen. LR set to {trainer.optimizer.param_groups[0]['lr']:.6f}")

    return on_epoch_start


# ── Main ──────────────────────────────────────────────────────────────────────

def main(config_path: str, unfreeze_epoch: int) -> None:
    # Load training config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    print(f"[train_yolo] Loaded config: {config_path}")
    print(f"[train_yolo] Model: {cfg.get('model', 'yolov8n.pt')}")
    print(f"[train_yolo] Epochs: {cfg.get('epochs', 100)}, freeze: {cfg.get('freeze', 10)}")
    print(f"[train_yolo] Unfreeze epoch: {unfreeze_epoch}")

    # Initialize model
    model = YOLO(cfg.get("model", "yolov8n.pt"))

    # Register unfreeze callback
    unfreeze_cb = build_unfreeze_callback(unfreeze_epoch=unfreeze_epoch)
    model.add_callback("on_train_epoch_start", unfreeze_cb)

    # Build training kwargs from config (pass all recognized YOLO train params)
    train_kwargs = {
        k: v for k, v in cfg.items()
        if k not in ("model",)  # exclude keys not passed to train()
    }

    # Run training
    results = model.train(**train_kwargs)

    print(f"\n[train_yolo] Training complete.")
    print(f"[train_yolo] Best weights: {results.save_dir}/weights/best.pt")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Two-phase YOLOv8-nano training for papilla detection.")
    parser.add_argument(
        "--config",
        default=os.path.join(PROJECT_ROOT, "configs", "yolo", "train.yaml"),
        help="Path to train.yaml config file.",
    )
    parser.add_argument(
        "--unfreeze_epoch",
        type=int,
        default=50,
        help="Epoch at which to unfreeze all backbone layers (default 50).",
    )
    args = parser.parse_args()
    main(args.config, args.unfreeze_epoch)
