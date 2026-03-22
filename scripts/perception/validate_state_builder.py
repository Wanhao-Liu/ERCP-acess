"""
validate_state_builder.py — End-to-end StateBuilder validation.

Runs the full StateBuilder chain (with PapillaDetector) on 3 val episodes,
comparing predicted e_x/e_y to annotated GT bounding box centroids.
Generates a scatter plot of predicted vs GT e_x and e_y.

Requires:
    - Trained detector weights (--weights)
    - Annotated val split in data/yolo_dataset/labels/val/
    - Video data in data/cannulation/

Usage:
    python scripts/perception/validate_state_builder.py \\
        --weights outputs/perception/yolo_papilla/yolov8n_v1/weights/best.pt \\
        [--val_episodes 5 12 23] \\
        [--max_frames 100] \\
        [--out_dir outputs/perception/validation]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

import numpy as np

PROJECT_ROOT = "/data/ERCP/ercp_access"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, "/root/.local/lib/python3.10/site-packages")


def load_gt_for_frame(label_dir: str, episode_id: int, frame_idx: int) -> Optional[tuple[float, float]]:
    """
    Load GT centroid (e_x, e_y) from YOLO label file for a specific frame.

    Returns None if label file missing or no boxes annotated.
    """
    fname = f"ep{episode_id:03d}_f{frame_idx:05d}.txt"
    label_path = os.path.join(label_dir, fname)

    if not os.path.exists(label_path):
        return None

    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]

    if not lines:
        return None

    # Use first box (should only be one papilla per frame)
    parts = lines[0].split()
    if len(parts) != 5:
        return None

    cx_n = float(parts[1])  # normalized [0,1]
    cy_n = float(parts[2])

    # Convert to e_x, e_y using project axis mapping
    e_x = (cx_n - 0.5) / 0.5
    e_y = -((cy_n - 0.5) / 0.5)

    return e_x, e_y


def validate_episode(
    episode_id: int,
    label_dir: str,
    state_builder,
    max_frames: int,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """
    Process one episode: run StateBuilder, collect (pred_ex, gt_ex, pred_ey, gt_ey).
    """
    from src.data.phantom_dataset import PhantomEpisode

    pred_ex_list, gt_ex_list = [], []
    pred_ey_list, gt_ey_list = [], []

    state_builder.reset()

    with PhantomEpisode(episode_id) as ep:
        for frame_idx, (rgb, _action) in enumerate(ep.iter_frames()):
            if frame_idx >= max_frames:
                break

            gt = load_gt_for_frame(label_dir, episode_id, frame_idx)
            if gt is None:
                continue  # no annotation for this frame

            # Run StateBuilder in non-oracle mode
            state = state_builder.update(rgb)

            pred_ex_list.append(state.e_x)
            gt_ex_list.append(gt[0])
            pred_ey_list.append(state.e_y)
            gt_ey_list.append(gt[1])

    return pred_ex_list, gt_ex_list, pred_ey_list, gt_ey_list


def make_scatter_plot(
    pred_ex: list[float],
    gt_ex:   list[float],
    pred_ey: list[float],
    gt_ey:   list[float],
    out_path: str,
) -> None:
    """Generate e_x and e_y scatter plots (predicted vs GT)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, pred, gt, label in [
        (axes[0], pred_ex, gt_ex, "e_x"),
        (axes[1], pred_ey, gt_ey, "e_y"),
    ]:
        ax.scatter(gt, pred, alpha=0.4, s=10, color="steelblue")
        lim = [-1.1, 1.1]
        ax.plot(lim, lim, "r--", linewidth=1, label="ideal")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel(f"GT {label}")
        ax.set_ylabel(f"Predicted {label}")
        ax.set_title(f"{label}: Predicted vs GT (n={len(pred)})")
        ax.legend()

        if len(pred) > 1:
            corr = float(np.corrcoef(gt, pred)[0, 1])
            mae  = float(np.mean(np.abs(np.array(pred) - np.array(gt))))
            ax.text(
                0.05, 0.92,
                f"corr={corr:.3f}  MAE={mae:.3f}",
                transform=ax.transAxes,
                fontsize=9,
            )

    fig.suptitle("StateBuilder Validation: Predicted vs GT e_x/e_y")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Plot saved to: {out_path}")


def main(
    weights: str,
    val_episodes: list[int],
    max_frames: int,
    out_dir: str,
) -> None:
    from src.perception.state_builder import StateBuilder

    label_dir = os.path.join(PROJECT_ROOT, "data", "yolo_dataset", "labels", "val")

    if not os.path.isdir(label_dir):
        print(f"ERROR: label dir not found: {label_dir}")
        print("Run verify_dataset.py first to assemble the YOLO dataset.")
        sys.exit(1)

    print(f"[validate_state_builder] Weights: {weights}")
    print(f"[validate_state_builder] Episodes: {val_episodes}")
    print(f"[validate_state_builder] Max frames per episode: {max_frames}")

    # Build StateBuilder in non-oracle mode with detector
    sb = StateBuilder(
        use_oracle=False,
        detector_weights=weights,
        detector_conf_thresh=0.25,
        detector_device="cuda",
    )

    all_pred_ex, all_gt_ex = [], []
    all_pred_ey, all_gt_ey = [], []

    for ep_id in val_episodes:
        print(f"\nProcessing ep{ep_id:03d}...")
        try:
            pred_ex, gt_ex, pred_ey, gt_ey = validate_episode(
                ep_id, label_dir, sb, max_frames
            )
            n = len(pred_ex)
            if n == 0:
                print(f"  No annotated frames found in val labels for ep{ep_id:03d}.")
                continue
            print(f"  {n} annotated frames processed.")
            all_pred_ex.extend(pred_ex)
            all_gt_ex.extend(gt_ex)
            all_pred_ey.extend(pred_ey)
            all_gt_ey.extend(gt_ey)
        except Exception as e:
            print(f"  ERROR processing ep{ep_id:03d}: {e}")

    if not all_pred_ex:
        print("\nNo data collected. Check annotation and episode configuration.")
        return

    # Summary statistics
    n = len(all_pred_ex)
    pred_ex_arr = np.array(all_pred_ex)
    gt_ex_arr   = np.array(all_gt_ex)
    pred_ey_arr = np.array(all_pred_ey)
    gt_ey_arr   = np.array(all_gt_ey)

    corr_ex = float(np.corrcoef(gt_ex_arr, pred_ex_arr)[0, 1]) if n > 1 else float("nan")
    corr_ey = float(np.corrcoef(gt_ey_arr, pred_ey_arr)[0, 1]) if n > 1 else float("nan")
    mae_ex  = float(np.mean(np.abs(pred_ex_arr - gt_ex_arr)))
    mae_ey  = float(np.mean(np.abs(pred_ey_arr - gt_ey_arr)))

    print(f"\n=== Summary (n={n}) ===")
    print(f"  e_x: corr={corr_ex:.4f}, MAE={mae_ex:.4f}")
    print(f"  e_y: corr={corr_ey:.4f}, MAE={mae_ey:.4f}")

    # Save stats
    os.makedirs(out_dir, exist_ok=True)
    stats = {
        "n_frames": n,
        "episodes": val_episodes,
        "e_x_corr": corr_ex,
        "e_x_mae":  mae_ex,
        "e_y_corr": corr_ey,
        "e_y_mae":  mae_ey,
    }
    stats_path = os.path.join(out_dir, "state_builder_validation.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats saved to: {stats_path}")

    # Generate scatter plot
    plot_path = os.path.join(out_dir, "state_builder_scatter.png")
    make_scatter_plot(all_pred_ex, all_gt_ex, all_pred_ey, all_gt_ey, plot_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate StateBuilder e_x/e_y vs GT annotations."
    )
    parser.add_argument(
        "--weights",
        default=os.path.join(
            PROJECT_ROOT, "outputs", "perception", "yolo_papilla", "yolov8n_v1", "weights", "best.pt"
        ),
        help="Path to trained YOLOv8 weights.",
    )
    parser.add_argument(
        "--val_episodes",
        nargs="+",
        type=int,
        default=[5, 12, 23],
        help="Episode IDs to use for validation (default: 5 12 23).",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=500,
        help="Maximum frames per episode to process (default 500).",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(PROJECT_ROOT, "outputs", "perception", "validation"),
        help="Output directory for results.",
    )
    args = parser.parse_args()
    main(args.weights, args.val_episodes, args.max_frames, args.out_dir)
