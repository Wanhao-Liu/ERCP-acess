"""
eval_yolo.py — Offline evaluation of trained papilla detector.

Computes:
  - mAP@0.5, mAP@0.5:0.95 (YOLO built-in)
  - Centroid pixel error: mean/median/p90/p95 distance between predicted and GT bbox centers

Output:
    outputs/perception/eval_report.json

Usage:
    python scripts/perception/eval_yolo.py \\
        --weights outputs/perception/yolo_papilla/yolov8n_v1/weights/best.pt \\
        [--data configs/yolo/papilla_dataset.yaml] \\
        [--split val] \\
        [--conf 0.25] \\
        [--out outputs/perception/eval_report.json]
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


def compute_centroid_error(
    pred_boxes: np.ndarray,
    gt_boxes: np.ndarray,
    img_w: int,
    img_h: int,
) -> float:
    """
    Match each GT box to the nearest predicted box and compute centroid pixel error.

    Parameters
    ----------
    pred_boxes : float32, shape (N, 4) — predicted boxes [cx, cy, w, h] normalized [0,1]
    gt_boxes   : float32, shape (M, 4) — GT boxes [cx, cy, w, h] normalized [0,1]
    img_w, img_h : image dimensions in pixels

    Returns
    -------
    mean_error : float — mean centroid pixel distance (GT-matched)
                 Returns np.nan if either pred_boxes or gt_boxes is empty.
    """
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return float("nan")

    # Convert normalized coords to pixels
    pred_cx = pred_boxes[:, 0] * img_w
    pred_cy = pred_boxes[:, 1] * img_h
    gt_cx   = gt_boxes[:, 0] * img_w
    gt_cy   = gt_boxes[:, 1] * img_h

    errors = []
    for i in range(len(gt_boxes)):
        # Match GT box i to nearest prediction (minimum centroid distance)
        dists = np.sqrt((pred_cx - gt_cx[i]) ** 2 + (pred_cy - gt_cy[i]) ** 2)
        errors.append(float(np.min(dists)))

    return float(np.mean(errors))


def run_yolo_val(weights: str, data: str, split: str, conf: float, device: str) -> dict:
    """Run YOLO validation and return metrics dict."""
    from ultralytics import YOLO

    model = YOLO(weights)
    results = model.val(
        data=data,
        split=split,
        conf=conf,
        device=device,
        verbose=False,
    )

    metrics = {
        "mAP50":    float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall":    float(results.box.mr),
    }
    return metrics, results


def run_centroid_eval(
    weights: str,
    data_yaml: str,
    split: str,
    conf: float,
    device: str,
) -> dict:
    """
    Run per-image inference on the val/test split and compute centroid error stats.
    """
    import cv2
    import yaml
    from ultralytics import YOLO

    with open(data_yaml) as f:
        dataset_cfg = yaml.safe_load(f)

    dataset_root = dataset_cfg.get("path", "")
    img_dir = os.path.join(dataset_root, dataset_cfg.get(split, f"images/{split}"))
    lbl_dir = img_dir.replace("images", "labels")

    if not os.path.isdir(img_dir):
        print(f"  WARNING: image dir not found for centroid eval: {img_dir}")
        return {}

    model = YOLO(weights)

    all_errors = []
    n_no_pred  = 0
    n_no_gt    = 0
    n_total    = 0

    img_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    for fname in img_files:
        img_path = os.path.join(img_dir, fname)
        lbl_path = os.path.join(lbl_dir, os.path.splitext(fname)[0] + ".txt")

        # Load GT
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_boxes.append([float(x) for x in parts[1:5]])
        gt_boxes = np.array(gt_boxes, dtype=np.float32).reshape(-1, 4) if gt_boxes else np.zeros((0, 4))

        # Load image dimensions
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Run prediction
        results = model.predict(img_path, conf=conf, device=device, verbose=False)
        preds = results[0].boxes

        if preds is None or len(preds) == 0:
            pred_boxes = np.zeros((0, 4))
        else:
            # xywhn = normalized cx,cy,w,h
            pred_boxes = preds.xywhn.cpu().numpy()

        n_total += 1

        if len(gt_boxes) == 0:
            n_no_gt += 1
            continue
        if len(pred_boxes) == 0:
            n_no_pred += 1
            continue

        err = compute_centroid_error(pred_boxes, gt_boxes, img_w, img_h)
        if not np.isnan(err):
            all_errors.append(err)

    centroid_stats: dict = {"n_total": n_total, "n_no_pred": n_no_pred, "n_no_gt": n_no_gt}
    if all_errors:
        arr = np.array(all_errors)
        centroid_stats.update({
            "centroid_error_mean":   float(np.mean(arr)),
            "centroid_error_median": float(np.median(arr)),
            "centroid_error_p90":    float(np.percentile(arr, 90)),
            "centroid_error_p95":    float(np.percentile(arr, 95)),
            "centroid_error_max":    float(np.max(arr)),
            "n_evaluated":           len(all_errors),
        })
    else:
        centroid_stats["centroid_error_mean"] = float("nan")

    return centroid_stats


def main(
    weights: str,
    data: str,
    split: str,
    conf: float,
    device: str,
    out: str,
) -> None:
    print(f"[eval_yolo] Weights:  {weights}")
    print(f"[eval_yolo] Dataset:  {data}")
    print(f"[eval_yolo] Split:    {split}")
    print(f"[eval_yolo] Conf:     {conf}")

    print("\n[eval_yolo] Running YOLO validation (mAP)...")
    map_metrics, _ = run_yolo_val(weights, data, split, conf, device)
    print(f"  mAP@0.50:     {map_metrics['mAP50']:.4f}")
    print(f"  mAP@0.50:0.95:{map_metrics['mAP50_95']:.4f}")
    print(f"  Precision:    {map_metrics['precision']:.4f}")
    print(f"  Recall:       {map_metrics['recall']:.4f}")

    print("\n[eval_yolo] Running centroid error evaluation...")
    centroid_stats = run_centroid_eval(weights, data, split, conf, device)
    if centroid_stats:
        mean_err = centroid_stats.get("centroid_error_mean", float("nan"))
        print(f"  Centroid error mean:   {mean_err:.2f} px")
        if "centroid_error_p95" in centroid_stats:
            print(f"  Centroid error p95:    {centroid_stats['centroid_error_p95']:.2f} px")

    report = {
        "weights": weights,
        "dataset": data,
        "split":   split,
        "conf":    conf,
        **map_metrics,
        **centroid_stats,
    }

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[eval_yolo] Report written to: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline evaluation of papilla detector.")
    parser.add_argument(
        "--weights",
        default=os.path.join(
            PROJECT_ROOT, "outputs", "perception", "yolo_papilla", "yolov8n_v1", "weights", "best.pt"
        ),
        help="Path to trained model weights.",
    )
    parser.add_argument(
        "--data",
        default=os.path.join(PROJECT_ROOT, "configs", "yolo", "papilla_dataset.yaml"),
        help="Dataset YAML config.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate on (default: val).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default 0.25).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(PROJECT_ROOT, "outputs", "perception", "eval_report.json"),
        help="Output JSON report path.",
    )
    args = parser.parse_args()
    main(args.weights, args.data, args.split, args.conf, args.device, args.out)
