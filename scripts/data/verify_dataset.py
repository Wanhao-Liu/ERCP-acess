"""
verify_dataset.py — Verify CVAT-exported annotations and assemble YOLO dataset.

After annotation is complete and exported from CVAT (YOLO format), run this
script to:
1. Parse annotation files and verify them.
2. Split by episode (not by frame) into train/val.
3. Copy images and labels to data/yolo_dataset/{images,labels}/{train,val}.
4. Generate dataset_report.txt.

CVAT export format expected:
    annotation_dir/
        images/
            ep001_f00512.jpg
            static_normal_001.jpg
            ...
        labels/
            ep001_f00512.txt   (YOLO format: class cx cy w h, normalized)
            static_normal_001.txt
            ...

Usage:
    python scripts/data/verify_dataset.py \\
        --annotation_dir /path/to/cvat_export \\
        [--out_dir data/yolo_dataset] \\
        [--val_ratio 0.2] \\
        [--seed 42]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from typing import Optional

import numpy as np

PROJECT_ROOT = "/data/ERCP/ercp_access"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, "/root/.local/lib/python3.10/site-packages")


# ── Helpers ───────────────────────────────────────────────────────────────────

def parse_episode_id(filename: str) -> Optional[int]:
    """
    Extract episode_id from filename stem.

    'ep003_f00512' -> 3
    'static_*'     -> None (static image)
    """
    m = re.match(r"ep(\d+)_f\d+", filename)
    if m:
        return int(m.group(1))
    return None


def is_static(filename: str) -> bool:
    return filename.startswith("static_")


def verify_label_file(label_path: str) -> tuple[bool, str]:
    """
    Basic YOLO label validation.
    Returns (ok, message).
    """
    if not os.path.exists(label_path):
        return False, "label file missing"
    with open(label_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    if len(lines) == 0:
        # Empty label = background (allowed)
        return True, "ok (background)"
    for ln in lines:
        parts = ln.split()
        if len(parts) != 5:
            return False, f"bad format: '{ln}'"
        try:
            cls_id = int(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            return False, f"non-numeric value: '{ln}'"
        if cls_id != 0:
            return False, f"unexpected class id {cls_id} (expected 0)"
        if not all(0.0 <= v <= 1.0 for v in vals):
            return False, f"values out of [0,1]: {vals}"
    return True, f"ok ({len(lines)} boxes)"


def copy_pair(img_src: str, lbl_src: str, img_dst: str, lbl_dst: str) -> None:
    os.makedirs(os.path.dirname(img_dst), exist_ok=True)
    os.makedirs(os.path.dirname(lbl_dst), exist_ok=True)
    shutil.copy2(img_src, img_dst)
    # If label file exists copy it; otherwise create empty label
    if os.path.exists(lbl_src):
        shutil.copy2(lbl_src, lbl_dst)
    else:
        open(lbl_dst, "w").close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(annotation_dir: str, out_dir: str, val_ratio: float, seed: int) -> None:
    img_dir = os.path.join(annotation_dir, "images")
    lbl_dir = os.path.join(annotation_dir, "labels")

    if not os.path.isdir(img_dir):
        print(f"ERROR: image directory not found: {img_dir}")
        sys.exit(1)

    # Collect all image files
    img_files = sorted(
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    if not img_files:
        print(f"ERROR: no images found in {img_dir}")
        sys.exit(1)

    print(f"Found {len(img_files)} images in annotation directory.")

    # ── Categorize images ────────────────────────────────────────────────────
    # Group by episode_id; static images grouped separately
    episode_to_files: dict[int, list[str]] = defaultdict(list)
    static_files: list[str] = []
    errors: list[str] = []

    for fname in img_files:
        stem = os.path.splitext(fname)[0]
        eid = parse_episode_id(stem)
        lbl_path = os.path.join(lbl_dir, stem + ".txt")
        ok, msg = verify_label_file(lbl_path)
        if not ok:
            errors.append(f"  {fname}: {msg}")
        if eid is not None:
            episode_to_files[eid].append(fname)
        elif is_static(stem):
            static_files.append(fname)
        else:
            print(f"  WARNING: cannot categorize file: {fname}")

    if errors:
        print(f"\nAnnotation errors ({len(errors)}):")
        for e in errors:
            print(e)
        print("\nFix errors before proceeding. Aborting.")
        sys.exit(1)

    # ── Split episodes ────────────────────────────────────────────────────────
    episode_ids = sorted(episode_to_files.keys())
    rng = np.random.default_rng(seed)
    shuffled = list(episode_ids)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    val_episodes = set(shuffled[:n_val])
    train_episodes = set(shuffled[n_val:])

    # Static images: random 80/20
    rng2 = np.random.default_rng(seed + 1)
    static_shuffled = list(static_files)
    rng2.shuffle(static_shuffled)
    n_val_static = max(0, int(len(static_shuffled) * val_ratio))
    val_static   = set(static_shuffled[:n_val_static])
    train_static = set(static_shuffled[n_val_static:])

    print(f"\nEpisode split: {len(train_episodes)} train, {len(val_episodes)} val episodes")
    print(f"Static split:  {len(train_static)} train, {len(val_static)} val")

    # ── Copy files ────────────────────────────────────────────────────────────
    stats = {"train": 0, "val": 0}

    def copy_file(fname: str, split: str) -> None:
        stem = os.path.splitext(fname)[0]
        img_src = os.path.join(img_dir, fname)
        lbl_src = os.path.join(lbl_dir, stem + ".txt")
        img_dst = os.path.join(out_dir, "images", split, fname)
        lbl_dst = os.path.join(out_dir, "labels", split, stem + ".txt")
        copy_pair(img_src, lbl_src, img_dst, lbl_dst)
        stats[split] += 1

    # Episode frames
    for eid, files in episode_to_files.items():
        split = "val" if eid in val_episodes else "train"
        for fname in files:
            copy_file(fname, split)

    # Static images
    for fname in static_files:
        split = "val" if fname in val_static else "train"
        copy_file(fname, split)

    total = stats["train"] + stats["val"]
    print(f"\nCopied {total} total: {stats['train']} train, {stats['val']} val")

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = os.path.join(out_dir, "dataset_report.txt")
    with open(report_path, "w") as f:
        f.write("YOLO Dataset Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Annotation source: {annotation_dir}\n")
        f.write(f"Output directory:  {out_dir}\n")
        f.write(f"Val ratio:         {val_ratio}\n")
        f.write(f"Seed:              {seed}\n\n")
        f.write(f"Total images:      {total}\n")
        f.write(f"  Train:           {stats['train']}\n")
        f.write(f"  Val:             {stats['val']}\n\n")
        f.write(f"Episode split:\n")
        f.write(f"  Train episodes:  {sorted(train_episodes)}\n")
        f.write(f"  Val episodes:    {sorted(val_episodes)}\n\n")
        f.write(f"Static images:\n")
        f.write(f"  Train: {len(train_static)}, Val: {len(val_static)}\n")
        if errors:
            f.write(f"\nErrors ({len(errors)}):\n")
            for e in errors:
                f.write(e + "\n")

    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify CVAT annotations and assemble YOLO dataset."
    )
    parser.add_argument(
        "--annotation_dir",
        required=True,
        help="CVAT export root directory (must contain images/ and labels/).",
    )
    parser.add_argument(
        "--out_dir",
        default=os.path.join(PROJECT_ROOT, "data", "yolo_dataset"),
        help="Output YOLO dataset directory.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.2,
        help="Fraction of episodes for validation (default 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for split (default 42).",
    )
    args = parser.parse_args()
    main(args.annotation_dir, args.out_dir, args.val_ratio, args.seed)
