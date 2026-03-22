"""
sample_frames_for_annotation.py — Sample frames from phantom episodes for annotation.

Samples 30 frames per episode (10 per temporal third) using sequential video reads
to avoid H.264 random-seek performance issues.

Output:
    data/annotation_staging/video_frames/ep{id:03d}_f{idx:05d}.jpg
    data/annotation_staging/video_frame_manifest.csv

Usage:
    python scripts/data/sample_frames_for_annotation.py [--episodes 1 2 3 ...]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

# Add project root to path
PROJECT_ROOT = "/data/ERCP/ercp_access"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, "/root/.local/lib/python3.10/site-packages")

import cv2
import numpy as np

from src.data.phantom_dataset import PhantomDataset, PhantomEpisode, load_episode_meta, N_EPISODES

# ── Configuration ─────────────────────────────────────────────────────────────
FRAMES_PER_THIRD = 10    # frames sampled per temporal third
MARGIN_FRAMES    = 5     # skip first/last N frames

OUT_DIR   = os.path.join(PROJECT_ROOT, "data", "annotation_staging", "video_frames")
MANIFEST  = os.path.join(PROJECT_ROOT, "data", "annotation_staging", "video_frame_manifest.csv")


def sample_indices_for_episode(n_frames: int) -> list[tuple[int, int]]:
    """
    Compute (frame_idx, temporal_third) pairs to sample.

    Splits [MARGIN_FRAMES, n_frames - MARGIN_FRAMES) into 3 thirds.
    Samples FRAMES_PER_THIRD uniformly spaced indices from each third.

    Returns list of (frame_idx, third) where third ∈ {0, 1, 2}.
    """
    start = MARGIN_FRAMES
    end   = n_frames - MARGIN_FRAMES
    if end <= start:
        return []

    usable = end - start
    third_len = usable // 3

    samples: list[tuple[int, int]] = []
    for t in range(3):
        t_start = start + t * third_len
        t_end   = start + (t + 1) * third_len if t < 2 else end
        t_len   = t_end - t_start
        if t_len <= 0:
            continue
        # Uniformly spaced indices within this third
        indices = np.linspace(t_start, t_end - 1, FRAMES_PER_THIRD, dtype=int)
        for idx in indices:
            samples.append((int(idx), t))

    return samples


def process_episode(episode_id: int, writer: csv.DictWriter) -> int:
    """
    Sample frames from one episode using sequential reads.

    Returns number of frames saved.
    """
    meta = load_episode_meta(episode_id)
    n_frames = meta.n_frames

    target_pairs = sample_indices_for_episode(n_frames)
    if not target_pairs:
        print(f"  [ep{episode_id:03d}] Too few frames ({n_frames}), skipping.")
        return 0

    target_set = {idx: third for idx, third in target_pairs}
    target_indices = sorted(target_set.keys())

    saved = 0
    cap = cv2.VideoCapture(meta.video_path)
    try:
        frame_idx = 0
        ti = 0  # pointer into sorted target_indices
        max_target = target_indices[-1]

        while frame_idx <= max_target and ti < len(target_indices):
            ret, frame_bgr = cap.read()
            if not ret:
                print(f"  [ep{episode_id:03d}] Video ended early at frame {frame_idx}")
                break

            if frame_idx == target_indices[ti]:
                third = target_set[frame_idx]

                # Save as JPEG
                fname = f"ep{episode_id:03d}_f{frame_idx:05d}.jpg"
                out_path = os.path.join(OUT_DIR, fname)
                cv2.imwrite(out_path, frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

                writer.writerow({
                    "episode_id":    episode_id,
                    "frame_idx":     frame_idx,
                    "file_path":     out_path,
                    "temporal_third": third,
                })
                saved += 1
                ti += 1

            frame_idx += 1
    finally:
        cap.release()

    return saved


def main(episode_ids: list[int]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest_fields = ["episode_id", "frame_idx", "file_path", "temporal_third"]
    total_saved = 0

    with open(MANIFEST, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=manifest_fields)
        writer.writeheader()

        for ep_id in episode_ids:
            print(f"Processing ep{ep_id:03d}...")
            try:
                n = process_episode(ep_id, writer)
                print(f"  -> Saved {n} frames.")
                total_saved += n
            except Exception as e:
                print(f"  [ep{ep_id:03d}] ERROR: {e}")

    print(f"\nDone. Total frames saved: {total_saved}")
    print(f"Manifest written to: {MANIFEST}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample frames from phantom episodes for annotation.")
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=list(range(1, N_EPISODES + 1)),
        help="Episode IDs to process (1-based). Default: all 1..71.",
    )
    args = parser.parse_args()
    main(args.episodes)
