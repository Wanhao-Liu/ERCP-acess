#!/usr/bin/env python3
"""
Export 71 phantom episodes to LeRobot dataset v2.1 format.

Output structure:
    data/lerobot/ercp_phantom_v1/
      meta/info.json, episodes.jsonl, tasks.jsonl, stats.json
      data/chunk-000/episode_{N:06d}.parquet  (71 files)
      videos/chunk-000/observation.images.endoscope/episode_{N:06d}.mp4

observation.state: 3D cumulative motor angles [cum_M1_deg, cum_M3_deg, cum_M4_deg]
                   relative to episode start (reset to 0 at frame 0)
action:            [delta_insert, delta_pitch, delta_yaw] from load_actions()
NOTE: Perception fields (e_x, e_y, scale, conf...) are NOT included in this version.
      They will be back-filled after YOLO training.

Usage:
    cd /data/ERCP/ercp_access
    conda run -n ercp python scripts/data/export_lerobot_ercp.py
    conda run -n ercp python scripts/data/export_lerobot_ercp.py --dry_run  # test 3 episodes
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy as np
import polars as pl
from pathlib import Path
from typing import List, Tuple

# Import phantom dataset — module-level constants and classes
from src.data.phantom_dataset import (
    PhantomDataset,
    PhantomEpisode,
    load_actions,
    COL_M1_DEG,
    COL_M3_DEG,
    COL_M4_DEG,
)

FPS = 30
VIDEO_KEY = "observation.images.endoscope"
TASK_STR  = "Perform biliary cannulation via ERCP daughter scope"
CHUNK_SIZE = 100  # episodes per chunk (all 71 fit in chunk-000)


def load_cumulative_motor(
    csv_path: str,
    has_pitch: bool,
    has_yaw: bool,
) -> np.ndarray:
    """
    Load motor CSV and return cumulative motor angles relative to episode start.
    Returns: float32 (T, 3) array: [cum_M1_deg, cum_M3_deg, cum_M4_deg]
    Columns inactive for this episode are zero-filled.

    COL_M1_DEG=5, COL_M3_DEG=13, COL_M4_DEG=17 (0-based, from phantom_dataset.py)
    """
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    T = data.shape[0]

    m1 = data[:, COL_M1_DEG]
    m3 = data[:, COL_M3_DEG] if has_pitch else np.zeros(T, dtype=np.float64)
    m4 = data[:, COL_M4_DEG] if has_yaw   else np.zeros(T, dtype=np.float64)

    # Relative to episode start (frame 0)
    cum = np.stack([m1 - m1[0], m3 - m3[0], m4 - m4[0]], axis=1)
    return cum.astype(np.float32)


def write_video_mp4(
    video_src_path: str,
    n_frames: int,
    out_path: str,
    size: int = 512,
) -> None:
    """
    Read AVI, resize frames to size×size, write MP4 (mp4v codec).
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, float(FPS), (size, size))
    cap = cv2.VideoCapture(video_src_path)
    written = 0
    try:
        while written < n_frames:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
            writer.write(resized)
            written += 1
    finally:
        cap.release()
        writer.release()


def build_episode_parquet(
    episode_index: int,
    obs_state: np.ndarray,  # (T, 3) float32
    actions: np.ndarray,    # (T, 3) float32
    global_frame_offset: int,
    out_path: str,
) -> None:
    """Write a single episode as a parquet file."""
    T = obs_state.shape[0]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pl.DataFrame({
        "observation.state": pl.Series(
            obs_state.tolist(), dtype=pl.List(pl.Float32)
        ),
        "action": pl.Series(
            actions.tolist(), dtype=pl.List(pl.Float32)
        ),
        "timestamp":     [float(t) / FPS  for t in range(T)],
        "frame_index":   list(range(T)),
        "episode_index": [episode_index] * T,
        "index":         list(range(global_frame_offset, global_frame_offset + T)),
        "task_index":    [0] * T,
        "next.done":     [False] * (T - 1) + [True],
    })
    df.write_parquet(out_path)


def compute_stats(
    all_obs: np.ndarray,     # (N_total, 3)
    all_actions: np.ndarray, # (N_total, 3)
) -> dict:
    """Compute per-feature statistics for meta/stats.json."""
    def feat_stats(arr: np.ndarray) -> dict:
        return {
            "mean": arr.mean(axis=0).tolist(),
            "std":  arr.std(axis=0).tolist(),
            "min":  arr.min(axis=0).tolist(),
            "max":  arr.max(axis=0).tolist(),
            "q01":  np.quantile(arr, 0.01, axis=0).tolist(),
            "q99":  np.quantile(arr, 0.99, axis=0).tolist(),
        }
    return {
        "observation.state": feat_stats(all_obs),
        "action":            feat_stats(all_actions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir",    default="data/lerobot/ercp_phantom_v1")
    parser.add_argument("--val_ratio",  type=float, default=0.2)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--video_size", type=int,   default=512)
    parser.add_argument("--dry_run",    action="store_true",
                        help="Process only first 3 episodes for testing")
    args = parser.parse_args()

    base_dir = Path("/data/ERCP/ercp_access")
    out_root = base_dir / args.out_dir

    # ── 1. Load dataset and compute split ─────────────────────────────────
    # PhantomDataset loads all 71 episodes by default
    dataset = PhantomDataset()
    all_episode_ids = dataset.episode_ids  # List[int], 1-based

    # train_val_split is a static method: returns (train_ids, val_ids)
    train_ids, val_ids = PhantomDataset.train_val_split(
        episode_ids=all_episode_ids,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    # train first, val second (so episode_index 0..N_train-1=train, N_train..=val)
    ordered_ids = sorted(train_ids) + sorted(val_ids)
    n_train, n_val = len(train_ids), len(val_ids)

    if args.dry_run:
        print("[dry_run] Processing only first 3 episodes")
        ordered_ids = ordered_ids[:3]

    print(f"[export] {n_train} train + {n_val} val = {len(all_episode_ids)} episodes total")
    print(f"[export] Processing {len(ordered_ids)} episodes")
    print(f"[export] Output: {out_root}")

    # ── 2. Process episodes ────────────────────────────────────────────────
    global_frame_offset = 0
    episodes_meta = []
    all_obs_list     = []
    all_actions_list = []

    for episode_index, orig_id in enumerate(ordered_ids):
        chunk_idx = episode_index // CHUNK_SIZE
        chunk_str = f"chunk-{chunk_idx:03d}"

        parquet_path = str(
            out_root / "data" / chunk_str / f"episode_{episode_index:06d}.parquet"
        )
        video_out_path = str(
            out_root / "videos" / chunk_str / VIDEO_KEY
            / f"episode_{episode_index:06d}.mp4"
        )

        print(f"  ep{orig_id:03d} -> episode_{episode_index:06d} ...", end=" ", flush=True)

        # Load episode via PhantomDataset.get_episode(episode_id: int)
        ep = dataset.get_episode(orig_id)  # returns PhantomEpisode

        # ep.meta holds EpisodeMeta with csv_path, video_path, has_pitch, has_yaw
        meta = ep.meta

        # Load cumulative motor state (observation.state)
        cum_motor = load_cumulative_motor(
            csv_path  = meta.csv_path,
            has_pitch = meta.has_pitch,
            has_yaw   = meta.has_yaw,
        )

        # Load normalized actions: ep.actions is float32 (T, 3)
        actions = ep.actions  # already loaded in PhantomEpisode.__init__

        T = cum_motor.shape[0]
        # Trim to minimum length in case of mismatch
        min_T = min(T, actions.shape[0])
        cum_motor = cum_motor[:min_T]
        actions   = actions[:min_T]

        # Write parquet
        build_episode_parquet(
            episode_index       = episode_index,
            obs_state           = cum_motor,
            actions             = actions,
            global_frame_offset = global_frame_offset,
            out_path            = parquet_path,
        )

        # Write video
        write_video_mp4(
            video_src_path = meta.video_path,
            n_frames       = min_T,
            out_path       = video_out_path,
            size           = args.video_size,
        )

        all_obs_list.append(cum_motor)
        all_actions_list.append(actions)
        global_frame_offset += min_T

        episodes_meta.append({
            "episode_index": episode_index,
            "tasks":         [TASK_STR],
            "length":        min_T,
        })
        print(f"frames={min_T} OK")

    total_frames = global_frame_offset

    # ── 3. Compute stats ───────────────────────────────────────────────────
    all_obs     = np.concatenate(all_obs_list,     axis=0)
    all_actions = np.concatenate(all_actions_list, axis=0)
    stats = compute_stats(all_obs, all_actions)

    # ── 4. Write meta files ────────────────────────────────────────────────
    if args.dry_run:
        print("[dry_run] Skipping meta file write")
        print(f"[dry_run] Would write {total_frames} frames total")
        return

    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # info.json
    info = {
        "codebase_version": "v2.1",
        "robot_type":       "ercp_daughter_scope",
        "total_episodes":   len(ordered_ids),
        "total_frames":     total_frames,
        "fps":              FPS,
        "video_backend":    "opencv",
        "features": {
            "observation.state": {
                "dtype": "float32", "shape": [3],
                "names": ["cum_insert_deg", "cum_pitch_deg", "cum_yaw_deg"],
            },
            f"{VIDEO_KEY}": {
                "dtype": "video",
                "shape": [args.video_size, args.video_size, 3],
                "names": ["height", "width", "channel"],
                "video_info": {
                    "fps": FPS, "codec": "mp4v", "pix_fmt": "yuv420p",
                },
            },
            "action": {
                "dtype": "float32", "shape": [3],
                "names": ["delta_insert", "delta_pitch", "delta_yaw"],
            },
            "timestamp":     {"dtype": "float64", "shape": [1], "names": None},
            "frame_index":   {"dtype": "int64",   "shape": [1], "names": None},
            "episode_index": {"dtype": "int64",   "shape": [1], "names": None},
            "index":         {"dtype": "int64",   "shape": [1], "names": None},
            "task_index":    {"dtype": "int64",   "shape": [1], "names": None},
            "next.done":     {"dtype": "bool",    "shape": [1], "names": None},
        },
        "splits": {
            "train": f"0:{n_train}",
            "val":   f"{n_train}:{n_train + n_val}",
        },
        "data_path":  "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": f"videos/chunk-{{episode_chunk:03d}}/{VIDEO_KEY}/episode_{{episode_index:06d}}.mp4",
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    # tasks.jsonl
    with open(meta_dir / "tasks.jsonl", "w") as f:
        f.write(json.dumps({"task_index": 0, "task": TASK_STR}) + "\n")

    # episodes.jsonl
    with open(meta_dir / "episodes.jsonl", "w") as f:
        for ep_meta in episodes_meta:
            f.write(json.dumps(ep_meta) + "\n")

    # stats.json
    with open(meta_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n[export] Done.")
    print(f"  Episodes : {len(ordered_ids)} ({n_train} train + {n_val} val)")
    print(f"  Frames   : {total_frames}")
    print(f"  Output   : {out_root}")


if __name__ == "__main__":
    main()
