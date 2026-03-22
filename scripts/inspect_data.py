"""
Inspect and validate all 71 phantom episodes.

Usage:
    cd /data/ERCP/ercp_access
    /usr/bin/python3.10 scripts/inspect_data.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "/root/.local/lib/python3.10/site-packages")

import cv2
import numpy as np

from src.data.phantom_dataset import (
    PhantomDataset,
    PhantomEpisode,
    load_episode_meta,
    N_EPISODES,
    INSERT_SCALE, PITCH_SCALE, YAW_SCALE,
)


def check_video_frame_count(meta) -> tuple[int, bool]:
    """Return (actual_video_frames, matches_csv).

    Tolerates video = CSV + 1 (common AVI trailing-empty-frame artefact).
    """
    cap = cv2.VideoCapture(meta.video_path)
    count = 0
    while True:
        ret, _ = cap.read()
        if not ret:
            break
        count += 1
    cap.release()
    ok = count == meta.n_frames or count == meta.n_frames + 1
    return count, ok


def sample_frame_check(episode_id: int, frame_idx: int) -> tuple[bool, str]:
    """Load one frame and verify shape/dtype/range + action."""
    try:
        ep = PhantomEpisode(episode_id)
        rgb = ep.get_frame(frame_idx)
        action = ep.get_action(frame_idx)
        ep.close()

        checks = [
            (rgb.shape == (3, 1080, 1920),   f"rgb.shape={rgb.shape} expected (3,1080,1920)"),
            (rgb.dtype == np.float32,          f"rgb.dtype={rgb.dtype} expected float32"),
            (float(rgb.min()) >= 0.0,          f"rgb.min()={rgb.min():.4f} < 0"),
            (float(rgb.max()) <= 1.0,          f"rgb.max()={rgb.max():.4f} > 1"),
            (action.shape == (3,),             f"action.shape={action.shape} expected (3,)"),
            (action.dtype == np.float32,       f"action.dtype={action.dtype} expected float32"),
            (bool(np.all(action >= -1.0)),     f"action min={action.min():.3f} < -1"),
            (bool(np.all(action <= 1.0)),      f"action max={action.max():.3f} > 1"),
        ]
        fails = [msg for ok, msg in checks if not ok]
        if fails:
            return False, " | ".join(fails)
        return True, f"rgb {rgb.shape} {rgb.dtype} [{rgb.min():.2f},{rgb.max():.2f}]  action {action}"
    except Exception as e:
        return False, str(e)


def main():
    print("=" * 70)
    print("PHANTOM DATA INSPECTION REPORT")
    print("=" * 70)
    print()

    # ── 1. Load dataset (CSV only) ─────────────────────────────────────────
    print("Step 1: Loading PhantomDataset (CSV only) ...")
    dataset = PhantomDataset()
    summaries = dataset.episode_summary()
    print(f"  Loaded {len(dataset.episode_ids)} episodes, {len(dataset)} total frames")
    print()

    # ── 2. Per-episode table ───────────────────────────────────────────────
    print("Step 2: Per-episode statistics")
    print(f"  {'ep':>3}  {'frames':>6}  {'pitch':>5}  {'yaw':>5}  "
          f"{'ins_mean':>8}  {'ins_std':>8}  {'ins%':>5}  {'pit%':>5}  {'yaw%':>5}")
    print("  " + "-" * 62)
    for s in summaries:
        print(
            f"  {s['episode_id']:>3}  {s['n_frames']:>6}  "
            f"{'Y' if s['has_pitch'] else 'N':>5}  "
            f"{'Y' if s['has_yaw'] else 'N':>5}  "
            f"{s['action_insert_mean']:>8.4f}  "
            f"{s['action_insert_std']:>8.4f}  "
            f"{s['insert_nonzero_frac']:>5.1%}  "
            f"{s['pitch_nonzero_frac']:>5.1%}  "
            f"{s['yaw_nonzero_frac']:>5.1%}"
        )
    print()

    # ── 3. Video frame count vs CSV alignment ─────────────────────────────
    print("Step 3: Video ↔ CSV frame count alignment")
    print("  (counting video frames by actual read — may take a few minutes)")
    mismatches = 0
    for eid in dataset.episode_ids:
        meta = load_episode_meta(eid)
        video_count, ok = check_video_frame_count(meta)
        status = "OK " if ok else "MISMATCH"
        if not ok:
            mismatches += 1
            print(f"  ep{eid:03d}: CSV={meta.n_frames}  video={video_count}  *** {status} ***")
        else:
            print(f"  ep{eid:03d}: frames={meta.n_frames}  {status}")

    print()
    if mismatches == 0:
        print(f"  All {len(dataset.episode_ids)} episodes aligned (0 mismatches)  PASS")
    else:
        print(f"  WARNING: {mismatches} mismatch(es) found  FAIL")
    print()

    # ── 4. Sample frame checks ─────────────────────────────────────────────
    print("Step 4: Sample frame shape / dtype / range checks")
    sample_episodes = [1, 10, 40, 71]
    all_pass = True
    for eid in sample_episodes:
        meta = load_episode_meta(eid)
        for label, fidx in [("first", 0), ("mid", meta.n_frames // 2), ("last", meta.n_frames - 1)]:
            ok, msg = sample_frame_check(eid, fidx)
            status = "PASS" if ok else "FAIL"
            if not ok:
                all_pass = False
            print(f"  ep{eid:03d} frame[{label:5s}={fidx:4d}]: {status}  {msg}")
    print()

    # ── 5. M1 vs M2 differential verification ─────────────────────────────
    print("Step 5: M1 / M2 differential roller verification (ep1)")
    import csv as csv_mod
    with open(f"/data/ERCP/ercp_access/data/cannulation/data/data (1).csv") as f:
        rows = list(csv_mod.reader(f))
    m1 = np.array([float(r[5])  for r in rows[1:]])
    m2 = np.array([float(r[9])  for r in rows[1:]])
    dm1 = np.diff(m1)
    dm2 = np.diff(m2)
    print(f"  M1 total: {m1[-1]-m1[0]:.2f} deg   M2 total: {m2[-1]-m2[0]:.2f} deg")
    print(f"  M1 mean delta: {dm1.mean():.4f}   M2 mean delta: {dm2.mean():.4f}")
    print(f"  M1+M2 mean (should ≈ 0 if differential): {(dm1+dm2).mean():.4f}")
    corr = np.corrcoef(dm1, dm2)[0, 1]
    print(f"  corr(dM1, dM2) = {corr:.4f}  (expect ≈ -1.0 for differential)")
    if corr < -0.8:
        print("  Differential confirmed: use M1 only for insert action  PASS")
    else:
        print("  WARNING: M1/M2 not clearly differential — check mapping  WARN")
    print()

    # ── 6. Global summary ─────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_frames = sum(s["n_frames"] for s in summaries)
    ins_all = np.concatenate([
        dataset.get_episode(eid).actions[:, 0]
        for eid in dataset.episode_ids
    ])
    pit_all = np.concatenate([
        dataset.get_episode(eid).actions[:, 1]
        for eid in dataset.episode_ids
    ])
    yaw_all = np.concatenate([
        dataset.get_episode(eid).actions[:, 2]
        for eid in dataset.episode_ids
    ])
    print(f"  Episodes loaded : {len(dataset.episode_ids)} / {N_EPISODES}")
    print(f"  Total frames    : {total_frames}")
    print(f"  Action insert   : mean_abs={np.mean(np.abs(ins_all)):.4f}  "
          f"nonzero={np.mean(np.abs(ins_all)>0.01):.1%}")
    print(f"  Action pitch    : mean_abs={np.mean(np.abs(pit_all)):.4f}  "
          f"nonzero={np.mean(np.abs(pit_all)>0.01):.1%}")
    print(f"  Action yaw      : mean_abs={np.mean(np.abs(yaw_all)):.4f}  "
          f"nonzero={np.mean(np.abs(yaw_all)>0.01):.1%}")
    print(f"  Video mismatches: {mismatches}")
    print(f"  Sample checks   : {'ALL PASS' if all_pass else 'SOME FAILED'}")
    overall = (mismatches == 0) and all_pass
    print()
    print(f"  OVERALL: {'PASS' if overall else 'FAIL'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
