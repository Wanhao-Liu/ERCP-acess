#!/usr/bin/env python3
"""
run_perception_loop.py — Full perception + FSM closed-loop validation on phantom episodes.

For each episode:
  RGB frame → YOLOv8 (best.pt) → StateBuilder → AccessState
  → PhaseManager → phase
  → InsertionGate → gate_open

Produces per-episode and aggregate statistics + timeline/summary plots.

Usage:
    cd /data/ERCP/ercp_access
    conda run -n ercp python scripts/perception/run_perception_loop.py
    conda run -n ercp python scripts/perception/run_perception_loop.py --episodes 1 5 12 23 45
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

WEIGHTS_DEFAULT = os.path.join(
    PROJECT_ROOT, "outputs", "perception", "yolo_papilla",
    "yolov8n_v1", "weights", "best.pt"
)
OUT_DIR_DEFAULT = os.path.join(PROJECT_ROOT, "outputs", "perception", "loop")
ALL_EPISODES = list(range(1, 72))      # episodes 1-71


# ── Per-episode processing ──────────────────────────────────────────────────

def process_episode(
    episode_id: int,
    state_builder,
    phase_manager,
    insertion_gate,
    safety_rules,
    conf_thresh: float = 0.25,
) -> dict:
    """
    Run perception + FSM on one phantom episode.
    Returns a dict of per-episode statistics.
    """
    from src.data.phantom_dataset import PhantomEpisode

    state_builder.reset()
    phase_manager.reset()
    safety_rules.reset()

    confs, readiness_vals, phases, gate_opens = [], [], [], []

    try:
        with PhantomEpisode(episode_id) as ep:
            for _frame_idx, (rgb, _action) in enumerate(ep.iter_frames()):
                access_state = state_builder.update(rgb)
                phase = phase_manager.update(access_state, {})
                gate_open = insertion_gate.check(access_state)

                confs.append(access_state.conf)
                readiness_vals.append(access_state.readiness)
                phases.append(phase)
                gate_opens.append(gate_open)

    except Exception as e:
        print(f"  WARNING: ep{episode_id:03d} error: {e}")
        return {"episode_id": episode_id, "error": str(e), "n_frames": 0}

    n = len(confs)
    if n == 0:
        return {"episode_id": episode_id, "n_frames": 0}

    confs_arr = np.array(confs)
    ready_arr = np.array(readiness_vals)
    phases_arr = np.array(phases)
    gate_arr = np.array(gate_opens)

    phase_dist = {
        str(p): float(np.mean(phases_arr == p)) for p in range(4)
    }

    return {
        "episode_id":         episode_id,
        "n_frames":           n,
        "detection_rate":     float(np.mean(confs_arr > conf_thresh)),
        "mean_conf":          float(np.mean(confs_arr)),
        "mean_readiness":     float(np.mean(ready_arr)),
        "max_readiness":      float(np.max(ready_arr)),
        "readiness_gt07_rate": float(np.mean(ready_arr >= 0.7)),
        "phase2_reached":     bool(np.any(phases_arr == 2)),
        "gate_ever_open":     bool(np.any(gate_arr)),
        "phase_dist":         phase_dist,
        # raw arrays saved for plotting (later)
        "_confs":             confs,
        "_readiness":         readiness_vals,
        "_phases":            phases,
    }


# ── Timeline plot ────────────────────────────────────────────────────────────

def plot_timeline(result: dict, out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    confs = result.get("_confs", [])
    readiness = result.get("_readiness", [])
    phases = result.get("_phases", [])
    if not confs:
        return

    t = np.arange(len(confs))
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    fig.suptitle(f"Episode {result['episode_id']:03d} — Perception Loop Timeline", fontsize=12)

    axes[0].plot(t, confs, color="#3498db", linewidth=0.8)
    axes[0].axhline(0.25, color="red", linestyle="--", linewidth=0.8, label="conf_thresh=0.25")
    axes[0].axhline(0.6,  color="orange", linestyle="--", linewidth=0.8, label="conf_min=0.60")
    axes[0].set_ylabel("Confidence")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(fontsize=7, loc="lower right")

    axes[1].plot(t, readiness, color="#2ecc71", linewidth=0.8)
    axes[1].axhline(0.7, color="red", linestyle="--", linewidth=0.8, label="readiness_min=0.70")
    axes[1].set_ylabel("Readiness")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=7, loc="lower right")

    phase_colors = {0: "#95a5a6", 1: "#3498db", 2: "#2ecc71", 3: "#e74c3c"}
    phase_labels = {0: "alignment", 1: "approach", 2: "insertion", 3: "recovery"}
    for p_val, color in phase_colors.items():
        mask = np.array(phases) == p_val
        axes[2].fill_between(t, p_val - 0.4, p_val + 0.4,
                             where=mask, color=color, alpha=0.7, label=phase_labels[p_val])
    axes[2].set_yticks([0, 1, 2, 3])
    axes[2].set_yticklabels(["align", "approach", "insert", "recover"], fontsize=8)
    axes[2].set_ylabel("Phase")
    axes[2].set_xlabel("Frame")
    axes[2].legend(fontsize=7, loc="upper right", ncol=4)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


# ── Summary plot ─────────────────────────────────────────────────────────────

def plot_summary(results: list[dict], out_path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    valid = [r for r in results if r.get("n_frames", 0) > 0 and "error" not in r]
    if not valid:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Perception Loop Summary — 71 Phantom Episodes", fontsize=13, fontweight="bold")

    # (0,0) Readiness distribution
    ax = axes[0, 0]
    all_ready = [r["mean_readiness"] for r in valid]
    ax.hist(all_ready, bins=20, color="#2ecc71", edgecolor="white")
    ax.axvline(0.7, color="red", linestyle="--", linewidth=1.2, label="readiness_min=0.70")
    ax.set_title("Mean Readiness per Episode")
    ax.set_xlabel("Mean Readiness")
    ax.set_ylabel("Episodes")
    ax.legend(fontsize=8)

    # (0,1) Detection rate distribution
    ax = axes[0, 1]
    det_rates = [r["detection_rate"] for r in valid]
    ax.hist(det_rates, bins=20, color="#3498db", edgecolor="white")
    ax.axvline(0.8, color="red", linestyle="--", linewidth=1.2, label="target=0.80")
    ax.set_title("Detection Rate per Episode")
    ax.set_xlabel("Detection Rate (conf > 0.25)")
    ax.set_ylabel("Episodes")
    ax.legend(fontsize=8)

    # (1,0) Phase distribution (stacked bar per episode)
    ax = axes[1, 0]
    ep_ids = [r["episode_id"] for r in valid]
    phase_colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c"]
    phase_labels = ["align", "approach", "insert", "recover"]
    bottoms = np.zeros(len(valid))
    for p in range(4):
        vals = [float(r["phase_dist"].get(str(p), 0)) for r in valid]
        ax.bar(range(len(valid)), vals, bottom=bottoms, color=phase_colors[p],
               label=phase_labels[p], width=1.0)
        bottoms += np.array(vals)
    ax.set_title("Phase Distribution per Episode")
    ax.set_xlabel("Episode index")
    ax.set_ylabel("Fraction")
    ax.legend(fontsize=8, loc="upper right")

    # (1,1) Gate-ever-open rate
    ax = axes[1, 1]
    gate_rate = np.mean([r["gate_ever_open"] for r in valid])
    phase2_rate = np.mean([r["phase2_reached"] for r in valid])
    labels = ["Gate ever open", "Phase 2 reached"]
    vals = [gate_rate, phase2_rate]
    bars = ax.bar(labels, vals, color=["#2ecc71", "#3498db"], edgecolor="white")
    ax.set_ylim(0, 1.1)
    ax.set_title("Episode-level Success Indicators")
    ax.set_ylabel("Fraction of episodes")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[loop] Summary plot saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main(
    weights: str,
    episodes: list[int],
    out_dir: str,
    conf_thresh: float,
    timeline_n: int,
) -> None:
    from src.perception.state_builder import StateBuilder
    from src.gating.phase_manager import PhaseManager
    from src.gating.insertion_gate import InsertionGate, SafetyRules

    os.makedirs(out_dir, exist_ok=True)

    print(f"[loop] Weights:   {weights}")
    print(f"[loop] Episodes:  {len(episodes)} total")
    print(f"[loop] Output:    {out_dir}")

    state_builder = StateBuilder(
        use_oracle=False,
        detector_weights=weights,
        detector_conf_thresh=conf_thresh,
        detector_device="cuda",
    )
    phase_manager = PhaseManager(conf_thresh=0.5, recovery_steps=8,
                                  insertion_gate=InsertionGate())
    insertion_gate = InsertionGate()
    safety_rules = SafetyRules()

    results = []
    for i, ep_id in enumerate(episodes):
        print(f"  [{i+1:3d}/{len(episodes)}] ep{ep_id:03d}...", end=" ", flush=True)
        r = process_episode(ep_id, state_builder, phase_manager,
                            insertion_gate, safety_rules, conf_thresh)
        n = r.get("n_frames", 0)
        if n > 0 and "error" not in r:
            print(f"n={n}, conf={r['mean_conf']:.2f}, "
                  f"ready={r['mean_readiness']:.2f}, "
                  f"gate={'✓' if r['gate_ever_open'] else '✗'}")
        else:
            print(f"SKIP ({r.get('error', 'n=0')})")
        results.append(r)

    # ── Timeline plots for first N valid episodes ─────────────────────────
    valid_results = [r for r in results if r.get("n_frames", 0) > 0 and "error" not in r]
    for r in valid_results[:timeline_n]:
        plot_path = os.path.join(out_dir, f"timeline_ep{r['episode_id']:03d}.png")
        plot_timeline(r, plot_path)
        print(f"[loop] Timeline saved: {plot_path}")

    # ── Strip raw arrays before saving JSON ──────────────────────────────
    for r in results:
        r.pop("_confs", None)
        r.pop("_readiness", None)
        r.pop("_phases", None)

    # Save per-episode JSON
    per_ep_path = os.path.join(out_dir, "per_episode.json")
    with open(per_ep_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[loop] Per-episode saved: {per_ep_path}")

    # ── Aggregate statistics ──────────────────────────────────────────────
    valid = [r for r in results if r.get("n_frames", 0) > 0 and "error" not in r]
    if not valid:
        print("[loop] No valid episodes.")
        return

    keys = ["detection_rate", "mean_conf", "mean_readiness",
            "max_readiness", "readiness_gt07_rate"]
    agg = {"n_episodes": len(valid)}
    for k in keys:
        vals = [r[k] for r in valid]
        agg[k + "_mean"] = float(np.mean(vals))
        agg[k + "_std"]  = float(np.std(vals))
        agg[k + "_median"] = float(np.median(vals))
    agg["gate_ever_open_rate"]  = float(np.mean([r["gate_ever_open"] for r in valid]))
    agg["phase2_reached_rate"]  = float(np.mean([r["phase2_reached"] for r in valid]))

    agg_path = os.path.join(out_dir, "aggregate.json")
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    # ── Summary plot ──────────────────────────────────────────────────────
    plot_summary(valid_results, os.path.join(out_dir, "loop_summary.png"))

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  PERCEPTION LOOP — AGGREGATE RESULTS")
    print("=" * 55)
    print(f"  Episodes evaluated        : {len(valid)}")
    print(f"  detection_rate (mean)     : {agg['detection_rate_mean']:.3f}")
    print(f"  mean_conf      (mean)     : {agg['mean_conf_mean']:.3f}")
    print(f"  mean_readiness (mean)     : {agg['mean_readiness_mean']:.3f}")
    print(f"  max_readiness  (mean)     : {agg['max_readiness_mean']:.3f}")
    print(f"  readiness≥0.7  (mean)     : {agg['readiness_gt07_rate_mean']:.3f}")
    print(f"  gate_ever_open (rate)     : {agg['gate_ever_open_rate']:.1%}")
    print(f"  phase2_reached (rate)     : {agg['phase2_reached_rate']:.1%}")
    print("=" * 55)
    print(f"\n[loop] Results saved to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full perception + FSM loop on phantom episodes.")
    parser.add_argument("--weights",     default=WEIGHTS_DEFAULT)
    parser.add_argument("--episodes",    nargs="+", type=int, default=ALL_EPISODES)
    parser.add_argument("--out_dir",     default=OUT_DIR_DEFAULT)
    parser.add_argument("--conf_thresh", type=float, default=0.25)
    parser.add_argument("--timeline_n", type=int, default=5,
                        help="Number of episodes for which to save timeline plots.")
    args = parser.parse_args()
    main(args.weights, args.episodes, args.out_dir, args.conf_thresh, args.timeline_n)
