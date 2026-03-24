#!/usr/bin/env python3
"""
Baseline comparison visualization.

Generates a multi-panel figure comparing all available baselines:
  - Bar charts for key metrics
  - Episode-level distributions (steps, alignment error, readiness)
  - Per-episode success/failure scatter

Usage:
    cd /data/ERCP/ercp_access
    conda run -n ercp python scripts/eval/visualize_baselines.py
    conda run -n ercp python scripts/eval/visualize_baselines.py --out_dir outputs/figures
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Baseline registry ──────────────────────────────────────────────────────────
# Add new baselines here as they become available
BASELINES = {
    "B0\nScripted":  "outputs/scripted/val",
    "B2\nPPO-State": "outputs/b2_ppo/b2_ppo_state/final_eval/val",
    "B1\nSeg+PID":  "outputs/b1_pid/val",
    # Future baselines (uncomment when available):
    # "B3\nw/o Gate": "outputs/b3_no_gate/val",
    # "B4\nImg-only": "outputs/b4_image/val",
    # "Ours\nV2":     "outputs/ours_v2/val",
}

# Color scheme
COLORS = {
    "B0\nScripted":  "#2ecc71",   # green
    "B2\nPPO-State": "#e74c3c",   # red
    "B1\nSeg+PID":   "#3498db",   # blue
    "B3\nw/o Gate":  "#f39c12",   # orange
    "B4\nImg-only":  "#9b59b6",   # purple
    "Ours\nV2":      "#1abc9c",   # teal
}
DEFAULT_COLOR = "#95a5a6"


def load_baseline(run_dir: str) -> tuple[dict, pd.DataFrame] | None:
    metrics_path = os.path.join(run_dir, "val_run_metrics.json")
    episodes_path = os.path.join(run_dir, "val_episodes.csv")
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as f:
        metrics = json.load(f)
    episodes = pd.read_csv(episodes_path) if os.path.exists(episodes_path) else pd.DataFrame()
    return metrics, episodes


def get_color(name: str) -> str:
    return COLORS.get(name, DEFAULT_COLOR)


def plot_metric_bars(ax, names, values, title, fmt=".0%", color_fn=None, ylim=None, hline=None):
    colors = [get_color(n) for n in names]
    bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_ylim(ylim or (0, 1.05))
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    if hline is not None:
        ax.axhline(hline, color="#2c3e50", linestyle="--", linewidth=1, alpha=0.5)
    for bar, val in zip(bars, values):
        if val > 0:
            label = f"{val:{fmt[1:]}}" if fmt.startswith(".") else format(val, fmt)
            if "%" in fmt:
                label = f"{val:.0%}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                label, ha="center", va="bottom", fontsize=9, fontweight="bold"
            )
    return ax


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="outputs/figures")
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(os.path.join(base, args.out_dir), exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    data = {}
    for name, rel_dir in BASELINES.items():
        full_dir = os.path.join(base, rel_dir)
        result = load_baseline(full_dir)
        if result is not None:
            data[name] = result
            print(f"  Loaded: {name.replace(chr(10), ' ')}")
        else:
            print(f"  Missing: {name.replace(chr(10), ' ')} ({full_dir})")

    if not data:
        print("No baseline data found.")
        return

    names = list(data.keys())
    metrics_list = [data[n][0] for n in names]
    episodes_list = [data[n][1] for n in names]

    # ── Figure 1: Metrics comparison (main figure) ────────────────────────────
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("Baseline Comparison — ERCP Autonomous Biliary Access",
                 fontsize=14, fontweight="bold", y=0.98)

    gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.35,
                  top=0.92, bottom=0.08, left=0.07, right=0.97)

    # Row 0: Primary metrics
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Row 1: Secondary metrics
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    ax5 = fig.add_subplot(gs[1, 2])

    # Row 2: Episode distributions
    ax6 = fig.add_subplot(gs[2, 0])
    ax7 = fig.add_subplot(gs[2, 1])
    ax8 = fig.add_subplot(gs[2, 2])

    # ── Primary metrics ────────────────────────────────────────────────────────
    plot_metric_bars(
        ax0, names,
        [m["access_success_rate"] for m in metrics_list],
        "Access Success Rate ↑",
        ylim=(0, 1.15),
    )
    ax0.set_ylabel("Rate", fontsize=9)

    plot_metric_bars(
        ax1, names,
        [m["abort_rate"] for m in metrics_list],
        "Abort Rate ↓",
        ylim=(0, 1.15),
    )

    plot_metric_bars(
        ax2, names,
        [m["off_axis_rate"] for m in metrics_list],
        "Off-Axis Rate ↓",
        ylim=(0, 1.15),
    )

    # ── Secondary metrics ──────────────────────────────────────────────────────
    plot_metric_bars(
        ax3, names,
        [m["mean_alignment_error"] for m in metrics_list],
        "Mean Alignment Error ↓",
        fmt=".3f",
        ylim=(0, max(m["mean_alignment_error"] for m in metrics_list) * 1.3),
    )
    ax3.set_ylabel("Error (norm.)", fontsize=9)

    # Steps to access (only meaningful for successful baselines)
    steps_vals = [m["mean_steps_to_access"] for m in metrics_list]
    plot_metric_bars(
        ax4, names,
        steps_vals,
        "Mean Steps to Access ↓\n(successful episodes only)",
        fmt=".1f",
        ylim=(0, max(steps_vals) * 1.35 + 1),
    )
    ax4.set_ylabel("Steps", fontsize=9)

    insert_vals = [m["mean_insert_attempts"] for m in metrics_list]
    plot_metric_bars(
        ax5, names,
        insert_vals,
        "Mean Insert Attempts ↓",
        fmt=".1f",
        ylim=(0, max(insert_vals) * 1.35 + 1),
    )
    ax5.set_ylabel("Attempts", fontsize=9)

    # ── Episode-level distributions ────────────────────────────────────────────
    # Steps distribution
    ax6.set_title("Steps Distribution\n(per episode)", fontsize=11, fontweight="bold", pad=8)
    ax6.set_xlabel("Steps", fontsize=9)
    ax6.set_ylabel("Episodes", fontsize=9)
    ax6.spines[["top", "right"]].set_visible(False)
    ax6.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax6.set_axisbelow(True)
    for name, eps in zip(names, episodes_list):
        if "steps" in eps.columns and len(eps) > 0:
            steps = eps["steps"].values
            ax6.hist(steps, bins=20, alpha=0.55, color=get_color(name),
                     label=name.replace("\n", " "), edgecolor="white", linewidth=0.5)
    ax6.legend(fontsize=8, framealpha=0.7)

    # Alignment error distribution
    ax7.set_title("Alignment Error Distribution\n(per episode mean)", fontsize=11,
                  fontweight="bold", pad=8)
    ax7.set_xlabel("Mean Alignment Error", fontsize=9)
    ax7.set_ylabel("Episodes", fontsize=9)
    ax7.spines[["top", "right"]].set_visible(False)
    ax7.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax7.set_axisbelow(True)
    for name, eps in zip(names, episodes_list):
        if "mean_alignment_error" in eps.columns and len(eps) > 0:
            ax7.hist(eps["mean_alignment_error"].values, bins=20, alpha=0.55,
                     color=get_color(name), label=name.replace("\n", " "),
                     edgecolor="white", linewidth=0.5)
    ax7.legend(fontsize=8, framealpha=0.7)

    # Success / Fail scatter: steps vs alignment error
    ax8.set_title("Steps vs Alignment Error\n(● success  ✕ failure)", fontsize=11,
                  fontweight="bold", pad=8)
    ax8.set_xlabel("Mean Alignment Error", fontsize=9)
    ax8.set_ylabel("Steps", fontsize=9)
    ax8.spines[["top", "right"]].set_visible(False)
    ax8.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
    ax8.set_axisbelow(True)
    for name, eps in zip(names, episodes_list):
        if len(eps) == 0 or "success" not in eps.columns:
            continue
        c = get_color(name)
        lbl = name.replace("\n", " ")
        succ = eps[eps["success"] == 1]
        fail = eps[eps["success"] == 0]
        if len(succ):
            ax8.scatter(succ["mean_alignment_error"], succ["steps"],
                        color=c, marker="o", s=18, alpha=0.7, label=f"{lbl} ✓")
        if len(fail):
            ax8.scatter(fail["mean_alignment_error"], fail["steps"],
                        color=c, marker="x", s=20, alpha=0.5, label=f"{lbl} ✗")
    ax8.legend(fontsize=7, framealpha=0.7, ncol=1)

    out_path = os.path.join(base, args.out_dir, "baseline_comparison.png")
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n[viz] Saved: {out_path}")

    # ── Figure 2: Summary table ────────────────────────────────────────────────
    fig2, ax = plt.subplots(figsize=(len(names) * 2.2 + 1.5, 4.5))
    ax.axis("off")
    fig2.suptitle("Baseline Metrics Summary", fontsize=13, fontweight="bold", y=0.97)

    col_labels = [
        "Metric",
        *[n.replace("\n", " ") for n in names],
    ]
    row_data = [
        ["Success Rate ↑",     *[f"{m['access_success_rate']:.0%}" for m in metrics_list]],
        ["Abort Rate ↓",       *[f"{m['abort_rate']:.0%}"           for m in metrics_list]],
        ["Off-Axis Rate ↓",    *[f"{m['off_axis_rate']:.0%}"        for m in metrics_list]],
        ["Mean Steps ↓",       *[f"{m['mean_steps_to_access']:.1f}" for m in metrics_list]],
        ["Insert Attempts ↓",  *[f"{m['mean_insert_attempts']:.1f}" for m in metrics_list]],
        ["Align. Error ↓",     *[f"{m['mean_alignment_error']:.3f}" for m in metrics_list]],
        ["Target Loss Rate ↓", *[f"{m['target_loss_rate']:.0%}"     for m in metrics_list]],
        ["Recovery Rate",      *[f"{m['recovery_rate']:.0%}"        for m in metrics_list]],
    ]

    table = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)

    # Style header
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e50")
        cell.set_text_props(color="white", fontweight="bold")

    # Style metric column
    for i in range(1, len(row_data) + 1):
        table[i, 0].set_facecolor("#ecf0f1")
        table[i, 0].set_text_props(fontweight="bold")

    # Color-code success rate row
    for j, name in enumerate(names):
        cell = table[1, j + 1]  # success rate row
        val = metrics_list[j]["access_success_rate"]
        if val >= 0.8:
            cell.set_facecolor("#d5f5e3")
        elif val >= 0.4:
            cell.set_facecolor("#fef9e7")
        else:
            cell.set_facecolor("#fadbd8")

    # Alternating row colors
    for i in range(1, len(row_data) + 1):
        for j in range(len(col_labels)):
            if i % 2 == 0 and j > 0:
                current = table[i, j].get_facecolor()
                if list(current[:3]) == [1.0, 1.0, 1.0]:
                    table[i, j].set_facecolor("#f8f9fa")

    out_path2 = os.path.join(base, args.out_dir, "baseline_table.png")
    fig2.savefig(out_path2, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"[viz] Saved: {out_path2}")

    # ── Print summary to terminal ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BASELINE METRICS SUMMARY")
    print("=" * 60)
    header = f"{'Metric':<22}" + "".join(f"{n.replace(chr(10),' '):>14}" for n in names)
    print(header)
    print("-" * len(header))
    rows = [
        ("Success Rate ↑",    "access_success_rate",   ".0%"),
        ("Abort Rate ↓",      "abort_rate",             ".0%"),
        ("Off-Axis Rate ↓",   "off_axis_rate",          ".0%"),
        ("Mean Steps ↓",      "mean_steps_to_access",   ".1f"),
        ("Insert Attempts ↓", "mean_insert_attempts",   ".1f"),
        ("Align. Error ↓",    "mean_alignment_error",   ".4f"),
    ]
    for label, key, fmt in rows:
        vals = [format(m[key], fmt) for m in metrics_list]
        print(f"  {label:<20}" + "".join(f"{v:>14}" for v in vals))
    print("=" * 60)
    print(f"\nFigures saved to: {os.path.join(base, args.out_dir)}/")


if __name__ == "__main__":
    main()
