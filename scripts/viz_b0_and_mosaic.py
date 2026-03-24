"""
Generate B0 trajectory visualizations and papilla mosaic.
Outputs saved to outputs/figures/
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from PIL import Image

OUT_DIR = "/data/ERCP/ercp_access/outputs/figures"
os.makedirs(OUT_DIR, exist_ok=True)

CSV = "/data/ERCP/ercp_access/outputs/scripted/val/val_episodes.csv"
METRICS = {
    "access_success_rate": 1.0,
    "mean_steps_to_access": 24.05,
    "mean_insert_attempts": 5.0,
    "target_loss_rate": 0.0,
    "off_axis_rate": 0.53,
    "recovery_rate": 0.0,
    "abort_rate": 0.0,
    "mean_alignment_error": 0.1500,
}
PAPILLA_DIR = "/data/ERCP/ercp_access/data/Duodenal Papilla/Duodenal Papilla_Protruding"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  B0  TRAJECTORY  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_csv(CSV)

BLUE   = "#2563EB"
GREEN  = "#16A34A"
ORANGE = "#EA580C"
GRAY   = "#9CA3AF"
BG     = "#F8FAFC"
CARD   = "#FFFFFF"

fig = plt.figure(figsize=(18, 11), facecolor=BG)
fig.suptitle(
    "B0 Scripted Baseline — Evaluation Dashboard  (N = 100 episodes)",
    fontsize=17, fontweight="bold", y=0.97, color="#1E293B",
)

gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38,
                       left=0.06, right=0.97, top=0.91, bottom=0.08)

# ── (A) Steps distribution ──────────────────────────────────────────────────
ax_steps = fig.add_subplot(gs[0, 0])
ax_steps.set_facecolor(CARD)
bins = np.arange(df["steps"].min() - 0.5, df["steps"].max() + 1.5, 1)
ax_steps.hist(df["steps"], bins=bins, color=BLUE, edgecolor="white", linewidth=0.5)
ax_steps.axvline(df["steps"].mean(), color=ORANGE, lw=2, linestyle="--",
                 label=f"Mean = {df['steps'].mean():.1f}")
ax_steps.set_xlabel("Steps to Access", fontsize=10)
ax_steps.set_ylabel("Episode Count", fontsize=10)
ax_steps.set_title("Steps Distribution", fontsize=11, fontweight="bold")
ax_steps.legend(fontsize=9)
ax_steps.spines[["top", "right"]].set_visible(False)

# ── (B) Alignment error distribution ────────────────────────────────────────
ax_err = fig.add_subplot(gs[0, 1])
ax_err.set_facecolor(CARD)
ax_err.hist(df["mean_alignment_error"], bins=20, color=ORANGE,
            edgecolor="white", linewidth=0.5)
ax_err.axvline(df["mean_alignment_error"].mean(), color=BLUE, lw=2,
               linestyle="--",
               label=f"Mean = {df['mean_alignment_error'].mean():.3f}")
ax_err.set_xlabel("Mean Alignment Error", fontsize=10)
ax_err.set_ylabel("Episode Count", fontsize=10)
ax_err.set_title("Alignment Error Distribution", fontsize=11, fontweight="bold")
ax_err.legend(fontsize=9)
ax_err.spines[["top", "right"]].set_visible(False)

# ── (C) Insert attempts distribution ────────────────────────────────────────
ax_ins = fig.add_subplot(gs[0, 2])
ax_ins.set_facecolor(CARD)
bins_i = np.arange(df["num_insert_attempts"].min() - 0.5,
                   df["num_insert_attempts"].max() + 1.5, 1)
ax_ins.hist(df["num_insert_attempts"], bins=bins_i, color=GREEN,
            edgecolor="white", linewidth=0.5)
ax_ins.axvline(df["num_insert_attempts"].mean(), color=ORANGE, lw=2,
               linestyle="--",
               label=f"Mean = {df['num_insert_attempts'].mean():.1f}")
ax_ins.set_xlabel("Insert Attempts", fontsize=10)
ax_ins.set_ylabel("Episode Count", fontsize=10)
ax_ins.set_title("Insertion Attempts Distribution", fontsize=11, fontweight="bold")
ax_ins.legend(fontsize=9)
ax_ins.spines[["top", "right"]].set_visible(False)

# ── (D) Key metrics summary (bar) ───────────────────────────────────────────
ax_bar = fig.add_subplot(gs[0, 3])
ax_bar.set_facecolor(CARD)
metric_names  = ["Success\nRate", "Target\nLoss Rate", "Off-Axis\nRate",
                 "Recovery\nRate", "Abort\nRate"]
metric_values = [METRICS["access_success_rate"],
                 METRICS["target_loss_rate"],
                 METRICS["off_axis_rate"],
                 METRICS["recovery_rate"],
                 METRICS["abort_rate"]]
colors_bar = [GREEN, ORANGE, ORANGE, ORANGE, ORANGE]
colors_bar[0] = GREEN  # success in green
bars = ax_bar.bar(metric_names, metric_values, color=colors_bar,
                  edgecolor="white", linewidth=0.8, width=0.55)
for bar, val in zip(bars, metric_values):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.02, f"{val:.0%}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_bar.set_ylim(0, 1.18)
ax_bar.set_ylabel("Rate", fontsize=10)
ax_bar.set_title("Key Metrics Summary", fontsize=11, fontweight="bold")
ax_bar.spines[["top", "right"]].set_visible(False)

# ── (E) Steps over episodes (scatter + rolling mean) ────────────────────────
ax_traj = fig.add_subplot(gs[1, :2])
ax_traj.set_facecolor(CARD)
ep_ids = np.arange(len(df))
ax_traj.scatter(ep_ids, df["steps"], s=18, color=BLUE, alpha=0.5, zorder=2,
                label="Episode steps")
roll = pd.Series(df["steps"].values).rolling(10, center=True).mean()
ax_traj.plot(ep_ids, roll, color=ORANGE, lw=2.5, zorder=3,
             label="Rolling mean (w=10)")
ax_traj.axhline(df["steps"].mean(), color=GREEN, lw=1.5, linestyle="--",
                label=f"Global mean = {df['steps'].mean():.1f}")
ax_traj.set_xlabel("Episode Index", fontsize=10)
ax_traj.set_ylabel("Steps to Access", fontsize=10)
ax_traj.set_title("Steps per Episode (All 100 Episodes)", fontsize=11, fontweight="bold")
ax_traj.legend(fontsize=9, loc="upper right")
ax_traj.spines[["top", "right"]].set_visible(False)

# ── (F) Alignment error vs steps (scatter) ──────────────────────────────────
ax_scatter = fig.add_subplot(gs[1, 2])
ax_scatter.set_facecolor(CARD)
sc = ax_scatter.scatter(df["steps"], df["mean_alignment_error"],
                        c=df["num_insert_attempts"], cmap="plasma",
                        s=28, alpha=0.75, zorder=2)
cb = fig.colorbar(sc, ax=ax_scatter, pad=0.02)
cb.set_label("Insert Attempts", fontsize=8)
ax_scatter.set_xlabel("Steps to Access", fontsize=10)
ax_scatter.set_ylabel("Mean Alignment Error", fontsize=10)
ax_scatter.set_title("Alignment Error vs. Steps\n(color = insert attempts)",
                     fontsize=11, fontweight="bold")
ax_scatter.spines[["top", "right"]].set_visible(False)

# ── (G) Event flag summary ───────────────────────────────────────────────────
ax_evt = fig.add_subplot(gs[1, 3])
ax_evt.set_facecolor(CARD)
evt_cols  = ["off_axis_count", "recovery_count",
             "target_loss_count", "unsafe_insert_attempts"]
evt_labels = ["Off-Axis\nEvents", "Recovery\nEvents",
              "Target\nLoss", "Unsafe\nInserts"]
evt_totals = [df[c].sum() for c in evt_cols]
bar_colors = [ORANGE, ORANGE, ORANGE, "#DC2626"]
bars2 = ax_evt.bar(evt_labels, evt_totals, color=bar_colors,
                   edgecolor="white", linewidth=0.8, width=0.55)
for bar, val in zip(bars2, evt_totals):
    ax_evt.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3, str(int(val)),
                ha="center", va="bottom", fontsize=9, fontweight="bold")
ax_evt.set_ylabel("Total Count (across 100 eps)", fontsize=9)
ax_evt.set_title("Safety Event Counts", fontsize=11, fontweight="bold")
ax_evt.spines[["top", "right"]].set_visible(False)

out_b0 = os.path.join(OUT_DIR, "b0_dashboard.png")
fig.savefig(out_b0, dpi=150, bbox_inches="tight", facecolor=BG)
plt.close(fig)
print(f"[✓] B0 dashboard saved → {out_b0}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  PAPILLA  IMAGE  MOSAIC
# ─────────────────────────────────────────────────────────────────────────────

paths = sorted(glob.glob(os.path.join(PAPILLA_DIR, "*.jpg")))
print(f"[i] Found {len(paths)} papilla images")

# pick up to 80 images, arrange as 10 × 8
N = min(80, len(paths))
paths = paths[:N]
COLS, ROWS = 10, 8
THUMB = 160  # px per thumbnail

mosaic_w = COLS * THUMB
mosaic_h = ROWS * THUMB
canvas = Image.new("RGB", (mosaic_w, mosaic_h), (245, 247, 250))

for idx, p in enumerate(paths):
    r, c = divmod(idx, COLS)
    if r >= ROWS:
        break
    try:
        img = Image.open(p).convert("RGB")
        img.thumbnail((THUMB, THUMB), Image.LANCZOS)
        # center-crop to exact THUMB × THUMB
        w, h = img.size
        left = (w - THUMB) // 2
        top  = (h - THUMB) // 2
        img = img.crop((left, top, left + THUMB, top + THUMB))
        canvas.paste(img, (c * THUMB, r * THUMB))
    except Exception as e:
        print(f"  [!] skip {p}: {e}")

# add thin grid lines
from PIL import ImageDraw
draw = ImageDraw.Draw(canvas)
for ci in range(1, COLS):
    draw.line([(ci * THUMB, 0), (ci * THUMB, mosaic_h)], fill=(200, 205, 215), width=1)
for ri in range(1, ROWS):
    draw.line([(0, ri * THUMB), (mosaic_w, ri * THUMB)], fill=(200, 205, 215), width=1)

# add title bar
title_h = 52
full = Image.new("RGB", (mosaic_w, mosaic_h + title_h), (30, 41, 59))
full.paste(canvas, (0, title_h))
draw2 = ImageDraw.Draw(full)
try:
    from PIL import ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    font_sm = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
except Exception:
    font = font_sm = ImageFont.load_default()

draw2.text((18, 10), "Duodenal Papilla — Real Endoscopic Images", fill=(255, 255, 255), font=font)
draw2.text((18, 36), f"Protruding morphology  ·  {N} frames sampled from phantom dataset",
           fill=(148, 163, 184), font=font_sm)

out_mosaic = os.path.join(OUT_DIR, "papilla_mosaic.png")
full.save(out_mosaic, quality=95)
print(f"[✓] Papilla mosaic saved  → {out_mosaic}")
