#!/usr/bin/env python3
"""
Split annotated dataset into YOLO train/val structure.
Moves (not copies) files from data/dataset/ → data/yolo_dataset/
  - Static images: random 80/20 shuffle
  - Video frames: split by episode to prevent temporal leakage
"""
import os, shutil, random
from pathlib import Path
from collections import defaultdict

random.seed(42)

BASE       = Path(__file__).resolve().parents[2] / "data"
SRC_STATIC = BASE / "dataset/static_prepared"
SRC_VIDEO  = BASE / "dataset/video_frames"
DST        = BASE / "yolo_dataset"

# Create dirs
for split in ("train", "val"):
    (DST / "images" / split).mkdir(parents=True, exist_ok=True)
    (DST / "labels" / split).mkdir(parents=True, exist_ok=True)


def move_pair(img: Path, split: str):
    lbl = img.with_suffix(".txt")
    shutil.move(str(img), DST / "images" / split / img.name)
    if lbl.exists():
        shutil.move(str(lbl), DST / "labels" / split / lbl.name)


# ── Static: 80/20 random split ───────────────────────────────────────
static_pairs = sorted(
    p for p in SRC_STATIC.glob("*.jpg") if p.with_suffix(".txt").exists()
)
random.shuffle(static_pairs)
cut = int(len(static_pairs) * 0.8)
for p in static_pairs[:cut]:
    move_pair(p, "train")
for p in static_pairs[cut:]:
    move_pair(p, "val")
print(f"Static  → train={cut}, val={len(static_pairs)-cut}  (total={len(static_pairs)})")

# ── Video: split by episode ──────────────────────────────────────────
ep_map = defaultdict(list)
for p in sorted(SRC_VIDEO.glob("*.jpg")):
    lbl = p.with_suffix(".txt")
    if not lbl.exists():
        continue
    if not lbl.read_text().strip():    # skip empty labels
        continue
    ep = p.name.split("_")[0]          # e.g. "ep001"
    ep_map[ep].append(p)

episodes = sorted(ep_map.keys())
random.shuffle(episodes)
cut_ep = int(len(episodes) * 0.8)
train_ep = set(episodes[:cut_ep])
val_ep   = set(episodes[cut_ep:])

n_train = n_val = 0
for ep, frames in ep_map.items():
    split = "train" if ep in train_ep else "val"
    for p in frames:
        move_pair(p, split)
        if split == "train":
            n_train += 1
        else:
            n_val += 1

print(f"Video   → train={n_train} ({len(train_ep)} eps), val={n_val} ({len(val_ep)} eps)")

# ── Final summary ────────────────────────────────────────────────────
print()
for split in ("train", "val"):
    ni = len(list((DST / "images" / split).glob("*.jpg")))
    nl = len(list((DST / "labels" / split).glob("*.txt")))
    print(f"  {split:5s}  images={ni}, labels={nl}")
print(f"\nDataset written to: {DST}")
