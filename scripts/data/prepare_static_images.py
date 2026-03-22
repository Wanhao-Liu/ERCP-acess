"""
prepare_static_images.py — Copy static papilla images to annotation staging directory.

Copies all images from the three subdirectories of 'Duodenal Papilla/' into
data/annotation_staging/static_prepared/, renaming them as:
    static_normal_NNN.jpg
    static_flat_NNN.jpg
    static_protruding_NNN.jpg

Generates data/annotation_staging/static_manifest.json.

Usage:
    python scripts/data/prepare_static_images.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

PROJECT_ROOT = "/data/ERCP/ercp_access"
sys.path.insert(0, "/root/.local/lib/python3.10/site-packages")

# ── Configuration ─────────────────────────────────────────────────────────────
STATIC_SRC_ROOT = os.path.join(PROJECT_ROOT, "data", "Duodenal Papilla")
OUT_DIR         = os.path.join(PROJECT_ROOT, "data", "annotation_staging", "static_prepared")
MANIFEST_PATH   = os.path.join(PROJECT_ROOT, "data", "annotation_staging", "static_manifest.json")

# Map subdirectory name suffix -> type label
SUBDIR_TYPE_MAP = {
    "Duodenal Papilla_Normal":     "normal",
    "Duodenal Papilla_Flat":       "flat",
    "Duodenal Papilla_Protruding": "protruding",
}


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    manifest: list[dict] = []
    total_copied = 0

    for subdir_name, type_label in SUBDIR_TYPE_MAP.items():
        src_dir = os.path.join(STATIC_SRC_ROOT, subdir_name)
        if not os.path.isdir(src_dir):
            print(f"  WARNING: source directory not found: {src_dir}")
            continue

        # Collect and sort all .jpg files
        files = sorted(
            f for f in os.listdir(src_dir)
            if f.lower().endswith(".jpg") or f.lower().endswith(".jpeg")
        )

        print(f"Copying {len(files)} images from {subdir_name} ...")

        for i, fname in enumerate(files, start=1):
            src_path = os.path.join(src_dir, fname)
            dst_name = f"static_{type_label}_{i:03d}.jpg"
            dst_path = os.path.join(OUT_DIR, dst_name)

            shutil.copy2(src_path, dst_path)

            manifest.append({
                "file_path":    dst_path,
                "file_name":    dst_name,
                "type":         type_label,
                "source_file":  fname,
                "source_dir":   src_dir,
            })
            total_copied += 1

    # Write manifest
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone. Total images copied: {total_copied}")
    print(f"Manifest written to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
