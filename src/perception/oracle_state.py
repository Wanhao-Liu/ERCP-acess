"""
Oracle state computation from Isaac Sim ground truth.

Used in Phase 0 to bypass perception and validate control logic.
"""
from __future__ import annotations

from typing import Dict, Any

import numpy as np


def compute_oracle_gt_info(
    scope_tip_pos: np.ndarray,   # [3] world coords
    papilla_pos: np.ndarray,     # [3] world coords
    camera_pos: np.ndarray,      # [3] world coords
    image_width: int = 512,
    image_height: int = 512,
    ref_distance: float = 0.05,  # reference distance for scale=1.0 (5cm)
    ref_scale: float = 0.15,
) -> Dict[str, Any]:
    """
    Compute oracle gt_info dict for StateBuilder.

    Projects papilla_pos into camera frame and computes:
      e_x, e_y: normalized pixel offset from image center [-1, 1]
      scale: ref_distance / current_distance, clamped [0, 1]
      conf: 1.0 (oracle has no uncertainty)

    Args:
        scope_tip_pos: Scope tip position in world frame.
        papilla_pos: Papilla target position in world frame.
        camera_pos: Camera position in world frame (approx = scope_tip_pos).
        image_width, image_height: Camera resolution.
        ref_distance: Distance at which scale=1.0.

    Returns:
        dict with keys: e_x, e_y, scale, conf
    """
    # Vector from camera to papilla in world frame
    delta = papilla_pos - camera_pos  # [3]

    # Simple projection: assume camera looks along +X axis (world frame)
    # e_x corresponds to Y-axis deviation, e_y to Z-axis deviation
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        return {"e_x": 0.0, "e_y": 0.0, "scale": 1.0, "conf": 1.0}

    # Normalized lateral offsets (Y and Z components)
    e_x = float(np.clip(delta[1] / ref_scale, -1.0, 1.0))
    e_y = float(np.clip(delta[2] / ref_scale, -1.0, 1.0))

    # Scale: inversely proportional to distance
    scale = float(np.clip(ref_distance / (dist + 1e-6), 0.0, 1.0))

    return {"e_x": e_x, "e_y": e_y, "scale": scale, "conf": 1.0}
