"""
ActionPostprocessor — clip → rate_limit → phase_mask → gating.
ActionMapper — normalized [-1,1]^3 → physical units.
"""
from __future__ import annotations

import numpy as np


class ActionPostprocessor:
    """
    Pipeline: clip → rate_limit → phase_mask → gating → output

    Phase masks:
      phase 0 (alignment): block insert (action[0] = 0)
      phase 3 (recovery):  block pitch/yaw (action[1:] = 0), force retreat
    """

    def __init__(
        self,
        max_delta_per_step: float = 0.4,
        recovery_insert: float = -0.8,
    ):
        self.max_delta_per_step = max_delta_per_step
        self.recovery_insert = recovery_insert
        self._prev_action = np.zeros(3, dtype=np.float32)

    def reset(self):
        self._prev_action = np.zeros(3, dtype=np.float32)

    def process(
        self,
        raw_action: np.ndarray,
        phase: int,
        gating: int,
    ) -> np.ndarray:
        action = np.clip(raw_action, -1.0, 1.0).astype(np.float32)

        # Rate limit
        delta = action - self._prev_action
        delta = np.clip(delta, -self.max_delta_per_step, self.max_delta_per_step)
        action = self._prev_action + delta

        # Phase mask
        if phase == 0:
            action[0] = 0.0   # no insertion during alignment
        elif phase == 3:
            action[1] = 0.0   # no pitch during recovery
            action[2] = 0.0   # no yaw during recovery
            action[0] = self.recovery_insert  # forced retreat

        # Gating
        if gating == 0:
            action = np.zeros(3, dtype=np.float32)  # hold
        elif gating == 3:
            action[0] = self.recovery_insert
            action[1:] = 0.0

        action = np.clip(action, -1.0, 1.0)
        self._prev_action = action.copy()
        return action


class ActionMapper:
    """
    Maps normalized action [-1,1]^3 to physical units.

    insertion: delta_m  = action[0] * insertion_scale
    pitch:     delta_rad = action[1] * bend_scale
    yaw:       delta_rad = action[2] * bend_scale
    """

    def __init__(
        self,
        insertion_scale: float = 0.0025,  # 2.5mm per unit
        bend_scale: float = 0.05,         # 0.05 rad per unit
    ):
        self.insertion_scale = insertion_scale
        self.bend_scale = bend_scale

    def to_physical(self, action: np.ndarray) -> dict:
        return {
            "delta_insert_m": float(action[0]) * self.insertion_scale,
            "delta_pitch_rad": float(action[1]) * self.bend_scale,
            "delta_yaw_rad": float(action[2]) * self.bend_scale,
        }
