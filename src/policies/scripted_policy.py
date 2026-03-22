"""
PolicyBase — abstract base for all policies.
ScriptedPolicy (B0) — heuristic FSM: align → approach → insert → recover.
"""
from __future__ import annotations

import abc
from typing import Any, Dict

import numpy as np

from src.perception.access_state import AccessState


class PolicyBase(abc.ABC):
    @abc.abstractmethod
    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """Returns action float32 [3] in [-1, 1]."""
        ...

    def reset(self):
        pass


class ScriptedPolicy(PolicyBase):
    """
    B0: rule-based FSM policy.

    phase 0/1 (alignment/approach):
      pitch = -kp * e_x, yaw = -kp * e_y
      insert = approach_insert if scale < target_scale else 0
    phase 2 (insertion):
      insert = insert_strength, pitch/yaw = small correction
    phase 3 (recovery):
      insert = -recover_speed, pitch/yaw = 0
    """

    def __init__(
        self,
        kp_align: float = 3.0,
        approach_insert: float = 0.3,
        target_scale: float = 0.35,
        insert_strength: float = 0.8,
        recover_speed: float = 0.6,
    ):
        self.kp_align = kp_align
        self.approach_insert = approach_insert
        self.target_scale = target_scale
        self.insert_strength = insert_strength
        self.recover_speed = recover_speed

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        state_arr = obs["state"]  # [6]: [e_x, e_y, scale, conf, stability, readiness]
        phase = int(obs["phase"])

        e_x, e_y, scale = float(state_arr[0]), float(state_arr[1]), float(state_arr[2])

        if phase == 3:  # recovery
            return np.array([-self.recover_speed, 0.0, 0.0], dtype=np.float32)

        # Alignment correction:
        #   e_x (Y-axis error) → corrected by yaw (action[2]): yaw = +kp * e_x
        #   e_y (Z-axis error) → corrected by pitch (action[1]): pitch = +kp * e_y
        pitch = float(np.clip(self.kp_align * e_y, -1.0, 1.0))
        yaw = float(np.clip(self.kp_align * e_x, -1.0, 1.0))

        if phase == 2:  # insertion
            insert = self.insert_strength
            pitch *= 0.3  # small correction only
            yaw *= 0.3
        elif scale < self.target_scale:  # approach
            insert = self.approach_insert
        else:
            insert = 0.0

        return np.array([insert, pitch, yaw], dtype=np.float32)
