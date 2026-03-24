"""
PIDPolicy (B1 Seg+PID baseline).

Architecture:
  Phase 0 (alignment):  PID_yaw(e_x) + PID_pitch(e_y), insert=0
  Phase 1 (approach):   PID_yaw(e_x) + PID_pitch(e_y) + PID_insert(target_scale - scale)
  Phase 2 (insertion):  small PID correction (×0.3) + fixed insert_strength
  Phase 3 (recovery):   zeros (ActionPostprocessor overrides to [-0.8, 0, 0])

Used with ActionPostprocessor — do NOT replicate phase mask logic here.
PID states reset on phase transitions to prevent integral windup propagation.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from src.controllers.pid_controller import PIDController
from src.policies.scripted_policy import PolicyBase


class PIDPolicy(PolicyBase):
    """
    B1 Seg+PID baseline policy.

    Key differences from B0 ScriptedPolicy:
    - Full PID (Kp + Ki + Kd) vs pure P control
    - Adaptive insert axis (PID on scale error) vs fixed constant
    - PID states reset on phase transitions
    """

    def __init__(
        self,
        # Alignment axes PID (shared gains for pitch and yaw)
        kp_align: float = 3.5,
        ki_align: float = 0.05,
        kd_align: float = 0.5,
        # Approach axis PID (scale -> insert)
        kp_insert: float = 2.0,
        ki_insert: float = 0.02,
        kd_insert: float = 0.2,
        # Fixed parameters
        target_scale: float = 0.35,
        insert_strength: float = 0.8,
        insertion_phase_align_scale: float = 0.3,
        dt: float = 0.02,
        integral_limit_align: float = 2.0,
        integral_limit_insert: float = 1.0,
    ):
        self.target_scale = target_scale
        self.insert_strength = insert_strength
        self.insertion_phase_align_scale = insertion_phase_align_scale
        self.dt = dt

        self._pid_yaw = PIDController(
            kp=kp_align, ki=ki_align, kd=kd_align,
            output_limits=(-1.0, 1.0),
            integral_limit=integral_limit_align,
        )
        self._pid_pitch = PIDController(
            kp=kp_align, ki=ki_align, kd=kd_align,
            output_limits=(-1.0, 1.0),
            integral_limit=integral_limit_align,
        )
        self._pid_insert = PIDController(
            kp=kp_insert, ki=ki_insert, kd=kd_insert,
            output_limits=(-1.0, 1.0),
            integral_limit=integral_limit_insert,
        )

        self._prev_phase: int = -1

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        """
        Compute raw action from observation.

        Args:
            obs: dict with keys 'state' float32[6] and 'phase' int64

        Returns:
            float32 [3]: [delta_insert, delta_pitch, delta_yaw] in [-1, 1]
        """
        state_arr = obs["state"]
        phase = int(obs["phase"])

        e_x   = float(state_arr[0])
        e_y   = float(state_arr[1])
        scale = float(state_arr[2])

        # Detect phase change -> reset PIDs
        if phase != self._prev_phase:
            self._on_phase_change(phase)
        self._prev_phase = phase

        # Phase 3 (recovery): return zeros, Postprocessor overrides
        if phase == 3:
            return np.zeros(3, dtype=np.float32)

        # Alignment axes: e_x -> yaw, e_y -> pitch
        yaw   = self._pid_yaw.compute(error=e_x,  dt=self.dt)
        pitch = self._pid_pitch.compute(error=e_y, dt=self.dt)

        # Insert axis
        if phase == 0:
            # Phase 0: Postprocessor will zero out insert anyway.
            # Reset insert PID each step to prevent integral buildup.
            self._pid_insert.reset()
            insert = 0.0
        elif phase == 1:
            scale_error = self.target_scale - scale  # positive = need to advance
            insert = self._pid_insert.compute(error=scale_error, dt=self.dt)
        elif phase == 2:
            # Fixed insertion strength; small alignment correction
            insert = self.insert_strength
            pitch *= self.insertion_phase_align_scale
            yaw   *= self.insertion_phase_align_scale
        else:
            insert = 0.0

        return np.array([insert, pitch, yaw], dtype=np.float32)

    def reset(self) -> None:
        """Reset all PID states (call at episode start)."""
        self._pid_yaw.reset()
        self._pid_pitch.reset()
        self._pid_insert.reset()
        self._prev_phase = -1

    def _on_phase_change(self, new_phase: int) -> None:
        """
        Selectively reset PID state on phase transitions.

        Cross-episode detection: new_phase==0 and prev!=0 means a new episode
        started (env.reset() happened). Reset all PIDs.
        """
        prev = self._prev_phase

        # Cross-episode boundary OR re-entering alignment: full reset
        if new_phase == 0 and prev != 0:
            self._pid_yaw.reset()
            self._pid_pitch.reset()
            self._pid_insert.reset()
            return

        # Normal phase transitions
        if new_phase in (2, 3):
            # Entering insertion or recovery: full reset
            self._pid_yaw.reset()
            self._pid_pitch.reset()
            self._pid_insert.reset()
        elif new_phase == 1:
            # Entering approach: only reset insert PID
            # Keep alignment PID continuity (error is already small)
            self._pid_insert.reset()
