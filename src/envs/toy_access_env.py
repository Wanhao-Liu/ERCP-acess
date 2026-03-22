"""
ToyAccessEnv — 2D simplified biliary access environment.

No Isaac Sim dependency. Uses oracle state directly.
Purpose: debug FSM/gating/policy logic at high speed.

Dynamics:
  - Scope tip starts at random offset from papilla target
  - pitch/yaw actions move the scope laterally
  - insert action advances scope along approach axis
  - Success: aligned + close enough + inserted past threshold
"""
from __future__ import annotations

import uuid
from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.envs.base_env import AccessEnvBase, EpisodeMetrics
from src.perception.access_state import AccessState
from src.perception.state_builder import StateBuilder
from src.gating.phase_manager import PhaseManager
from src.gating.insertion_gate import InsertionGate, SafetyRules


class ToyAccessEnv(AccessEnvBase):
    """
    2D oracle-state environment for rapid iteration.

    State space:
      scope_pos: [x, y, z] in local frame (x=insertion axis, y/z=lateral)
      papilla_pos: fixed at [target_depth, 0, 0]

    Action:
      [delta_insert, delta_pitch, delta_yaw] in [-1, 1]
      delta_insert → moves scope along x
      delta_pitch  → moves scope along z
      delta_yaw    → moves scope along y
    """

    RGB_SHAPE = (3, 64, 64)  # small placeholder for toy env

    def __init__(
        self,
        max_steps: int = 200,
        target_noise_std: float = 0.02,
        success_insert_depth: float = 0.02,
        init_offset_range: Tuple[float, float] = (0.02, 0.15),
        init_angle_range: Tuple[float, float] = (0.0, 20.0),
        insertion_scale: float = 0.005,   # m per unit action
        bend_scale: float = 0.01,         # m per unit action (lateral)
        dt: float = 0.02,
        seed: Optional[int] = None,
    ):
        self._max_steps = max_steps
        self.target_noise_std = target_noise_std
        self.success_insert_depth = success_insert_depth
        self.init_offset_range = init_offset_range
        self.init_angle_range = init_angle_range
        self.insertion_scale = insertion_scale
        self.bend_scale = bend_scale
        self._dt = dt

        self._rng = np.random.default_rng(seed)
        self._state_builder = StateBuilder(history_len=8, use_oracle=True)
        self._insertion_gate = InsertionGate()
        self._safety_rules = SafetyRules()
        self._phase_manager = PhaseManager(
            conf_thresh=0.5,
            recovery_steps=8,
            insertion_gate=self._insertion_gate,
        )

        # Internal state
        self._scope_pos = np.zeros(3, dtype=np.float32)   # [x, y, z]
        self._papilla_pos = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._phase = 0
        self._insert_depth = 0.0
        self._inserted = False

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    def reset(
        self,
        seed: Optional[int] = None,
        case_id: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        difficulty = difficulty or "medium"
        offset_range = self._get_offset_range(difficulty)

        # Papilla at fixed depth, scope starts offset
        target_depth = 0.08  # 8cm approach distance
        self._papilla_pos = np.array([target_depth, 0.0, 0.0], dtype=np.float32)

        # Random lateral offset
        offset_mag = self._rng.uniform(*offset_range)
        angle = self._rng.uniform(0, 2 * np.pi)
        self._scope_pos = np.array([
            0.0,
            offset_mag * np.cos(angle),
            offset_mag * np.sin(angle),
        ], dtype=np.float32)

        self._step_count = 0
        self._prev_action = np.zeros(3, dtype=np.float32)
        self._phase = 0
        self._insert_depth = 0.0
        self._inserted = False
        self._state_builder.reset()
        self._phase_manager.reset()
        self._safety_rules.reset()

        obs = self._get_obs()
        episode_id = case_id or str(uuid.uuid4())[:8]
        info = {"episode_id": episode_id, "case_id": episode_id,
                "difficulty": difficulty, "init_pose": self._scope_pos.copy()}
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        action = self.validate_action(action)

        # Apply dynamics
        # action[0]=insert → scope moves toward papilla (X axis)
        # action[1]=pitch  → scope moves in Z direction (sign: positive pitch = +Z)
        # action[2]=yaw    → scope moves in Y direction (sign: positive yaw = +Y)
        # e_x = (papilla_Y - scope_Y) / ref → positive yaw should increase scope_Y
        # e_y = (papilla_Z - scope_Z) / ref → positive pitch should increase scope_Z
        self._scope_pos[0] += action[0] * self.insertion_scale
        self._scope_pos[2] += action[1] * self.bend_scale   # pitch → Z
        self._scope_pos[1] += action[2] * self.bend_scale   # yaw → Y

        self._step_count += 1
        self._prev_action = action.copy()

        # Compute state
        access_state = self._compute_access_state()

        # Update phase via FSM
        event_flags_pre = {
            "target_loss": False, "off_axis": False,
            "no_progress": False, "recovery_triggered": False,
            "unsafe_insert_attempt": False,
        }
        self._phase = self._phase_manager.update(access_state, event_flags_pre)

        # Check insertion (only in phase 2)
        insert_executed = False
        if self._phase == 2 and action[0] > 0.1:
            self._insert_depth += action[0] * self.insertion_scale
            insert_executed = True

        # Success condition
        success = (
            access_state.is_aligned(e_thresh=0.08)
            and access_state.scale >= 0.3
            and self._insert_depth >= self.success_insert_depth
        )

        # Reward
        reward = self._compute_reward(access_state, insert_executed, success)

        # Termination
        terminated = success
        truncated = self._step_count >= self._max_steps

        # Build info
        event_flags = self._safety_rules.check(
            access_state, self._phase, insert_executed
        )
        info = {
            "phase": self._phase,
            "gating": self._insertion_gate.get_gating(access_state, self._phase),
            "alignment_error": access_state.alignment_error,
            "conf": access_state.conf,
            "readiness": access_state.readiness,
            "target_visible": True,
            "insert_executed": insert_executed,
            "event_flags": event_flags,
            "success": success,
        }

        obs = self._get_obs(access_state)
        return obs, reward, terminated, truncated, info

    def close(self):
        pass

    # ── Internal helpers ───────────────────────────────────────────────

    def _get_offset_range(self, difficulty: str) -> Tuple[float, float]:
        return {
            "easy": (0.01, 0.05),
            "medium": (0.05, 0.12),
            "hard": (0.12, 0.25),
        }.get(difficulty, (0.05, 0.12))

    def _compute_access_state(self) -> AccessState:
        delta = self._papilla_pos - self._scope_pos
        dist = float(np.linalg.norm(delta)) + 1e-6

        # e_x: yaw axis (Y lateral offset), e_y: pitch axis (Z lateral offset)
        # Use absolute lateral offset normalized by a reference scale (not by dist)
        # This gives a stable, controllable error signal
        ref_scale = 0.15  # 15cm reference lateral range
        e_x = float(np.clip(delta[1] / ref_scale, -1.0, 1.0))
        e_y = float(np.clip(delta[2] / ref_scale, -1.0, 1.0))
        scale = float(np.clip(0.04 / dist, 0.0, 1.0))
        conf = 1.0  # oracle

        gt_info = {"e_x": e_x, "e_y": e_y, "scale": scale, "conf": conf}
        return self._state_builder.update(rgb=None, gt_info=gt_info)

    def _get_obs(self, access_state: Optional[AccessState] = None) -> Dict[str, Any]:
        if access_state is None:
            access_state = self._compute_access_state()
        return self._make_obs_dict(
            state=access_state.to_array(),
            phase=self._phase,
            prev_action=self._prev_action,
            valid=True,
            rgb=np.zeros(self.RGB_SHAPE, dtype=np.float32),
        )

    def _compute_reward(
        self, state: AccessState, insert_executed: bool, success: bool
    ) -> float:
        r = 0.0
        r -= state.alignment_error * 1.0          # alignment penalty
        r += (state.scale - 0.1) * 2.0            # approach reward
        if insert_executed and not state.is_aligned():
            r -= 5.0                               # unsafe insert penalty
        if success:
            r += 50.0
        return float(r)
