"""
AccessEnvBase — frozen interface for all biliary access environments.

obs dict, action, step info, and metrics are all specified here.
Subclasses: ToyAccessEnv, IsaacAccessEnv
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np


# ── Observation ────────────────────────────────────────────────────────

@dataclass
class AccessObs:
    rgb: np.ndarray        # float32, [3, H, W], [0, 1]
    state: np.ndarray      # float32, [6]: [e_x, e_y, scale, conf, stability, readiness]
    phase: int             # 0=alignment, 1=approach, 2=insertion, 3=recovery
    prev_action: np.ndarray  # float32, [3]
    valid: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rgb": self.rgb,
            "state": self.state,
            "phase": np.int64(self.phase),
            "prev_action": self.prev_action,
            "valid": np.bool_(self.valid),
        }


# ── Step info ──────────────────────────────────────────────────────────

@dataclass
class EventFlags:
    target_loss: bool = False
    off_axis: bool = False
    no_progress: bool = False
    recovery_triggered: bool = False
    unsafe_insert_attempt: bool = False

    def to_dict(self) -> Dict[str, bool]:
        return {
            "target_loss": self.target_loss,
            "off_axis": self.off_axis,
            "no_progress": self.no_progress,
            "recovery_triggered": self.recovery_triggered,
            "unsafe_insert_attempt": self.unsafe_insert_attempt,
        }


@dataclass
class AccessInfo:
    phase: int = 0           # 0=alignment, 1=approach, 2=insertion, 3=recovery
    gating: int = 0          # 0=hold, 1=approach, 2=insert, 3=recover
    alignment_error: float = 0.0
    conf: float = 0.0
    readiness: float = 0.0
    target_visible: bool = True
    insert_executed: bool = False
    event_flags: EventFlags = field(default_factory=EventFlags)
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "phase": self.phase,
            "gating": self.gating,
            "alignment_error": self.alignment_error,
            "conf": self.conf,
            "readiness": self.readiness,
            "target_visible": self.target_visible,
            "insert_executed": self.insert_executed,
            "event_flags": self.event_flags.to_dict(),
            "success": self.success,
        }


# ── Reset info ─────────────────────────────────────────────────────────

@dataclass
class ResetInfo:
    episode_id: str = ""
    case_id: str = ""
    difficulty: str = "medium"
    init_pose: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "case_id": self.case_id,
            "difficulty": self.difficulty,
            "init_pose": self.init_pose,
        }


# ── Episode metrics ────────────────────────────────────────────────────

@dataclass
class EpisodeMetrics:
    episode_id: str = ""
    case_id: str = ""
    success: int = 0
    total_reward: float = 0.0
    steps: int = 0
    time_to_access: float = 0.0
    num_insert_attempts: int = 0
    target_loss_count: int = 0
    off_axis_count: int = 0
    recovery_count: int = 0
    no_progress_count: int = 0
    unsafe_insert_attempts: int = 0
    mean_alignment_error: float = 0.0
    max_alignment_error: float = 0.0
    mean_conf: float = 0.0
    mean_readiness: float = 0.0


@dataclass
class RunMetrics:
    n_episodes: int = 0
    access_success_rate: float = 0.0
    mean_time_to_access: float = 0.0
    mean_steps_to_access: float = 0.0
    mean_insert_attempts: float = 0.0
    target_loss_rate: float = 0.0
    off_axis_rate: float = 0.0
    recovery_rate: float = 0.0
    abort_rate: float = 0.0
    mean_alignment_error: float = 0.0


# ── Abstract base class ────────────────────────────────────────────────

class AccessEnvBase(abc.ABC):
    """
    Abstract base class for biliary access environments.

    All subclasses must implement reset(), step(), close().
    The obs dict, action shape, and info dict are frozen here.
    """

    RGB_SHAPE: Tuple[int, int, int] = (3, 512, 512)
    STATE_DIM: int = 6
    ACTION_DIM: int = 3

    @abc.abstractmethod
    def reset(
        self,
        seed: Optional[int] = None,
        case_id: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Returns (obs_dict, reset_info_dict)."""
        ...

    @abc.abstractmethod
    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Returns (obs_dict, reward, terminated, truncated, info_dict)."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def dt(self) -> float:
        """Control timestep in seconds."""
        ...

    @property
    @abc.abstractmethod
    def max_episode_steps(self) -> int:
        ...

    def validate_action(self, action: np.ndarray) -> np.ndarray:
        assert action.shape == (self.ACTION_DIM,), \
            f"Expected action shape ({self.ACTION_DIM},), got {action.shape}"
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def _make_obs_dict(
        self,
        state: np.ndarray,
        phase: int,
        prev_action: np.ndarray,
        valid: bool,
        rgb: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        return {
            "rgb": rgb.astype(np.float32) if rgb is not None
                   else np.zeros(self.RGB_SHAPE, dtype=np.float32),
            "state": state.astype(np.float32),
            "phase": np.int64(phase),
            "prev_action": prev_action.astype(np.float32),
            "valid": np.bool_(valid),
        }
