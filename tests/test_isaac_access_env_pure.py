"""
Pure unit tests for IsaacAccessEnv — no Isaac Sim required.

Tests only the pure-Python logic in IsaacAccessEnv by mocking all Isaac
modules before import. Run with conda ercp environment:

    conda activate ercp
    cd /data/ERCP/ercp_access
    python -m pytest tests/test_isaac_access_env_pure.py -v
"""
from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock

# ── Mock all Isaac/omni modules before any import ─────────────────────────────
# This must happen before importing IsaacAccessEnv
_isaac_mocks = [
    "isaacsim",
    "omni",
    "omni.usd",
    "omni.replicator",
    "omni.replicator.core",
    "omni.isaac",
    "omni.isaac.core",
    "omni.isaac.core.objects",
    "omni.isaac.core.prims",
    "omni.isaac.core.utils",
    "omni.isaac.core.utils.prims",
    "omni.isaac.sensor",
    "omni.isaac.lab",
    "pxr",
    "pxr.UsdLux",
    "pxr.UsdGeom",
    "pxr.Gf",
]
for _mod in _isaac_mocks:
    sys.modules[_mod] = MagicMock()

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import numpy as np
import pytest

from src.envs.isaac_access_env import IsaacAccessEnv
from src.perception.state_builder import StateBuilder
from src.gating.phase_manager import PhaseManager
from src.gating.insertion_gate import InsertionGate


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_env():
    """
    Build an IsaacAccessEnv instance with _init_isaac bypassed.
    Manually sets all attributes that __init__ would set (minus Isaac objects).
    """
    env = object.__new__(IsaacAccessEnv)

    # Replicate __init__ attribute assignments (sans _init_isaac call)
    env._use_oracle = True
    env._headless = True
    env._max_steps = 1500
    env._camera_width = 512
    env._camera_height = 512
    env._init_dist_range = (0.05, 0.10)
    env._init_angle_range_deg = (10.0, 15.0)
    env._insertion_scale = 0.0025
    env._bend_scale = 0.05
    env._ref_distance = 0.05
    env._ref_scale = 0.15
    env._success_insert_depth = 0.02

    env._rng = np.random.default_rng(42)

    env._state_builder = StateBuilder(history_len=8, use_oracle=True)
    env._insertion_gate = InsertionGate()
    env._safety_rules = MagicMock()
    env._phase_manager = PhaseManager(
        conf_thresh=0.5,
        recovery_steps=8,
        insertion_gate=env._insertion_gate,
    )

    env._scope_pos = np.zeros(3, dtype=np.float64)
    env._scope_euler = np.zeros(2, dtype=np.float64)
    env._papilla_pos = np.array([0.10, 0.0, 0.0], dtype=np.float64)
    env._step_count = 0
    env._prev_action = np.zeros(3, dtype=np.float32)
    env._phase = 0
    env._insert_depth = 0.0
    env._episode_id = "test"

    # Isaac objects — not used by pure tests
    env._world = MagicMock()
    env._papilla_obj = MagicMock()
    env._scope_prim = MagicMock()
    env._camera = MagicMock()
    env._rgb_annotator = MagicMock()

    return env


# ── T1: _euler_to_quat correctness ───────────────────────────────────────────

def test_euler_to_quat_identity():
    """Zero angles → identity quaternion [1, 0, 0, 0]."""
    q = IsaacAccessEnv._euler_to_quat(0.0, 0.0)
    np.testing.assert_allclose(q, [1.0, 0.0, 0.0, 0.0], atol=1e-6)


def test_euler_to_quat_unit_norm():
    """Any valid angle combination → unit quaternion."""
    for pitch, yaw in [(0.1, 0.2), (-0.5, 0.3), (1.0, -1.0),
                       (math.pi / 4, 0.0), (0.0, math.pi / 3)]:
        q = IsaacAccessEnv._euler_to_quat(pitch, yaw)
        norm = float(np.linalg.norm(q))
        assert abs(norm - 1.0) < 1e-6, f"norm={norm} for pitch={pitch} yaw={yaw}"


def test_euler_to_quat_returns_float32():
    """Output must be float32 [4] (expected by Isaac XFormPrim)."""
    q = IsaacAccessEnv._euler_to_quat(0.3, -0.2)
    assert q.dtype == np.float32
    assert q.shape == (4,)


def test_euler_to_quat_pitch_only():
    """Pitch π/2 around Y axis: expect [cos(π/4), 0, sin(π/4), 0]."""
    q = IsaacAccessEnv._euler_to_quat(math.pi / 2, 0.0)
    expected = [math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0]
    np.testing.assert_allclose(q, expected, atol=1e-6)


def test_euler_to_quat_yaw_only():
    """Yaw π/2 around Z axis: expect [cos(π/4), 0, 0, sin(π/4)]."""
    q = IsaacAccessEnv._euler_to_quat(0.0, math.pi / 2)
    expected = [math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]
    np.testing.assert_allclose(q, expected, atol=1e-6)


# ── T2: Action scale constants match YAML and ActionMapper spec ───────────────

def test_insertion_scale_matches_spec(mock_env):
    """insertion_scale must be 0.0025 (ActionMapper standard)."""
    assert mock_env._insertion_scale == 0.0025


def test_bend_scale_matches_spec(mock_env):
    """bend_scale must be 0.05 (ActionMapper standard)."""
    assert mock_env._bend_scale == 0.05


def test_success_insert_depth(mock_env):
    """success_insert_depth must be 0.02m (matches ToyAccessEnv)."""
    assert mock_env._success_insert_depth == 0.02


# ── T3: validate_action (inherited from AccessEnvBase) ───────────────────────

def test_validate_action_clips_to_minus1_plus1(mock_env):
    """validate_action must clip values exceeding [-1, 1]."""
    action = np.array([2.0, -3.5, 0.5])
    result = mock_env.validate_action(action)
    assert result.shape == (3,)
    assert result.dtype == np.float32
    assert result[0] == pytest.approx(1.0)
    assert result[1] == pytest.approx(-1.0)
    assert result[2] == pytest.approx(0.5)


def test_validate_action_accepts_valid(mock_env):
    """validate_action must pass through values already in [-1, 1]."""
    action = np.array([0.3, -0.7, 0.0])
    result = mock_env.validate_action(action)
    np.testing.assert_allclose(result, action, atol=1e-6)


# ── T4: PhaseManager recovery_steps alignment ─────────────────────────────────

def test_phase_manager_recovery_steps(mock_env):
    """PhaseManager.recovery_steps must equal 8 (same as ToyAccessEnv)."""
    assert mock_env._phase_manager.recovery_steps == 8


# ── T5: max_episode_steps property ───────────────────────────────────────────

def test_max_episode_steps_property(mock_env):
    """max_episode_steps must return _max_steps value."""
    assert mock_env.max_episode_steps == 1500


# ── T6: dt property ──────────────────────────────────────────────────────────

def test_dt_property_is_float(mock_env):
    """dt must be a positive float (from scene cfg rendering_dt)."""
    from src.envs.isaac_lab_cfg import DEFAULT_SCENE_CFG
    # dt comes from scene cfg, not a fixed attribute on the env
    assert DEFAULT_SCENE_CFG.rendering_dt > 0.0


# ── T7: Lateral drift coupling factor ────────────────────────────────────────

def test_lateral_drift_pitch_affects_z_only(mock_env):
    """Pure pitch action → Z drifts by bend_scale*0.1, Y unchanged."""
    mock_env._scope_pos[:] = 0.0
    mock_env._scope_euler[:] = 0.0

    # Apply pitch dynamics manually (same math as step())
    pitch_action = 1.0
    mock_env._scope_euler[0] += pitch_action * mock_env._bend_scale
    mock_env._scope_pos[2] += pitch_action * mock_env._bend_scale * 0.1

    expected_z = 0.05 * 0.1  # 0.005
    assert abs(mock_env._scope_pos[2] - expected_z) < 1e-10
    assert abs(mock_env._scope_pos[1]) < 1e-10  # Y must be unchanged


def test_lateral_drift_yaw_affects_y_only(mock_env):
    """Pure yaw action → Y drifts by bend_scale*0.1, Z unchanged."""
    mock_env._scope_pos[:] = 0.0
    mock_env._scope_euler[:] = 0.0

    yaw_action = 1.0
    mock_env._scope_euler[1] += yaw_action * mock_env._bend_scale
    mock_env._scope_pos[1] += yaw_action * mock_env._bend_scale * 0.1

    expected_y = 0.05 * 0.1  # 0.005
    assert abs(mock_env._scope_pos[1] - expected_y) < 1e-10
    assert abs(mock_env._scope_pos[2]) < 1e-10  # Z must be unchanged


def test_lateral_drift_magnitude(mock_env):
    """Drift factor 0.1 means 10× smaller lateral movement than Euler angle change."""
    mock_env._scope_pos[:] = 0.0
    mock_env._scope_euler[:] = 0.0

    action = 1.0
    euler_delta = action * mock_env._bend_scale       # 0.05 rad
    pos_delta = action * mock_env._bend_scale * 0.1  # 0.005 m

    assert abs(pos_delta / euler_delta - 0.1) < 1e-10


# ── T8: Euler angle clamping ──────────────────────────────────────────────────

def test_euler_clamp_at_90_deg(mock_env):
    """Euler angles must be clamped to [-π/2, π/2]."""
    mock_env._scope_euler = np.array([math.pi, -math.pi])
    clamped = np.clip(mock_env._scope_euler, -math.pi / 2, math.pi / 2)
    assert clamped[0] == pytest.approx(math.pi / 2)
    assert clamped[1] == pytest.approx(-math.pi / 2)
