"""Tests for AccessEnvBase contract and ToyAccessEnv."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.toy_access_env import ToyAccessEnv
from src.envs.base_env import AccessEnvBase


def make_env():
    return ToyAccessEnv(max_steps=50, seed=0)


def test_env_is_subclass():
    env = make_env()
    assert isinstance(env, AccessEnvBase)


def test_reset_returns_correct_keys():
    env = make_env()
    obs, info = env.reset()
    assert set(obs.keys()) == {"rgb", "state", "phase", "prev_action", "valid"}
    assert obs["state"].shape == (6,)
    assert obs["prev_action"].shape == (3,)
    assert obs["rgb"].shape == (3, 64, 64)
    assert obs["state"].dtype == np.float32
    assert "episode_id" in info


def test_step_returns_correct_shapes():
    env = make_env()
    obs, _ = env.reset()
    action = np.zeros(3, dtype=np.float32)
    obs2, reward, terminated, truncated, info = env.step(action)
    assert obs2["state"].shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "success" in info
    assert "event_flags" in info


def test_validate_action_clips():
    env = make_env()
    action = np.array([2.0, -3.0, 1.5], dtype=np.float32)
    clipped = env.validate_action(action)
    assert np.all(clipped >= -1.0) and np.all(clipped <= 1.0)


def test_truncation_at_max_steps():
    env = ToyAccessEnv(max_steps=5, seed=1)
    env.reset()
    for _ in range(5):
        _, _, terminated, truncated, _ = env.step(np.zeros(3, dtype=np.float32))
    assert truncated or terminated


def test_difficulty_levels():
    for diff in ["easy", "medium", "hard"]:
        env = make_env()
        obs, info = env.reset(difficulty=diff)
        assert info["difficulty"] == diff
        assert obs["state"].shape == (6,)


def test_state_values_in_range():
    env = make_env()
    for _ in range(10):
        obs, _ = env.reset()
        s = obs["state"]
        assert -1.5 <= s[0] <= 1.5, f"e_x out of range: {s[0]}"
        assert -1.5 <= s[1] <= 1.5, f"e_y out of range: {s[1]}"
        assert 0.0 <= s[2] <= 1.0, f"scale out of range: {s[2]}"
        assert 0.0 <= s[3] <= 1.0, f"conf out of range: {s[3]}"
        assert 0.0 <= s[4] <= 1.0, f"stability out of range: {s[4]}"
        assert 0.0 <= s[5] <= 1.0, f"readiness out of range: {s[5]}"
