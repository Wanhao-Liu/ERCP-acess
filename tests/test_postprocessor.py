"""Tests for ActionPostprocessor gating logic."""
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controllers.postprocessor import ActionPostprocessor


def make_pp():
    return ActionPostprocessor()


def test_clip():
    pp = make_pp()
    action = np.array([2.0, -3.0, 1.5], dtype=np.float32)
    out = pp.process(action, phase=1, gating=1)
    assert np.all(out >= -1.0) and np.all(out <= 1.0)


def test_phase0_blocks_insert():
    pp = make_pp()
    action = np.array([1.0, 0.5, 0.5], dtype=np.float32)
    out = pp.process(action, phase=0, gating=0)
    assert out[0] == 0.0, "Insert should be blocked in phase 0"


def test_gating_hold_zeros_action():
    pp = make_pp()
    action = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    out = pp.process(action, phase=1, gating=0)
    assert np.allclose(out, 0.0), "Gating=hold should zero action"


def test_recovery_phase_forces_retreat():
    pp = make_pp()
    action = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    out = pp.process(action, phase=3, gating=3)
    assert out[0] < 0.0, "Recovery should force negative insert"
    assert out[1] == 0.0
    assert out[2] == 0.0


def test_rate_limit():
    pp = make_pp()
    # First step: action = [0.5, 0, 0]
    pp.process(np.array([0.5, 0.0, 0.0], dtype=np.float32), phase=1, gating=1)
    # Second step: try to jump to [1.0, 0, 0] — should be rate-limited
    out = pp.process(np.array([1.0, 0.0, 0.0], dtype=np.float32), phase=1, gating=1)
    assert out[0] <= 0.5 + pp.max_delta_per_step + 1e-5
