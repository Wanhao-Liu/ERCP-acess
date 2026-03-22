"""Tests for logger (no crash, correct types)."""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logging.logger import StepLogger
from src.envs.base_env import EpisodeMetrics


def test_step_logger_no_crash():
    sl = StepLogger()
    sl.start_episode("ep_test", dt=0.02)
    info = {
        "phase": 0, "gating": 0, "alignment_error": 0.1,
        "conf": 0.9, "readiness": 0.5, "target_visible": True,
        "insert_executed": False,
        "event_flags": {
            "target_loss": False, "off_axis": False,
            "no_progress": False, "recovery_triggered": False,
            "unsafe_insert_attempt": False,
        },
        "success": False,
    }
    for i in range(5):
        sl.log_step(i, np.zeros(3), np.zeros(3), 0.0, info)
    metrics = sl.finalize(success=False)
    assert isinstance(metrics, EpisodeMetrics)
    assert metrics.success == 0
    assert metrics.steps == 5


def test_step_logger_empty_finalize():
    sl = StepLogger()
    sl.start_episode("empty")
    metrics = sl.finalize(success=False)
    assert metrics.steps == 0
