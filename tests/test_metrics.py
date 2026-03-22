"""Tests for EpisodeMetrics and RunMetrics aggregation."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.base_env import EpisodeMetrics, RunMetrics
from src.logging.logger import StepLogger, EpisodeLogger
import tempfile


def make_step_info(success=False, insert_executed=False):
    return {
        "phase": 1, "gating": 1,
        "alignment_error": 0.05, "conf": 0.8, "readiness": 0.7,
        "target_visible": True, "insert_executed": insert_executed,
        "event_flags": {
            "target_loss": False, "off_axis": False,
            "no_progress": False, "recovery_triggered": False,
            "unsafe_insert_attempt": False,
        },
        "success": success,
    }


def test_step_logger_finalize():
    import numpy as np
    sl = StepLogger()
    sl.start_episode("ep_001", dt=0.02)
    for i in range(10):
        sl.log_step(i, np.zeros(3), np.zeros(3), 1.0, make_step_info())
    metrics = sl.finalize(success=True)
    assert metrics.success == 1
    assert metrics.steps == 10
    assert abs(metrics.time_to_access - 0.2) < 1e-5


def test_episode_logger_run_metrics():
    with tempfile.TemporaryDirectory() as tmpdir:
        el = EpisodeLogger(tmpdir)
        for i in range(10):
            el.add(EpisodeMetrics(
                episode_id=str(i), success=1 if i < 8 else 0,
                steps=50, time_to_access=1.0,
            ))
        rm = el.get_run_metrics()
        assert abs(rm.access_success_rate - 0.8) < 1e-5
        assert rm.n_episodes == 10
        assert abs(rm.abort_rate - 0.2) < 1e-5


def test_episode_logger_save():
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        el = EpisodeLogger(tmpdir)
        el.add(EpisodeMetrics(episode_id="e1", success=1, steps=30))
        el.save(prefix="test")
        assert os.path.exists(os.path.join(tmpdir, "test_episodes.csv"))
        assert os.path.exists(os.path.join(tmpdir, "test_run_metrics.json"))
