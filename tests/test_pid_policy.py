"""Tests for PIDController and PIDPolicy."""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.controllers.pid_controller import PIDController
from src.policies.pid_policy import PIDPolicy


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_obs(phase: int, e_x: float = 0.0, e_y: float = 0.0,
             scale: float = 0.2) -> dict:
    state = np.array([e_x, e_y, scale, 1.0, 1.0, 0.5], dtype=np.float32)
    return {
        "state": state,
        "phase": np.int64(phase),
        "prev_action": np.zeros(3, dtype=np.float32),
        "valid": np.bool_(True),
        "rgb": np.zeros((3, 64, 64), dtype=np.float32),
    }


# ── PIDController tests ───────────────────────────────────────────────────────

class TestPIDController:

    def test_pure_p_control(self):
        pid = PIDController(kp=2.0, ki=0.0, kd=0.0, output_limits=(-10.0, 10.0))
        out = pid.compute(error=0.5, dt=1.0)
        assert abs(out - 1.0) < 1e-5

    def test_output_clip(self):
        pid = PIDController(kp=10.0, ki=0.0, kd=0.0, output_limits=(-1.0, 1.0))
        out = pid.compute(error=1.0, dt=1.0)
        assert out == 1.0

    def test_integral_accumulates(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0,
                            integral_limit=100.0, output_limits=(-100.0, 100.0))
        for _ in range(5):
            pid.compute(error=0.1, dt=1.0)
        assert abs(pid.integral - 0.5) < 1e-5

    def test_integral_anti_windup(self):
        pid = PIDController(kp=0.0, ki=1.0, kd=0.0,
                            integral_limit=2.0, output_limits=(-100.0, 100.0))
        for _ in range(100):
            pid.compute(error=1.0, dt=1.0)
        assert abs(pid.integral) <= 2.0 + 1e-5

    def test_reset_clears_state(self):
        pid = PIDController(kp=1.0, ki=1.0, kd=1.0)
        pid.compute(error=1.0, dt=1.0)
        pid.reset()
        assert pid.integral == 0.0

    def test_no_cold_start_derivative_spike(self):
        pid = PIDController(kp=0.0, ki=0.0, kd=100.0,
                            output_limits=(-100.0, 100.0))
        out = pid.compute(error=1.0, dt=1.0)
        assert abs(out) < 1e-5, "First step should have zero derivative"

    def test_derivative_damping(self):
        pid = PIDController(kp=1.0, ki=0.0, kd=1.0,
                            output_limits=(-10.0, 10.0))
        pid.compute(error=1.0, dt=1.0)   # first step, derivative=0
        out = pid.compute(error=0.5, dt=1.0)  # error decreasing -> negative derivative
        # output = 1.0*0.5 + 1.0*(0.5-1.0)/1.0 = 0.5 - 0.5 = 0.0
        assert abs(out - 0.0) < 1e-5


# ── PIDPolicy tests ───────────────────────────────────────────────────────────

class TestPIDPolicy:

    def test_act_returns_correct_shape_and_dtype(self):
        policy = PIDPolicy()
        action = policy.act(make_obs(phase=0))
        assert action.shape == (3,)
        assert action.dtype == np.float32

    def test_act_output_in_range(self):
        policy = PIDPolicy()
        for phase in range(4):
            action = policy.act(make_obs(phase=phase, e_x=0.5, e_y=-0.3, scale=0.1))
            assert np.all(action >= -1.0) and np.all(action <= 1.0), \
                f"Phase {phase}: action out of [-1,1]: {action}"

    def test_phase0_positive_error_gives_positive_action(self):
        """e_x > 0 -> yaw > 0; e_y > 0 -> pitch > 0"""
        policy = PIDPolicy()
        action = policy.act(make_obs(phase=0, e_x=0.3, e_y=0.2))
        assert action[2] > 0.0, f"yaw should be positive when e_x>0, got {action[2]}"
        assert action[1] > 0.0, f"pitch should be positive when e_y>0, got {action[1]}"

    def test_phase0_insert_is_zero(self):
        """Phase 0: insert must be zero (Postprocessor will also enforce this)."""
        policy = PIDPolicy()
        action = policy.act(make_obs(phase=0, e_x=0.1, scale=0.1))
        assert action[0] == 0.0, f"Phase 0 insert should be 0, got {action[0]}"

    def test_phase3_returns_zeros(self):
        """Phase 3: policy returns zeros (Postprocessor overrides to retreat)."""
        policy = PIDPolicy()
        action = policy.act(make_obs(phase=3))
        assert np.allclose(action, 0.0), f"Phase 3 should return zeros, got {action}"

    def test_phase2_insert_equals_insert_strength(self):
        """Phase 2: insert == insert_strength (fixed, not PID-controlled)."""
        policy = PIDPolicy(insert_strength=0.8)
        action = policy.act(make_obs(phase=2, e_x=0.05, e_y=0.05, scale=0.35))
        assert abs(action[0] - 0.8) < 1e-5, f"Phase 2 insert should be 0.8, got {action[0]}"

    def test_phase2_align_correction_smaller_than_phase0(self):
        """Phase 2 alignment correction is scaled down vs Phase 0."""
        p0 = PIDPolicy(kp_align=3.5, insertion_phase_align_scale=0.3)
        a0 = p0.act(make_obs(phase=0, e_x=0.3, e_y=0.0))

        p2 = PIDPolicy(kp_align=3.5, insertion_phase_align_scale=0.3)
        a2 = p2.act(make_obs(phase=2, e_x=0.3, e_y=0.0))
        assert abs(a2[2]) < abs(a0[2]), \
            f"Phase 2 yaw {a2[2]:.4f} should be < Phase 0 yaw {a0[2]:.4f}"

    def test_reset_clears_all_pid_states(self):
        """reset() zeros all three PID integrals and prev_phase."""
        policy = PIDPolicy()
        for _ in range(10):
            policy.act(make_obs(phase=0, e_x=0.5, e_y=0.5))
        policy.reset()
        assert policy._pid_yaw.integral == 0.0
        assert policy._pid_pitch.integral == 0.0
        assert policy._pid_insert.integral == 0.0
        assert policy._prev_phase == -1

    def test_phase_transition_1_resets_insert_pid(self):
        """Phase 0->1: insert PID integral is not carried over from phase 0.

        After 5 phase-0 steps the insert PID is reset each step (integral=0).
        On the first phase-1 step, _on_phase_change resets then compute() runs
        once, so the integral equals exactly one step's accumulation.
        Verify it's << what it would be without a reset (5 prior steps worth).
        """
        policy = PIDPolicy(ki_insert=1.0)  # large ki to make the difference obvious
        scale_error = 0.35 - 0.1  # = 0.25
        dt = policy.dt  # 0.02

        # Simulate 5 phase-0 steps (insert PID reset each step, integral stays 0)
        for _ in range(5):
            policy.act(make_obs(phase=0, e_x=0.5, e_y=0.5))

        # Transition to phase 1: _on_phase_change resets, then compute runs once
        policy.act(make_obs(phase=1, e_x=0.1, scale=0.1))

        # Integral should be exactly one step's accumulation (no carryover)
        expected_one_step = scale_error * dt  # 0.25 * 0.02 = 0.005
        assert abs(policy._pid_insert.integral - expected_one_step) < 1e-5, \
            f"Insert PID integral should be one step only ({expected_one_step:.5f}), " \
            f"got {policy._pid_insert.integral:.5f}"

    def test_approach_larger_error_gives_larger_insert(self):
        """Phase 1: farther from target -> larger insert command."""
        p_far = PIDPolicy()
        a_far = p_far.act(make_obs(phase=1, e_x=0.0, e_y=0.0, scale=0.05))

        p_near = PIDPolicy()
        a_near = p_near.act(make_obs(phase=1, e_x=0.0, e_y=0.0, scale=0.30))

        assert a_far[0] > a_near[0], \
            f"Farther scope (scale=0.05) should have larger insert than near (scale=0.30)"

    def test_full_episode_no_crash(self):
        """Integration: PIDPolicy + Postprocessor runs one episode without crashing."""
        from src.envs.toy_access_env import ToyAccessEnv
        from src.controllers.postprocessor import ActionPostprocessor

        env = ToyAccessEnv(max_steps=200, seed=42)
        policy = PIDPolicy()
        pp = ActionPostprocessor()

        obs, info = env.reset(difficulty="easy")
        policy.reset()
        pp.reset()

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < 200:
            raw = policy.act(obs)
            # Use postprocessor with actual signature: process(raw_action, phase, gating)
            phase = int(obs["phase"])
            gating = int(info.get("gating", phase))
            exec_action = pp.process(raw, phase, gating)
            obs, reward, terminated, truncated, info = env.step(exec_action)
            steps += 1

        assert steps >= 1
        assert steps <= 200
        env.close()
