#!/usr/bin/env python3
"""
Isaac Sim integration tests for IsaacAccessEnv.

Requires Isaac Sim inside the Docker container. Run with:

    docker exec e29b1e5135c5 bash -c \
      "TERM=xterm /isaac-sim/IsaacLab/isaaclab.sh -p \
       /data/ERCP/ercp_access/tests/test_isaac_integration.py"

Tests:
  IT1  reset() obs contract (shapes, dtypes, value ranges)
  IT2  initial oracle state numerical ranges across 5 seeds
  IT3  step() info dict contains all required fields
  IT4  100 random steps — no crash, no NaN
  IT5  B0 ScriptedPolicy 3-episode success rate == 100%

NOTE: All tests share a single IsaacAccessEnv instance because
SimulationApp is a singleton — calling close() shuts it down permanently.
env.close() is called once at the very end.
"""
from __future__ import annotations

import sys
import os

# Make project root importable when run via isaaclab.sh -p
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Isaac Sim absorbs stdout; redirect to stderr
sys.stdout = sys.stderr

import numpy as np


def _assert(condition, msg):
    if not condition:
        raise AssertionError(msg)


# ─────────────────────────────────────────────────────────────────────────────
# IT1: reset() obs contract
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_obs_contract(env):
    print("\n[IT1] reset() obs contract ...")
    obs, info = env.reset(seed=0)

    _assert(obs["rgb"].shape == (3, 512, 512),
            f"rgb shape={obs['rgb'].shape} != (3,512,512)")
    _assert(obs["rgb"].dtype == np.float32,
            f"rgb dtype={obs['rgb'].dtype} != float32")
    _assert(obs["rgb"].min() >= 0.0 and obs["rgb"].max() <= 1.0,
            f"rgb out of [0,1]: min={obs['rgb'].min():.4f} max={obs['rgb'].max():.4f}")

    _assert(obs["state"].shape == (6,),
            f"state shape={obs['state'].shape} != (6,)")
    _assert(obs["state"].dtype == np.float32,
            f"state dtype={obs['state'].dtype} != float32")
    _assert(not np.any(np.isnan(obs["state"])),
            f"NaN in state: {obs['state']}")

    _assert(obs["phase"].dtype == np.int64,
            f"phase dtype={obs['phase'].dtype} != int64")

    _assert(obs["prev_action"].shape == (3,),
            f"prev_action shape={obs['prev_action'].shape} != (3,)")

    _assert(isinstance(obs["valid"], (bool, np.bool_)),
            f"valid type={type(obs['valid'])} not bool")

    _assert("episode_id" in info, "info missing 'episode_id'")

    print("  [IT1] PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# IT2: initial oracle state numerical ranges
# ─────────────────────────────────────────────────────────────────────────────

def test_initial_oracle_state_ranges(env):
    print("\n[IT2] initial oracle state numerical ranges ...")

    for seed in range(5):
        obs, _ = env.reset(seed=seed)
        s = obs["state"]
        e_x, e_y, scale, conf, stability, readiness = s

        _assert(-1.5 <= e_x <= 1.5, f"seed={seed}: e_x={e_x:.4f} out of [-1.5, 1.5]")
        _assert(-1.5 <= e_y <= 1.5, f"seed={seed}: e_y={e_y:.4f} out of [-1.5, 1.5]")
        _assert(0.0 < scale <= 1.0, f"seed={seed}: scale={scale:.4f} out of (0, 1]")
        _assert(conf == 1.0, f"seed={seed}: conf={conf} != 1.0 in oracle mode")
        _assert(0.0 <= stability <= 1.0, f"seed={seed}: stability={stability:.4f} out of [0,1]")
        _assert(0.0 <= readiness <= 1.0, f"seed={seed}: readiness={readiness:.4f} out of [0,1]")
        _assert(not np.any(np.isnan(s)), f"seed={seed}: NaN in state: {s}")

        print(f"  seed={seed}: e_x={e_x:.3f} e_y={e_y:.3f} scale={scale:.3f} "
              f"conf={conf:.3f} stab={stability:.3f} ready={readiness:.3f}")

    print("  [IT2] PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# IT3: step() info required fields
# ─────────────────────────────────────────────────────────────────────────────

def test_step_info_required_fields(env):
    print("\n[IT3] step() info required fields ...")
    env.reset(seed=0)

    action = np.zeros(3, dtype=np.float32)
    _, reward, _, _, info = env.step(action)

    required_keys = [
        "phase", "gating", "alignment_error", "conf", "readiness",
        "target_visible", "insert_executed", "event_flags", "success",
    ]
    for key in required_keys:
        _assert(key in info, f"info missing required key: '{key}'")

    ef = info["event_flags"]
    required_flags = [
        "target_loss", "off_axis", "no_progress",
        "recovery_triggered", "unsafe_insert_attempt",
    ]
    for flag in required_flags:
        _assert(flag in ef, f"event_flags missing: '{flag}'")

    _assert(isinstance(info["phase"], int), f"phase type={type(info['phase'])}")
    _assert(isinstance(info["gating"], int), f"gating type={type(info['gating'])}")
    _assert(isinstance(info["alignment_error"], float),
            f"alignment_error type={type(info['alignment_error'])}")
    _assert(isinstance(info["success"], bool), f"success type={type(info['success'])}")
    _assert(not np.isnan(reward), f"reward is NaN")

    print(f"  phase={info['phase']} gating={info['gating']} "
          f"align_err={info['alignment_error']:.4f} reward={reward:.4f}")

    print("  [IT3] PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# IT4: 100 random steps — no crash, no NaN
# ─────────────────────────────────────────────────────────────────────────────

def test_100_random_steps_no_crash(env):
    print("\n[IT4] 100 random steps — no crash, no NaN ...")
    env.reset(seed=42)
    rng = np.random.default_rng(42)

    for t in range(100):
        action = rng.uniform(-0.5, 0.5, 3).astype(np.float32)
        obs, r, term, trunc, info = env.step(action)

        _assert(not np.any(np.isnan(obs["state"])),
                f"NaN in state at step {t}: {obs['state']}")
        _assert(not np.isnan(r), f"NaN reward at step {t}")
        _assert(info["phase"] in (0, 1, 2, 3),
                f"Invalid phase={info['phase']} at step {t}")

        if t % 20 == 0:
            print(f"  t={t:3d} phase={info['phase']} "
                  f"align={info['alignment_error']:.3f} reward={r:.3f}")

        if term or trunc:
            print(f"  Episode ended at step {t}: term={term} trunc={trunc}")
            break

    print("  [IT4] PASSED")


# ─────────────────────────────────────────────────────────────────────────────
# IT5: B0 ScriptedPolicy — 3-episode success rate
# ─────────────────────────────────────────────────────────────────────────────

def test_b0_scripted_3_episodes(env):
    print("\n[IT5] B0 ScriptedPolicy — 3 episodes ...")
    from src.policies.scripted_policy import ScriptedPolicy
    from src.controllers.postprocessor import ActionPostprocessor

    policy = ScriptedPolicy()
    pp = ActionPostprocessor()

    successes = []
    for ep in range(3):
        obs, _ = env.reset(seed=42 + ep)
        policy.reset()
        pp.reset()

        last_info = {}
        term = trunc = False
        t = 0

        while not (term or trunc):
            raw = policy.act(obs)
            raw = env.validate_action(raw)
            _phase = int(last_info.get("phase", int(obs.get("phase", 0))))
            _gating = int(last_info.get("gating", _phase))
            action = pp.process(raw, _phase, _gating)
            action = env.validate_action(action)
            obs, _, term, trunc, last_info = env.step(action)
            t += 1

        success = last_info.get("success", False)
        result = "SUCCESS" if success else ("ABORT" if trunc else "TERM")
        print(f"  Ep {ep+1}: {result} in {t} steps "
              f"(phase={last_info.get('phase','?')} "
              f"align={last_info.get('alignment_error', 0):.3f})")
        successes.append(int(success))

    _assert(sum(successes) >= 3,
            f"B0 oracle success < 3/3: {successes}. Expected 100% in oracle mode.")

    print(f"  [IT5] PASSED  ({sum(successes)}/3 success)")


# ─────────────────────────────────────────────────────────────────────────────
# Runner — single shared env instance, close() at the very end
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.envs.isaac_access_env import IsaacAccessEnv

    print(f"\n{'='*60}")
    print(f"  IsaacAccessEnv Integration Tests")
    print(f"{'='*60}")

    # Single env — SimulationApp is singleton, cannot be re-created after close()
    env = IsaacAccessEnv(use_oracle=True, headless=True, max_steps=200)

    tests = [
        ("IT1", test_reset_obs_contract),
        ("IT2", test_initial_oracle_state_ranges),
        ("IT3", test_step_info_required_fields),
        ("IT4", test_100_random_steps_no_crash),
        ("IT5", test_b0_scripted_3_episodes),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn(env)
            passed += 1
        except AssertionError as e:
            print(f"\n  [{name}] FAILED: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"\n  [{name}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1

    # Close once after all tests
    env.close()

    print(f"\n{'='*60}")
    print(f"  Results: {passed}/{passed+failed} passed")
    print(f"{'='*60}\n")

    if failed > 0:
        sys.exit(1)
