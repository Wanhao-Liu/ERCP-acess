#!/usr/bin/env python3
"""
Run B0 ScriptedPolicy on IsaacAccessEnv (oracle mode).

Prints per-episode phase sequences, success/failure, alignment error,
and final run-level metrics.

Usage (inside container):
    cd /data/ERCP/ercp_access
    TERM=xterm /isaac-sim/IsaacLab/isaaclab.sh -p scripts/eval/run_isaac_scripted.py
    TERM=xterm /isaac-sim/IsaacLab/isaaclab.sh -p scripts/eval/run_isaac_scripted.py --n_episodes 10
"""
from __future__ import annotations

import argparse
import sys
import os

# Make project root importable when run via isaaclab.sh -p
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np

# Isaac Sim absorbs stdout; write all output to stderr so it's visible
import sys as _sys
_orig_stdout = _sys.stdout
_sys.stdout = _sys.stderr


def parse_args():
    parser = argparse.ArgumentParser(description="B0 ScriptedPolicy on IsaacAccessEnv")
    parser.add_argument("--n_episodes", type=int, default=5,
                        help="Number of episodes to run (default: 5)")
    parser.add_argument("--max_steps", type=int, default=1500,
                        help="Max steps per episode (default: 1500)")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Open Isaac Sim GUI window (default: headless)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    # isaaclab.sh passes extra args; absorb unknown ones
    args, _ = parser.parse_known_args()
    return args


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  B0 ScriptedPolicy — IsaacAccessEnv (oracle mode)")
    print(f"  n_episodes={args.n_episodes}  max_steps={args.max_steps}")
    print(f"{'='*60}\n")

    # ── Create environment ───────────────────────────────────────────────────
    # IsaacAccessEnv starts SimulationApp on __init__
    from src.envs.isaac_access_env import IsaacAccessEnv
    from src.policies.scripted_policy import ScriptedPolicy
    from src.controllers.postprocessor import ActionPostprocessor

    env = IsaacAccessEnv(
        use_oracle=True,
        headless=not args.gui,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    policy = ScriptedPolicy()
    postprocessor = ActionPostprocessor()

    # ── Run episodes ─────────────────────────────────────────────────────────
    successes = []
    step_counts = []
    insert_attempt_counts = []
    alignment_errors = []
    # Safety/event metric accumulators (per-episode: did event occur at all?)
    target_loss_occurred = []
    off_axis_occurred = []
    recovery_occurred = []

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=args.seed + ep)
        policy.reset()
        postprocessor.reset()

        ep_phases = []
        ep_align_errors = []
        ep_inserts = 0
        ep_target_loss = False
        ep_off_axis = False
        ep_recovery = False
        last_info = {}
        terminated = truncated = False
        t = 0

        while not (terminated or truncated):
            raw_action = policy.act(obs)
            raw_action = env.validate_action(raw_action)

            # Postprocessor needs phase and gating from last step
            _phase = int(last_info.get("phase", int(obs.get("phase", 0))))
            _gating = int(last_info.get("gating", _phase))
            exec_action = postprocessor.process(raw_action, _phase, _gating)
            exec_action = env.validate_action(exec_action)

            obs, reward, terminated, truncated, last_info = env.step(exec_action)

            ep_phases.append(last_info["phase"])
            ep_align_errors.append(last_info["alignment_error"])
            if last_info.get("insert_executed", False):
                ep_inserts += 1

            # Track safety events
            flags = last_info.get("event_flags", {})
            if flags.get("target_loss"):
                ep_target_loss = True
            if flags.get("off_axis"):
                ep_off_axis = True
            if last_info.get("phase") == 3:
                ep_recovery = True

            t += 1

        success = last_info.get("success", False)
        successes.append(int(success))
        step_counts.append(t)
        insert_attempt_counts.append(ep_inserts)
        alignment_errors.append(float(np.mean(ep_align_errors)) if ep_align_errors else 0.0)
        target_loss_occurred.append(int(ep_target_loss))
        off_axis_occurred.append(int(ep_off_axis))
        recovery_occurred.append(int(ep_recovery))

        # Per-episode summary
        phase_str = "".join(str(p) for p in ep_phases[-20:])  # last 20 steps
        result_str = "SUCCESS" if success else ("ABORT" if truncated else "TERM")
        print(
            f"  Ep {ep+1:02d}/{args.n_episodes}  {result_str:7s}  "
            f"steps={t:4d}  inserts={ep_inserts:2d}  "
            f"align_err={alignment_errors[-1]:.3f}  "
            f"phase_tail=[{phase_str}]"
        )

    # ── Run-level metrics ────────────────────────────────────────────────────
    n = args.n_episodes
    success_rate = float(np.mean(successes))
    mean_steps = float(np.mean(step_counts))
    mean_inserts = float(np.mean(insert_attempt_counts))
    mean_align = float(np.mean(alignment_errors))
    abort_rate = 1.0 - success_rate
    target_loss_rate = float(np.mean(target_loss_occurred))
    off_axis_rate = float(np.mean(off_axis_occurred))
    recovery_rate = float(np.mean(recovery_occurred))

    print(f"\n{'='*60}")
    print(f"  Run Metrics  ({n} episodes)")
    print(f"{'='*60}")
    print(f"  success_rate        : {success_rate:.1%}")
    print(f"  abort_rate          : {abort_rate:.1%}")
    print(f"  mean_steps          : {mean_steps:.1f}")
    print(f"  mean_insert_attempts: {mean_inserts:.1f}")
    print(f"  mean_alignment_error: {mean_align:.3f}")
    print(f"  target_loss_rate    : {target_loss_rate:.1%}")
    print(f"  off_axis_rate       : {off_axis_rate:.1%}")
    print(f"  recovery_rate       : {recovery_rate:.1%}")
    print(f"{'='*60}\n")

    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
