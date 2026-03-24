#!/usr/bin/env python3
"""
Run B1 PID baseline evaluation on ToyAccessEnv.

B1 architecture: Oracle AccessState -> PIDPolicy -> ActionPostprocessor -> env.step()
(Same FSM structure as Ours V2, PID controller instead of learned policy.)

Usage:
    cd /data/ERCP/ercp_access
    conda run -n ercp python scripts/run_b1_pid.py
    conda run -n ercp python scripts/run_b1_pid.py --n_episodes 100 --compare_b0
    conda run -n ercp python scripts/run_b1_pid.py --kp_align 3.0 --kd_align 0.2
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.toy_access_env import ToyAccessEnv
from src.policies.pid_policy import PIDPolicy
from src.controllers.postprocessor import ActionPostprocessor
from src.eval.evaluator import Evaluator


def parse_args():
    p = argparse.ArgumentParser(description="B1 PID baseline evaluation")
    p.add_argument("--n_episodes",  type=int,   default=100)
    p.add_argument("--output_dir",  type=str,   default="outputs/b1_pid")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--compare_b0",  action="store_true")
    p.add_argument("--kp_align",    type=float, default=3.0)
    p.add_argument("--ki_align",    type=float, default=0.05)
    p.add_argument("--kd_align",    type=float, default=0.2)
    p.add_argument("--kp_insert",   type=float, default=2.0)
    p.add_argument("--ki_insert",   type=float, default=0.02)
    p.add_argument("--kd_insert",   type=float, default=0.2)
    return p.parse_args()


def run_eval(env, policy, pp, output_dir, n_episodes):
    """Run evaluation using Evaluator with postprocessor."""
    evaluator = Evaluator(output_dir=output_dir)
    metrics = evaluator.evaluate(
        env, policy.act, n_episodes=n_episodes,
        postprocessor=pp.process,
    )
    return metrics


def print_metrics(name, m):
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    # RunMetrics is a dataclass; access fields as attributes
    fields = [
        ("access_success_rate",  "rate"),
        ("abort_rate",           "rate"),
        ("mean_steps_to_access", "float"),
        ("mean_insert_attempts", "float"),
        ("target_loss_rate",     "rate"),
        ("off_axis_rate",        "rate"),
        ("recovery_rate",        "rate"),
        ("mean_alignment_error", "float"),
    ]
    for k, kind in fields:
        v = getattr(m, k, None)
        if v is None:
            continue
        if kind == "rate":
            print(f"  {k:<30}: {v:.1%}")
        else:
            print(f"  {k:<30}: {v:.4f}")


def get_metric(m, key, default=0.0):
    return getattr(m, key, default)


def main():
    args = parse_args()

    # ── Optional B0 comparison ─────────────────────────────────────────
    b0_metrics = None
    if args.compare_b0:
        from src.policies.scripted_policy import ScriptedPolicy
        print("[B0 Scripted] Running for comparison...")
        env_b0 = ToyAccessEnv(seed=args.seed)
        policy_b0 = ScriptedPolicy()
        pp_b0 = ActionPostprocessor()
        b0_metrics = run_eval(
            env_b0, policy_b0, pp_b0,
            output_dir=str(Path(args.output_dir) / "b0_compare"),
            n_episodes=args.n_episodes,
        )
        env_b0.close()
        print_metrics("B0 Scripted", b0_metrics)

    # ── B1 PID evaluation ──────────────────────────────────────────────
    print(f"\n[B1 PID] Running {args.n_episodes} episodes...")
    env = ToyAccessEnv(seed=args.seed)
    policy = PIDPolicy(
        kp_align=args.kp_align,
        ki_align=args.ki_align,
        kd_align=args.kd_align,
        kp_insert=args.kp_insert,
        ki_insert=args.ki_insert,
        kd_insert=args.kd_insert,
    )
    pp = ActionPostprocessor()

    b1_metrics = run_eval(
        env, policy, pp,
        output_dir=args.output_dir,
        n_episodes=args.n_episodes,
    )
    env.close()

    print_metrics("B1 PID", b1_metrics)

    # ── Comparison summary ─────────────────────────────────────────────
    if b0_metrics is not None:
        print(f"\n{'='*50}")
        print("  DELTA: B1 PID vs B0 Scripted")
        print(f"{'='*50}")
        keys = ["access_success_rate", "abort_rate", "mean_steps_to_access",
                "off_axis_rate", "recovery_rate", "mean_alignment_error"]
        for k in keys:
            v0 = get_metric(b0_metrics, k)
            v1 = get_metric(b1_metrics, k)
            delta = v1 - v0
            arrow = "up" if delta > 0 else "down"
            print(f"  {k:<30}: {v0:.4f} -> {v1:.4f}  ({arrow} {abs(delta):.4f})")

    # ── Save JSON summary ──────────────────────────────────────────────
    out_path = Path(args.output_dir) / "val" / "b1_pid_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "baseline": "B1_PID",
        "n_episodes": args.n_episodes,
        "config": {
            "kp_align": args.kp_align, "ki_align": args.ki_align,
            "kd_align": args.kd_align, "kp_insert": args.kp_insert,
            "ki_insert": args.ki_insert, "kd_insert": args.kd_insert,
        },
        "metrics": {k: get_metric(b1_metrics, k) for k in [
            "access_success_rate", "abort_rate", "mean_steps_to_access",
            "mean_insert_attempts", "off_axis_rate", "recovery_rate",
            "mean_alignment_error",
        ]},
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Saved] {out_path}")


if __name__ == "__main__":
    main()
