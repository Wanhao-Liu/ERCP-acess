#!/usr/bin/env python3
"""
Run scripted baseline (B0) on ToyAccessEnv.

Usage:
    python scripts/run_scripted.py
    python scripts/run_scripted.py --n_episodes 100 --difficulty medium
    python scripts/run_scripted.py --output_dir outputs/scripted_run
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.envs.toy_access_env import ToyAccessEnv
from src.policies.scripted_policy import ScriptedPolicy
from src.eval.evaluator import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--difficulty", type=str, default=None,
                        help="fixed difficulty: easy/medium/hard (default: cycle all)")
    parser.add_argument("--output_dir", type=str, default="outputs/scripted")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    env = ToyAccessEnv(seed=args.seed)
    policy = ScriptedPolicy()
    evaluator = Evaluator(
        output_dir=args.output_dir,
        difficulties=[args.difficulty] if args.difficulty else ["easy", "medium", "hard"],
    )

    print(f"[B0 Scripted] Running {args.n_episodes} episodes on ToyAccessEnv...")
    metrics = evaluator.evaluate(env, policy.act, n_episodes=args.n_episodes)

    print("\n" + "=" * 50)
    print("B0 SCRIPTED BASELINE RESULTS")
    print("=" * 50)
    print(f"Success Rate:      {metrics.access_success_rate:.1%}")
    print(f"Abort Rate:        {metrics.abort_rate:.1%}")
    print(f"Mean Steps:        {metrics.mean_steps_to_access:.0f}")
    print(f"Mean Insert Tries: {metrics.mean_insert_attempts:.1f}")
    print(f"Off-axis Rate:     {metrics.off_axis_rate:.1%}")
    print(f"Recovery Rate:     {metrics.recovery_rate:.1%}")

    env.close()


if __name__ == "__main__":
    main()
