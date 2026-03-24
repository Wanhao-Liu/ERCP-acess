#!/usr/bin/env python3
"""
Train B2 PPO-State baseline on ToyAccessEnv.

B2 definition: no FSM at policy level, pure state -> action.
Observation: concat(state[6], prev_action[3]) = 9D flat.
Action: continuous [-1, 1]^3.

Usage:
    cd /data/ERCP/ercp_access
    conda run -n ercp python scripts/train/train_b2_ppo.py
    conda run -n ercp python scripts/train/train_b2_ppo.py --total_steps 1000000 --n_envs 8
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed

from src.envs.gymnasium_wrapper import ToyAccessGymEnv
from src.envs.toy_access_env import ToyAccessEnv
from src.eval.evaluator import Evaluator
from src.policies.ppo_state_policy import PPOStatePolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Train B2 PPO-State baseline")
    parser.add_argument("--total_steps",   type=int,   default=500_000)
    parser.add_argument("--n_envs",        type=int,   default=8)
    parser.add_argument("--n_steps",       type=int,   default=2048)
    parser.add_argument("--batch_size",    type=int,   default=64)
    parser.add_argument("--n_epochs",      type=int,   default=10)
    parser.add_argument("--gamma",         type=float, default=0.99)
    parser.add_argument("--gae_lambda",    type=float, default=0.95)
    parser.add_argument("--clip_range",    type=float, default=0.2)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--ent_coef",      type=float, default=0.01)
    parser.add_argument("--vf_coef",       type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--eval_freq",     type=int,   default=10_000)
    parser.add_argument("--n_eval_ep",     type=int,   default=50)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--output_dir",    type=str,   default="outputs/b2_ppo")
    parser.add_argument("--exp_name",      type=str,   default="b2_ppo_state")
    return parser.parse_args()


def make_env(rank: int, seed: int = 0):
    def _init():
        env = ToyAccessGymEnv(max_steps=200, difficulty_cycle=True, seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


def main():
    args = parse_args()

    output_dir = os.path.join(args.output_dir, args.exp_name)
    ckpt_dir   = os.path.join(output_dir, "checkpoints")
    eval_dir   = os.path.join(output_dir, "eval_log")
    tb_dir     = os.path.join(output_dir, "tb")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(tb_dir, exist_ok=True)

    print(f"[B2 PPO] Output dir: {output_dir}")
    print(f"[B2 PPO] total_steps={args.total_steps}, n_envs={args.n_envs}, seed={args.seed}")

    # ── Vectorized training envs ──────────────────────────────────────────
    train_env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    train_env = VecMonitor(train_env)

    # ── Eval env (single) ────────────────────────────────────────────────
    eval_gym = ToyAccessGymEnv(max_steps=200, difficulty_cycle=True, seed=args.seed + 999)

    # ── PPO model ─────────────────────────────────────────────────────────
    policy_kwargs = dict(net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128]))
    model = PPO(
        policy          = "MlpPolicy",
        env             = train_env,
        learning_rate   = args.lr,
        n_steps         = args.n_steps,
        batch_size      = args.batch_size,
        n_epochs        = args.n_epochs,
        gamma           = args.gamma,
        gae_lambda      = args.gae_lambda,
        clip_range      = args.clip_range,
        ent_coef        = args.ent_coef,
        vf_coef         = args.vf_coef,
        max_grad_norm   = args.max_grad_norm,
        policy_kwargs   = policy_kwargs,
        verbose         = 1,
        seed            = args.seed,
        tensorboard_log = tb_dir,
        device          = "auto",
    )

    # ── Callbacks ─────────────────────────────────────────────────────────
    eval_cb = EvalCallback(
        eval_gym,
        best_model_save_path = ckpt_dir,
        log_path             = eval_dir,
        eval_freq            = max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes      = args.n_eval_ep,
        deterministic        = True,
        verbose              = 1,
    )
    ckpt_cb = CheckpointCallback(
        save_freq   = max(50_000 // args.n_envs, 1),
        save_path   = ckpt_dir,
        name_prefix = "b2_ppo",
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[B2 PPO] Starting training...")
    model.learn(
        total_timesteps     = args.total_steps,
        callback            = [eval_cb, ckpt_cb],
        reset_num_timesteps = True,
        progress_bar        = False,
    )

    # ── Save final model ──────────────────────────────────────────────────
    final_path = os.path.join(ckpt_dir, "final_model")
    model.save(final_path)
    print(f"\n[B2 PPO] Saved final model: {final_path}.zip")

    train_env.close()
    eval_gym.close()

    # ── Final evaluation with Evaluator ───────────────────────────────────
    print("\n[B2 PPO] Running final 100-episode evaluation...")
    policy = PPOStatePolicy(final_path)
    toy_env = ToyAccessEnv(seed=args.seed)
    evaluator = Evaluator(output_dir=os.path.join(output_dir, "final_eval"))

    # evaluator.evaluate returns RunMetrics dataclass
    metrics = evaluator.evaluate(toy_env, policy.act, n_episodes=100, postprocessor=None)

    toy_env.close()

    print("\n" + "=" * 55)
    print("  B2 PPO-STATE BASELINE RESULTS (100 episodes)")
    print("=" * 55)
    # RunMetrics is a dataclass — use vars()
    if hasattr(metrics, '__dict__'):
        for k, v in vars(metrics).items():
            print(f"  {k:30s}: {v}")
    elif isinstance(metrics, dict):
        for k, v in metrics.items():
            print(f"  {k:30s}: {v}")
    else:
        print(metrics)
    print("=" * 55)


if __name__ == "__main__":
    main()
