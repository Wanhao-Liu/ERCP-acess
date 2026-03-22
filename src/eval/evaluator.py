"""
Evaluator — runs N episodes and produces RunMetrics.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.envs.base_env import AccessEnvBase, RunMetrics
from src.logging.logger import StepLogger, EpisodeLogger


class Evaluator:
    """
    Policy-agnostic, env-agnostic evaluator.

    Usage:
        ev = Evaluator("outputs/run_001")
        metrics = ev.evaluate(env, policy, n_episodes=100)
    """

    def __init__(
        self,
        output_dir: str = "outputs/eval",
        save_video: bool = False,
        difficulties: Optional[List[str]] = None,
    ):
        self.output_dir = Path(output_dir)
        self.save_video = save_video
        self.difficulties = difficulties or ["easy", "medium", "hard"]

    def evaluate(
        self,
        env: AccessEnvBase,
        policy: Callable,
        n_episodes: int = 100,
        postprocessor: Optional[Callable] = None,
        split: str = "val",
    ) -> RunMetrics:
        ep_logger = EpisodeLogger(str(self.output_dir / split))
        step_logger = StepLogger()

        for i in range(n_episodes):
            difficulty = self.difficulties[i % len(self.difficulties)]
            episode_id = f"{split}_{i:05d}_{uuid.uuid4().hex[:6]}"

            obs, reset_info = env.reset(case_id=episode_id, difficulty=difficulty)
            step_logger.start_episode(episode_id, dt=env.dt)

            terminated = truncated = False
            t = 0
            last_info: Dict[str, Any] = {}

            while not (terminated or truncated):
                raw_action = policy(obs)
                raw_action = env.validate_action(raw_action)

                exec_action = raw_action.copy()
                if postprocessor is not None:
                    _phase = int(last_info.get("phase", int(obs.get("phase", 0))))
                    _gating = int(last_info.get("gating", _phase))
                    exec_action = postprocessor(raw_action, _phase, _gating)
                    exec_action = env.validate_action(exec_action)

                obs, reward, terminated, truncated, last_info = env.step(exec_action)
                step_logger.log_step(t, raw_action, exec_action, reward, last_info)
                t += 1

            ep_metrics = step_logger.finalize(success=last_info.get("success", False))
            ep_logger.add(ep_metrics)

            if (i + 1) % 10 == 0:
                interim = ep_logger.get_run_metrics()
                print(f"  [{split}] {i+1}/{n_episodes}  "
                      f"success={interim.access_success_rate:.1%}")

        ep_logger.save(prefix=split)
        run_metrics = ep_logger.get_run_metrics()
        print(f"\n[Eval] {split}: success={run_metrics.access_success_rate:.1%}  "
              f"abort={run_metrics.abort_rate:.1%}  "
              f"mean_steps={run_metrics.mean_steps_to_access:.0f}")
        return run_metrics
