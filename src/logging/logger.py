"""
StepLogger, EpisodeLogger, RunLogger — structured logging for access experiments.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.envs.base_env import EpisodeMetrics, RunMetrics


class StepLogger:
    """Accumulates per-step data within a single episode."""

    def __init__(self):
        self._steps: List[Dict[str, Any]] = []
        self._episode_id: str = ""
        self._case_id: str = ""
        self._dt: float = 0.02

    def start_episode(self, episode_id: str, case_id: str = "", dt: float = 0.02):
        self._steps = []
        self._episode_id = episode_id
        self._case_id = case_id
        self._dt = dt

    def log_step(
        self,
        t: int,
        raw_action: np.ndarray,
        exec_action: np.ndarray,
        reward: float,
        info: Dict[str, Any],
    ):
        record = {
            "episode_id": self._episode_id,
            "t": t,
            "reward": reward,
            "raw_action": raw_action.tolist(),
            "exec_action": exec_action.tolist(),
        }
        for k, v in info.items():
            if k == "event_flags":
                for fk, fv in v.items():
                    record[f"flag_{fk}"] = fv
            else:
                record[k] = v
        self._steps.append(record)

    def finalize(self, success: bool) -> EpisodeMetrics:
        n = len(self._steps)
        if n == 0:
            return EpisodeMetrics(episode_id=self._episode_id, case_id=self._case_id)

        alignment_errors = [s.get("alignment_error", 0.0) for s in self._steps]
        confs = [s.get("conf", 0.0) for s in self._steps]
        readinesses = [s.get("readiness", 0.0) for s in self._steps]

        return EpisodeMetrics(
            episode_id=self._episode_id,
            case_id=self._case_id,
            success=int(success),
            total_reward=sum(s.get("reward", 0.0) for s in self._steps),
            steps=n,
            time_to_access=n * self._dt,
            num_insert_attempts=sum(1 for s in self._steps if s.get("insert_executed")),
            target_loss_count=sum(1 for s in self._steps if s.get("flag_target_loss")),
            off_axis_count=sum(1 for s in self._steps if s.get("flag_off_axis")),
            recovery_count=sum(1 for s in self._steps if s.get("flag_recovery_triggered")),
            no_progress_count=sum(1 for s in self._steps if s.get("flag_no_progress")),
            unsafe_insert_attempts=sum(1 for s in self._steps if s.get("flag_unsafe_insert_attempt")),
            mean_alignment_error=float(np.mean(alignment_errors)),
            max_alignment_error=float(np.max(alignment_errors)),
            mean_conf=float(np.mean(confs)),
            mean_readiness=float(np.mean(readinesses)),
        )


class EpisodeLogger:
    """Collects EpisodeMetrics and writes summary files."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._episodes: List[EpisodeMetrics] = []

    def add(self, metrics: EpisodeMetrics):
        self._episodes.append(metrics)

    def get_run_metrics(self) -> RunMetrics:
        n = len(self._episodes)
        if n == 0:
            return RunMetrics()
        successes = sum(e.success for e in self._episodes)
        successful = [e for e in self._episodes if e.success]
        return RunMetrics(
            n_episodes=n,
            access_success_rate=successes / n,
            mean_time_to_access=float(np.mean([e.time_to_access for e in successful])) if successful else 0.0,
            mean_steps_to_access=float(np.mean([e.steps for e in successful])) if successful else 0.0,
            mean_insert_attempts=float(np.mean([e.num_insert_attempts for e in self._episodes])),
            target_loss_rate=float(np.mean([e.target_loss_count > 0 for e in self._episodes])),
            off_axis_rate=float(np.mean([e.off_axis_count > 0 for e in self._episodes])),
            recovery_rate=float(np.mean([e.recovery_count > 0 for e in self._episodes])),
            abort_rate=(n - successes) / n,
            mean_alignment_error=float(np.mean([e.mean_alignment_error for e in self._episodes])),
        )

    def save(self, prefix: str = "eval"):
        if not self._episodes:
            return
        # CSV
        csv_path = self.output_dir / f"{prefix}_episodes.csv"
        fieldnames = list(asdict(self._episodes[0]).keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for ep in self._episodes:
                writer.writerow(asdict(ep))
        # JSON summary
        json_path = self.output_dir / f"{prefix}_run_metrics.json"
        with open(json_path, "w") as f:
            json.dump(asdict(self.get_run_metrics()), f, indent=2)
        print(f"[Logger] Saved to {self.output_dir}")


class RunLogger:
    """Optional W&B integration."""

    def __init__(self, cfg: Dict[str, Any], use_wandb: bool = False):
        self.use_wandb = use_wandb
        self._run = None
        if use_wandb:
            try:
                import wandb
                self._run = wandb.init(
                    project=cfg.get("project_name", "ercp_access"),
                    config=cfg,
                )
            except ImportError:
                print("[RunLogger] wandb not installed, disabling")
                self.use_wandb = False

    def log(self, data: Dict[str, Any], step: int):
        if self.use_wandb and self._run:
            import wandb
            wandb.log(data, step=step)

    def finish(self):
        if self.use_wandb and self._run:
            import wandb
            wandb.finish()
