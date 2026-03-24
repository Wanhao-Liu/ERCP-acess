"""
Gymnasium wrapper for ToyAccessEnv — used by B2 PPO-State training.

Observation: concat(state[6], prev_action[3]) = flat (9,) vector
Action:      Box(-1, 1, shape=(3,)) continuous
"""
import gymnasium
import numpy as np
from src.envs.toy_access_env import ToyAccessEnv


class ToyAccessGymEnv(gymnasium.Env):
    """
    Wraps ToyAccessEnv as a gymnasium.Env for SB3 PPO training.

    B2 definition: policy does NOT use phase (no FSM at policy level).
    The env's internal FSM still runs (side effect accepted for fair comparison).
    No postprocessor is applied — raw policy output goes directly to env.step().
    """
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 200, difficulty_cycle: bool = True, seed: int = 0):
        super().__init__()
        self._env = ToyAccessEnv(max_steps=max_steps, seed=seed)
        self._difficulties = ["easy", "medium", "hard"]
        self._difficulty_cycle = difficulty_cycle
        self._ep_count = 0

        # B2 observation: state(6) + prev_action(3) = 9D flat
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def _flatten_obs(self, obs_dict: dict) -> np.ndarray:
        state = np.asarray(obs_dict["state"], dtype=np.float32)          # (6,)
        prev_action = np.asarray(obs_dict["prev_action"], dtype=np.float32)  # (3,)
        return np.concatenate([state, prev_action])                       # (9,)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self._difficulty_cycle:
            difficulty = self._difficulties[self._ep_count % len(self._difficulties)]
        else:
            difficulty = "medium"
        self._ep_count += 1
        obs_dict, info = self._env.reset(seed=seed, difficulty=difficulty)
        return self._flatten_obs(obs_dict), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        obs_dict, reward, terminated, truncated, info = self._env.step(action)
        return self._flatten_obs(obs_dict), float(reward), terminated, truncated, info

    def close(self):
        self._env.close()
