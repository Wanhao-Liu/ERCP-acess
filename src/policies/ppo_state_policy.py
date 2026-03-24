"""
B2 PPO-State policy wrapper — adapts SB3 PPO model to the Evaluator interface.
"""
import numpy as np
from typing import Any, Dict


class PPOStatePolicy:
    """
    Wraps a saved SB3 PPO model for use with Evaluator.evaluate().

    Input:  obs dict with keys 'state'(6,) and 'prev_action'(3,)
    Output: action float32 (3,)
    """

    def __init__(self, model_path: str, device: str = "auto"):
        from stable_baselines3 import PPO
        self._model = PPO.load(model_path, device=device)

    def act(self, obs: Dict[str, Any]) -> np.ndarray:
        state = np.asarray(obs["state"], dtype=np.float32)           # (6,)
        prev_action = np.asarray(obs["prev_action"], dtype=np.float32)  # (3,)
        obs_flat = np.concatenate([state, prev_action])[np.newaxis]   # (1, 9)
        action, _ = self._model.predict(obs_flat, deterministic=True)
        return action.squeeze(0).astype(np.float32)                   # (3,)

    def reset(self):
        """No episode state in SB3 model — no-op."""
        pass
