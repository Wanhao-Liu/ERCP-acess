"""
StateBuilder — converts RGB (or oracle GT) into AccessState.

Phase 0: use_oracle=True, reads GT from sim.
Phase 1+: FastSAM segmentation → moments → AccessState.

Maintains a history deque for stability computation.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Dict, Any, Optional

import numpy as np

from src.perception.access_state import AccessState


class StateBuilder:
    """
    Stateful module: maintains history of [e_x, e_y] for stability.

    Args:
        history_len: Number of frames for stability window (default 8).
        use_oracle: If True, compute state from gt_info instead of RGB.
        e_thresh: Alignment threshold for readiness computation.
        scale_min: Minimum scale for readiness computation.
        detector_weights: Path to YOLOv8 weights for papilla detection.
                          Empty string (default) disables detector.
        detector_conf_thresh: Confidence threshold for PapillaDetector (default 0.25).
        detector_device: Device for PapillaDetector inference (default 'cuda').
    """

    def __init__(
        self,
        history_len: int = 8,
        use_oracle: bool = True,
        e_thresh: float = 0.1,
        scale_min: float = 0.25,
        detector_weights: str = "",
        detector_conf_thresh: float = 0.25,
        detector_device: str = "cuda",
    ):
        self.history_len = history_len
        self.use_oracle = use_oracle
        self.e_thresh = e_thresh
        self.scale_min = scale_min
        self._history: deque = deque(maxlen=history_len)

        # Phase 1+: instantiate PapillaDetector if weights are provided
        self._detector = None
        if detector_weights:
            from src.perception.papilla_detector import PapillaDetector
            self._detector = PapillaDetector(
                weights=detector_weights,
                conf_thresh=detector_conf_thresh,
                device=detector_device,
            )

    def reset(self):
        self._history.clear()

    def update(
        self,
        rgb: np.ndarray,
        gt_info: Optional[Dict[str, Any]] = None,
    ) -> AccessState:
        """
        Compute AccessState from current frame.

        Args:
            rgb: float32 [3, H, W] or [H, W, 3], normalized [0,1].
            gt_info: dict with keys {e_x, e_y, scale, conf} for oracle mode.

        Returns:
            AccessState
        """
        if self.use_oracle:
            assert gt_info is not None, "gt_info required in oracle mode"
            e_x = float(gt_info["e_x"])
            e_y = float(gt_info["e_y"])
            scale = float(gt_info.get("scale", 0.3))
            conf = float(gt_info.get("conf", 1.0))
        else:
            e_x, e_y, scale, conf = self._run_segmentation(rgb)

        # Update history and compute stability
        self._history.append([e_x, e_y])
        stability = self._compute_stability()

        readiness = AccessState.compute_readiness(
            e_x, e_y, scale, conf, stability,
            e_thresh=self.e_thresh,
            scale_min=self.scale_min,
        )

        return AccessState(
            e_x=e_x,
            e_y=e_y,
            scale=scale,
            conf=conf,
            stability=stability,
            readiness=readiness,
        )

    def _compute_stability(self) -> float:
        """
        stability = exp(-var([e_x, e_y] over history))
        Returns 1.0 if history has < 2 frames (not enough data).
        """
        if len(self._history) < 2:
            return 1.0
        arr = np.array(self._history)  # (N, 2)
        var = float(np.var(arr))
        return float(math.exp(-var * 10.0))  # scale factor 10 for sensitivity

    def _run_segmentation(
        self, rgb: np.ndarray
    ) -> tuple[float, float, float, float]:
        """
        Phase 1+: PapillaDetector (YOLOv8) → (e_x, e_y, scale, conf).
        Returns (0.0, 0.0, 0.0, 0.0) if no detector is configured or no detection.
        """
        if self._detector is not None:
            return self._detector.detect(rgb)
        return 0.0, 0.0, 0.0, 0.0
