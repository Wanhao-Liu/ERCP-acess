"""
AccessState — central data structure for biliary access control.

All modules (policy, gating, safety, logger) consume this.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class AccessState:
    e_x: float        # horizontal offset from image center, normalized [-1, 1]
    e_y: float        # vertical offset from image center, normalized [-1, 1]
    scale: float      # target apparent scale (area / image_area), [0, 1]
    conf: float       # perception confidence, [0, 1]
    stability: float  # temporal stability of [e_x, e_y] over recent frames, [0, 1]
    readiness: float  # composite insertion readiness, [0, 1]

    # ── Thresholds (class-level defaults, can be overridden) ──────────
    E_THRESH: float = 0.1
    SCALE_MIN: float = 0.25
    STABILITY_MIN: float = 0.6
    CONF_MIN: float = 0.6
    READINESS_MIN: float = 0.7

    def is_aligned(self, e_thresh: float | None = None) -> bool:
        t = e_thresh if e_thresh is not None else self.E_THRESH
        return abs(self.e_x) < t and abs(self.e_y) < t

    def is_ready_for_insert(self) -> bool:
        return (
            self.is_aligned()
            and self.scale >= self.SCALE_MIN
            and self.stability >= self.STABILITY_MIN
            and self.conf >= self.CONF_MIN
            and self.readiness >= self.READINESS_MIN
        )

    @property
    def alignment_error(self) -> float:
        return math.sqrt(self.e_x ** 2 + self.e_y ** 2)

    def to_array(self) -> np.ndarray:
        return np.array(
            [self.e_x, self.e_y, self.scale, self.conf, self.stability, self.readiness],
            dtype=np.float32,
        )

    @staticmethod
    def from_array(arr: np.ndarray) -> "AccessState":
        assert arr.shape == (6,)
        return AccessState(
            e_x=float(arr[0]),
            e_y=float(arr[1]),
            scale=float(arr[2]),
            conf=float(arr[3]),
            stability=float(arr[4]),
            readiness=float(arr[5]),
        )

    @staticmethod
    def compute_readiness(
        e_x: float,
        e_y: float,
        scale: float,
        conf: float,
        stability: float,
        e_thresh: float = 0.1,
        scale_min: float = 0.25,
        stability_min: float = 0.6,
        conf_min: float = 0.6,
    ) -> float:
        """
        Explicit readiness formula:
          alignment_ok = sigmoid(1 - (|e_x| + |e_y|) / e_thresh)
          scale_ok     = sigmoid((scale - scale_min) / 0.05)
          stability_ok = stability  (already in [0,1])
          conf_ok      = conf
          readiness    = min(alignment_ok, scale_ok, stability_ok, conf_ok)
        """
        def sigmoid(x: float) -> float:
            return 1.0 / (1.0 + math.exp(-x))

        alignment_ok = sigmoid(1.0 - (abs(e_x) + abs(e_y)) / max(e_thresh, 1e-6))
        scale_ok = sigmoid((scale - scale_min) / 0.05)
        stability_ok = float(np.clip(stability, 0.0, 1.0))
        conf_ok = float(np.clip(conf, 0.0, 1.0))
        return min(alignment_ok, scale_ok, stability_ok, conf_ok)
