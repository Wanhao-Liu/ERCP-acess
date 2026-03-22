"""
InsertionGate — checks whether insertion is safe to execute.
SafetyRules — detects dangerous events and fills event_flags.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from src.perception.access_state import AccessState


class InsertionGate:
    """
    All conditions must be met to allow insertion.
    Returns gating int: 0=hold, 1=approach, 2=insert, 3=recover
    """

    def __init__(
        self,
        e_thresh: float = 0.08,
        scale_min: float = 0.25,
        conf_min: float = 0.6,
        stability_min: float = 0.6,
        readiness_min: float = 0.7,
    ):
        self.e_thresh = e_thresh
        self.scale_min = scale_min
        self.conf_min = conf_min
        self.stability_min = stability_min
        self.readiness_min = readiness_min

    def check(self, state: AccessState) -> bool:
        return (
            abs(state.e_x) < self.e_thresh
            and abs(state.e_y) < self.e_thresh
            and state.scale >= self.scale_min
            and state.conf >= self.conf_min
            and state.stability >= self.stability_min
            and state.readiness >= self.readiness_min
        )

    def get_gating(self, state: AccessState, phase: int) -> int:
        if phase == 3:
            return 3  # recover
        if phase == 2:
            return 2 if self.check(state) else 0  # insert or hold
        if phase == 1:
            return 1  # approach
        # phase 0: alignment — allow pitch/yaw, block insert (postprocessor phase_mask handles this)
        return 1


class SafetyRules:
    """
    Detects dangerous events from AccessState history.
    """

    def __init__(
        self,
        target_loss_conf_thresh: float = 0.2,
        target_loss_frames: int = 5,
        off_axis_thresh: float = 0.4,
        no_progress_steps: int = 20,
        no_progress_scale_delta: float = 0.01,
    ):
        self.target_loss_conf_thresh = target_loss_conf_thresh
        self.target_loss_frames = target_loss_frames
        self.off_axis_thresh = off_axis_thresh
        self.no_progress_steps = no_progress_steps
        self.no_progress_scale_delta = no_progress_scale_delta

        self._low_conf_counter = 0
        self._scale_history: deque = deque(maxlen=no_progress_steps)

    def reset(self):
        self._low_conf_counter = 0
        self._scale_history.clear()

    def check(self, state: AccessState, phase: int, insert_executed: bool) -> dict:
        # Target loss: sustained low confidence
        if state.conf < self.target_loss_conf_thresh:
            self._low_conf_counter += 1
        else:
            self._low_conf_counter = 0
        target_loss = self._low_conf_counter >= self.target_loss_frames

        # Off-axis
        off_axis = (
            abs(state.e_x) > self.off_axis_thresh
            or abs(state.e_y) > self.off_axis_thresh
        )

        # No progress: scale not increasing during approach/insertion
        self._scale_history.append(state.scale)
        no_progress = False
        if phase in (1, 2) and len(self._scale_history) == self.no_progress_steps:
            scale_delta = float(np.max(self._scale_history)) - float(np.min(self._scale_history))
            no_progress = scale_delta < self.no_progress_scale_delta

        # Unsafe insert attempt
        unsafe_insert = insert_executed and not state.is_aligned()

        return {
            "target_loss": target_loss,
            "off_axis": off_axis,
            "no_progress": no_progress,
            "recovery_triggered": phase == 3,
            "unsafe_insert_attempt": unsafe_insert,
        }
