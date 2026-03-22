"""
PhaseManager — FSM for biliary access phases.

States:
  0: alignment   — steer to center target
  1: approach    — move closer while maintaining alignment
  2: insertion   — execute guarded insertion
  3: recovery    — retreat and re-align after failure
"""
from __future__ import annotations

from src.perception.access_state import AccessState
from src.gating.insertion_gate import InsertionGate


class PhaseManager:
    """
    FSM transitions:
      0 → 1: aligned and conf > conf_thresh
      1 → 2: ready_for_insert (all gating conditions met)
      2 → 3: off_axis or no_progress or target_loss
      3 → 0: recovery complete (scope retreated)
      2 → done: success
    """

    def __init__(
        self,
        conf_thresh: float = 0.5,
        recovery_steps: int = 10,
        insertion_gate: InsertionGate | None = None,
    ):
        self.conf_thresh = conf_thresh
        self.recovery_steps = recovery_steps
        self._gate = insertion_gate or InsertionGate()
        self._phase = 0
        self._recovery_counter = 0

    @property
    def phase(self) -> int:
        return self._phase

    def reset(self):
        self._phase = 0
        self._recovery_counter = 0

    def update(self, state: AccessState, event_flags: dict) -> int:
        """
        Update FSM based on current AccessState and event flags.
        Returns new phase.
        """
        if self._phase == 0:  # alignment
            if state.is_aligned() and state.conf >= self.conf_thresh:
                self._phase = 1

        elif self._phase == 1:  # approach
            if not state.is_aligned():
                self._phase = 0  # lost alignment, go back
            elif self._gate.check(state):
                self._phase = 2

        elif self._phase == 2:  # insertion
            if (event_flags.get("off_axis") or
                    event_flags.get("no_progress") or
                    event_flags.get("target_loss")):
                self._phase = 3
                self._recovery_counter = 0

        elif self._phase == 3:  # recovery
            self._recovery_counter += 1
            if self._recovery_counter >= self.recovery_steps:
                self._phase = 0
                self._recovery_counter = 0

        return self._phase
