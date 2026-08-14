"""
BridgeScript base class — the pattern every scripted bridge follows.

A "bridge" is a deterministic UI/interaction sequence the model can't
reliably learn on its own (usually because it requires specific menu
clicks with sparse-reward feedback). Examples: buying an egg, hatching
a bee, activating the Wealth Clock, returning to hive when the pollen
bar is full.

Each bridge implements:
  - can_run(hud_state)     — precondition check (should we fire this?)
  - execute()              — the actual sequence
  - cooldown_seconds       — earliest we'd fire again after success

The orchestrator (bridges/orchestrator.py) polls bridges each tick,
picks the highest-priority one whose preconditions are met, executes it,
then returns control to the imitation/RL model for regular gameplay.
"""

import time
from enum import Enum


class BridgeStatus(Enum):
    OK = "ok"
    FAILED = "failed"
    ABORTED = "aborted"       # e.g. user hit ESC mid-execution
    PRECONDITION_UNMET = "precondition_unmet"


class BridgeScript:
    """Base class. Subclass and override `can_run` + `execute`."""

    name = "base_bridge"
    priority = 5                # 0=lowest, 10=highest — orchestrator uses this to break ties
    cooldown_seconds = 60       # minimum gap between successful runs

    def __init__(self):
        self.last_success_ts = 0.0
        self.last_attempt_ts = 0.0

    def _cooldown_elapsed(self):
        return time.time() - self.last_success_ts >= self.cooldown_seconds

    def can_run(self, hud_state):
        """Override. Return True if this bridge's preconditions are met.
        `hud_state` is a dict from the HUD reader (Phase 2c). For Phase 2b
        this may be a partial dict — bridges that need specific fields
        should degrade gracefully if they're missing."""
        return False

    def execute(self):
        """Override. Run the interaction sequence. Return a BridgeStatus."""
        raise NotImplementedError

    def run_if_ready(self, hud_state):
        """Wrapper: check cooldown + preconditions, execute if OK, track timing."""
        if not self._cooldown_elapsed():
            return BridgeStatus.PRECONDITION_UNMET
        if not self.can_run(hud_state):
            return BridgeStatus.PRECONDITION_UNMET
        self.last_attempt_ts = time.time()
        try:
            status = self.execute()
        except Exception as e:
            print(f"[bridge:{self.name}] exception: {e}")
            return BridgeStatus.FAILED
        if status == BridgeStatus.OK:
            self.last_success_ts = time.time()
        return status
