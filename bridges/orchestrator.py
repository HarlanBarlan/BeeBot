"""
Orchestrator — the top-level routine planner.

Each tick:
  1. Poll all registered bridges for `can_run(hud_state)`
  2. Among runnable ones, pick the highest priority
  3. Execute it (blocks; model inference paused during script)
  4. Return control to the model until next tick

For Phase 2b this is a minimal implementation. Phase 3+ replaces the
hardcoded priority arbitration with a learned meta-policy that also
considers longer-horizon strategic goals.

Usage:
    from bridges.orchestrator import Orchestrator
    orch = Orchestrator()
    while playing:
        if orch.tick(hud_state):
            continue        # a bridge ran; skip model inference this tick
        run_model_inference(...)
"""

import time
from .base import BridgeStatus
from .wealth_clock import WealthClockBridge
from .return_to_hive import ReturnToHiveBridge
from .stubs import (BuyEggBridge, HatchBeeBridge, QuestCycleBridge,
                    MemoryMatchBridge, HiveConvertBridge)


class Orchestrator:
    def __init__(self, bridges=None, poll_interval_s=1.0):
        self.bridges = bridges or [
            ReturnToHiveBridge(),
            WealthClockBridge(),
            HiveConvertBridge(),
            MemoryMatchBridge(),
            BuyEggBridge(),
            HatchBeeBridge(),
            QuestCycleBridge(),
        ]
        self.poll_interval_s = poll_interval_s
        self.last_poll_ts = 0.0
        self.stats = {b.name: {"attempts": 0, "successes": 0, "failures": 0} for b in self.bridges}

    def tick(self, hud_state=None):
        """Check bridges, execute the best available one. Returns True if a
        bridge ran (model should skip inference this tick). False otherwise.
        Rate-limited by poll_interval_s so we don't hammer template matches."""
        now = time.time()
        if now - self.last_poll_ts < self.poll_interval_s:
            return False
        self.last_poll_ts = now

        # Find highest-priority runnable bridge
        candidates = []
        for b in self.bridges:
            if not b._cooldown_elapsed():
                continue
            try:
                if b.can_run(hud_state or {}):
                    candidates.append(b)
            except Exception as e:
                print(f"[orch] {b.name}.can_run raised: {e}")

        if not candidates:
            return False

        chosen = max(candidates, key=lambda b: b.priority)
        print(f"[orch] running bridge: {chosen.name} (priority {chosen.priority})")
        self.stats[chosen.name]["attempts"] += 1
        status = chosen.run_if_ready(hud_state or {})
        if status == BridgeStatus.OK:
            self.stats[chosen.name]["successes"] += 1
            return True
        else:
            self.stats[chosen.name]["failures"] += 1
            print(f"[orch] bridge {chosen.name} finished with {status.value}")
            return False

    def print_stats(self):
        print("[orch] session stats:")
        for name, s in self.stats.items():
            if s["attempts"]:
                print(f"  {name:20s}  {s['successes']}/{s['attempts']} ok, {s['failures']} failed")


if __name__ == "__main__":
    # Manual test: instantiate and print state
    orch = Orchestrator()
    print(f"Orchestrator loaded {len(orch.bridges)} bridges:")
    for b in orch.bridges:
        print(f"  {b.name:20s}  priority={b.priority}  cooldown={b.cooldown_seconds}s")
