"""
Wealth Clock bridge — once per hour of playtime, interact for +1% HPP
stack (max 5) and 1-5 tickets.

Requires:
  - Recorded path 'to_wealth_clock.json' (hive area -> wealth clock)
  - Recorded path 'from_wealth_clock.json' (wealth clock -> back to hive/field)
  - (optional) template 'wealth_clock_ui.png' snipped when you're near it,
    to confirm we're in position before pressing E
"""

import time
from pathlib import Path
import pydirectinput

from .base import BridgeScript, BridgeStatus

# Uses the play_path helper from our earlier phase
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from play_path import play as play_recorded_path


PATHS_DIR = Path(__file__).parent / "paths"


class WealthClockBridge(BridgeScript):
    name = "wealth_clock"
    priority = 9                        # high — big HPP value, easy pickup
    cooldown_seconds = 3600             # once per hour of real playtime

    def can_run(self, hud_state):
        # For Phase 2b: no HUD reader yet, so we fire on cooldown alone.
        # Once HUD reader exists (Phase 2c), require: honey/tickets showing
        # (means we're not in a menu), bee count >= 5, and we haven't just
        # spent the Wealth Clock (detect the +HPP buff icon).
        return True

    def execute(self):
        to_path = PATHS_DIR / "to_wealth_clock.json"
        back_path = PATHS_DIR / "from_wealth_clock.json"

        if not to_path.exists() or not back_path.exists():
            print(f"[bridge:{self.name}] missing recorded paths in {PATHS_DIR}. "
                  f"Record with: record_path.py to_wealth_clock")
            return BridgeStatus.FAILED

        print(f"[bridge:{self.name}] walking to Wealth Clock")
        if not play_recorded_path(str(to_path)):
            return BridgeStatus.ABORTED

        # Wait a beat, press E, wait for interaction to resolve
        time.sleep(0.5)
        print(f"[bridge:{self.name}] pressing E")
        pydirectinput.press("e")
        time.sleep(2.5)

        print(f"[bridge:{self.name}] returning to base")
        if not play_recorded_path(str(back_path)):
            return BridgeStatus.ABORTED

        return BridgeStatus.OK
