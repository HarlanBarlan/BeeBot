"""
Return-to-hive bridge — DISABLED (2026-08-13).

Two failed approaches:
  - Recorded path: bot's end-of-farming position is unpredictable, so
    a "field -> hive" playback only works from one exact starting spot.
  - Reset Character: LOSES ALL POLLEN in BSS. Would waste every full bag.

Per user (2026-08-13): return-to-hive is being moved to RL. Phase 3a
reward shaping — "bag full at cap = no more pollen gathered = no reward
until pollen converts" — will train the learned model to walk back to
the hive on its own when the bag fills up. No scripted navigation.

Keeping the class here so imports don't break, but can_run() always
returns False so it never fires.
"""

import time
from pathlib import Path
import pydirectinput

from .base import BridgeScript, BridgeStatus
from .ui_click import find_and_click


PROBES_DIR = Path(__file__).parent / "probes"
RESET_BUTTON = PROBES_DIR / "reset_character_button.png"
RESET_CONFIRM = PROBES_DIR / "reset_character_confirm.png"

# Fire when the bag is ~full. Not 100% — a slight buffer avoids wasted
# gathering time on the last few pollen units.
POLLEN_RETURN_THRESHOLD = 0.95

# After Reset Character, spawn to hive pad walk. Most alts have the hive
# right at spawn — 2 seconds of forward walk usually lands on the pad.
POST_RESPAWN_WAIT = 4.0
WALK_TO_PAD_SECONDS = 2.0

# Conversion wait — bees fly pollen to the hive. Real impl would poll
# pollen_fill back to ~0; for now, fixed timer.
CONVERT_WAIT_SECONDS = 15.0


class ReturnToHiveBridge(BridgeScript):
    """DISABLED — return-to-hive is handled by the RL-trained model
    (Phase 3a). See module docstring for reasoning."""
    name = "return_to_hive"
    priority = 0

    def can_run(self, hud_state):
        return False

    def execute(self):
        return BridgeStatus.FAILED
