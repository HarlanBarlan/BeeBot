"""
Unified HUD reader.

Wraps every specialized detector under one interface so bridges and the
orchestrator can query "what's the current game state?" with one call
rather than juggling six modules.

Each detector is independently loadable — missing templates just mean
that field is absent from the returned state dict. Bridges should
degrade gracefully.

Current detectors (add more as we build them):
  - pollen_bar → state["pollen_fill"] as float 0.0-1.0

Planned (stubs):
  - honey_ocr → state["honey"] as int
  - tickets_ocr → state["tickets"] as int
  - buff_bar → state["buffs"] as list[{"name","stacks","timer"}]
  - boss_bar → state["boss_hp_pct"] as float or None
  - quest_tracker → state["quests"] as list[{"text","progress"}]
"""

from .pollen_ocr import PollenOCRReader
from .honey_ocr import HoneyOCRReader


class HudReader:
    def __init__(self):
        self.pollen = PollenOCRReader()
        self.honey = HoneyOCRReader()

    def read(self, frame_bgr):
        """Return a dict of currently-detectable HUD state. Keys are only
        present if the corresponding detector succeeded on this frame."""
        state = {}

        if self.pollen.is_ready():
            fill, current, maximum, conf = self.pollen.read(frame_bgr)
            if fill is not None:
                state["pollen_fill"] = fill
                state["pollen_current"] = current
                state["pollen_max"] = maximum
                state["pollen_confidence"] = conf

        if self.honey.is_ready():
            honey, conf = self.honey.read(frame_bgr)
            if honey is not None:
                state["honey"] = honey
                state["honey_confidence"] = conf

        # future: tickets, buffs, boss, quests

        return state

    def status(self):
        """Print which detectors are loaded/missing (for setup diagnostics)."""
        print("[hud] detector status:")
        print(f"  pollen (OCR) : {'READY' if self.pollen.is_ready() else 'MISSING TEMPLATE'}")
        print(f"  honey (OCR)  : {'READY' if self.honey.is_ready() else 'MISSING TEMPLATE'}")


if __name__ == "__main__":
    reader = HudReader()
    reader.status()
