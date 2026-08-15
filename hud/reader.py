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
from .quest_ocr import QuestOCRReader
from .buff_classifier import BuffClassifier


class HudReader:
    def __init__(self):
        self.pollen = PollenOCRReader()
        self.honey = HoneyOCRReader()
        self.quest = QuestOCRReader()
        self.buffs = BuffClassifier()

    def read(self, frame_bgr):
        """Fast HUD read — pollen + honey (numbers displayed every frame).
        Called every HUD_READ_EVERY_N_STEPS (~1s) by env.py. Does NOT
        include quest state — quest OCR is expensive and rate-limited
        separately via read_quest()."""
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

        # future: tickets, buffs, boss

        return state

    def read_quest(self, frame_bgr):
        """Dedicated quest OCR — expensive (whole panel + many text lines).
        Env calls this at a slower cadence (QUEST_READ_EVERY_N_STEPS).
        Returns {quest_tab_open: bool, quests: list} or empty dict if
        the reader isn't ready."""
        if not self.quest.is_ready():
            return {}
        tab_open, quests = self.quest.read_quests(frame_bgr)
        return {"quest_tab_open": tab_open, "quests": quests}

    def read_buffs(self, frame_bgr):
        """Dedicated buff read — expensive (template match per buff type +
        OCR for stack counts). Env calls at a slower cadence (BUFF_READ_
        EVERY_N_STEPS). Returns {buffs: list} — empty list if no templates
        loaded or no buffs active."""
        return {"buffs": self.buffs.read_buffs(frame_bgr)}

    def status(self):
        """Print which detectors are loaded/missing (for setup diagnostics)."""
        print("[hud] detector status:")
        print(f"  pollen (OCR)  : {'READY' if self.pollen.is_ready() else 'MISSING TEMPLATE'}")
        print(f"  honey (OCR)   : {'READY' if self.honey.is_ready() else 'MISSING TEMPLATE'}")
        print(f"  quest (OCR)   : {'READY' if self.quest.is_ready() else 'MISSING TEMPLATE (snip hud/probes/quest_tab_indicator.png)'}")
        print(f"  buffs (tmpl)  : {len(self.buffs.templates)} template(s) loaded "
              f"(snip more via hud/probes/buff_<name>.png)")


if __name__ == "__main__":
    reader = HudReader()
    reader.status()
