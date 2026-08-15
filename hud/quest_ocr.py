"""
Quest tracker OCR — read active quest progress from the pop-out quest panel.

Unlike pollen/honey which are ALWAYS visible in the HUD, the quest tab is a
menu that must be OPENED by clicking the map icon in the top-left. Bot has
to learn to open it periodically to check progress. When it's open, we
OCR the panel and extract each quest's name + progress.

Panel layout (per user screenshot 2026-08-15):
  - Anchored on left side of screen, ~20-25% of width
  - Grouped by category header (bear name, event name, or quest chain name)
  - Each quest = descriptive text + progress bar + numeric "N/M" counter
  - Complete quests show "Complete!" instead of "N/M"

Approach:
  1. Template match on a stable visual signature of the OPEN panel (user
     snips hud/probes/quest_tab_indicator.png while panel is visible)
  2. If detected → OCR the left-side region (bounded relative to template)
  3. Parse text lines into structured {name, current, target, complete} dicts

The tab-open detector doubles as the "should reward fire" gate — quest
progress rewards only fire when we can actually SEE the current state.

If template file is missing, returns (False, []) — feature disabled but
doesn't crash. User can snip the template later without code changes.
"""

import re
from pathlib import Path
import cv2
import numpy as np

from .pollen_ocr import _get_ocr, SUFFIX_TO_MULT, parse_bss_number

TEMPLATE_PATH = Path(__file__).parent / "probes" / "quest_tab_indicator.png"
CONFIDENCE_THRESHOLD = 0.55

# When template matches, OCR this region relative to the template's top-left
# corner. Defaults assume the indicator is near the top of the panel and the
# panel extends down for ~800px. User can tune these if the panel geometry
# differs on their setup.
PANEL_OFFSET_X = -20          # extend a bit left of the template
PANEL_OFFSET_Y = -10          # extend a bit above
PANEL_EXTRA_WIDTH = 400       # width from template's left edge outward
PANEL_EXTRA_HEIGHT = 900      # panel extends this far down from template

# Parse "N/M" progress where N and M can have commas, decimals, and short-
# scale suffixes (like the pollen/honey OCR). Matches "702/1,500" or
# "24/250,000" or "1.2M / 3.5M" — same syntax as POLLEN_PAIR_RE.
PROGRESS_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*([kKMmBbTtqQsSoNdU]?)"
    r"\s*/\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*([kKMmBbTtqQsSoNdU]?)"
)

# Recognize a "Complete!" state so we can flag completed quests distinctly.
# BSS uses this exact string; matching case-insensitively for OCR robustness.
COMPLETE_RE = re.compile(r"complete\s*!", re.IGNORECASE)


class QuestOCRReader:
    """Detects quest-tab-open state and OCRs active quest progress."""

    def __init__(self, template_path=TEMPLATE_PATH):
        self.template_path = template_path
        self.template = None
        self.th = self.tw = 0
        if template_path.exists():
            self.template = cv2.imread(str(template_path))
            if self.template is not None:
                self.th, self.tw = self.template.shape[:2]

    def is_ready(self):
        return self.template is not None

    def is_tab_open(self, frame_bgr):
        """Return (is_open, confidence, match_location). match_location is
        (x, y) of template's top-left when found, None otherwise."""
        if not self.is_ready():
            return False, 0.0, None
        result = cv2.matchTemplate(frame_bgr, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < CONFIDENCE_THRESHOLD:
            return False, max_val, None
        return True, max_val, max_loc

    def read_quests(self, frame_bgr):
        """Return (is_open, [quest_dicts]) — OCR every visible quest.

        Each dict: {
            'raw_line': str,       # OCR'd text line that produced this entry
            'current': int|None,   # progress value (None if 'Complete!')
            'target': int|None,    # target value  (None if 'Complete!')
            'progress': float,     # 0..1 fraction (1.0 for complete)
            'complete': bool,
        }
        Returns (False, []) if tab isn't open or reader isn't ready.

        Quest NAMES aren't parsed here — they'd require line-context reasoning
        (each name is on a separate line ABOVE the progress line). The env-side
        reward tracker keys quests by their raw progress line text as a
        stable per-quest identifier, which works because BSS's progress line
        format is deterministic per quest.
        """
        is_open, conf, loc = self.is_tab_open(frame_bgr)
        if not is_open:
            return False, []

        # Compute panel bounding box relative to template match
        H, W = frame_bgr.shape[:2]
        x, y = loc
        panel_x = max(0, x + PANEL_OFFSET_X)
        panel_y = max(0, y + PANEL_OFFSET_Y)
        panel_x2 = min(W, x + self.tw + PANEL_EXTRA_WIDTH)
        panel_y2 = min(H, y + self.th + PANEL_EXTRA_HEIGHT)
        panel = frame_bgr[panel_y:panel_y2, panel_x:panel_x2]

        # OCR the panel — EasyOCR handles multi-line automatically
        ocr = _get_ocr()
        results = ocr.readtext(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        # results = list of (bbox, text, confidence) tuples

        quests = []
        for _bbox, text, _text_conf in results:
            # Try matching progress pattern first (most common)
            m = PROGRESS_RE.search(text)
            if m:
                cur_str, cur_suffix, tgt_str, tgt_suffix = m.groups()
                current = _parse_num(cur_str, cur_suffix)
                target = _parse_num(tgt_str, tgt_suffix)
                if current is not None and target is not None and target > 0:
                    # Quest key = raw text with the progress "N/M" stripped
                    # out. This is what identifies the SAME quest across
                    # reads even as its progress changes. Without this
                    # stripping, reward tracker would see every progress
                    # update as a new quest and never fire progress reward.
                    quest_key = _strip_progress(text)
                    quests.append({
                        "raw_line": text,
                        "quest_key": quest_key,
                        "current": current,
                        "target": target,
                        "progress": min(1.0, current / target),
                        "complete": False,
                    })
                continue

            # Check for "Complete!" marker
            if COMPLETE_RE.search(text):
                quest_key = _strip_progress(text)
                quests.append({
                    "raw_line": text,
                    "quest_key": quest_key,
                    "current": None,
                    "target": None,
                    "progress": 1.0,
                    "complete": True,
                })

        return True, quests


def _strip_progress(text):
    """Remove 'N/M' progress + 'Complete!' from OCR text to produce a
    stable per-quest identifier. e.g.:
      'Collect 1000 Blue Pollen. 100/1000' -> 'Collect 1000 Blue Pollen.'
      'Defeat 4 Scorpions. 0/4'            -> 'Defeat 4 Scorpions.'
      'Discover 10 Bee Types. Complete!'   -> 'Discover 10 Bee Types.'
    """
    text = PROGRESS_RE.sub("", text)
    text = COMPLETE_RE.sub("", text)
    return text.strip()


def _parse_num(num_str, suffix):
    """Parse '3,231' or '1.5M' → int. Returns None on failure."""
    if not num_str:
        return None
    try:
        base = float(num_str.replace(",", ""))
        mult = SUFFIX_TO_MULT.get(suffix, 1)
        return int(base * mult)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    import sys
    import mss
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from roblox_window import get_roblox_region

    reader = QuestOCRReader()
    if not reader.is_ready():
        raise SystemExit(
            f"Missing {TEMPLATE_PATH}. Snip a template first:\n"
            f"  Open the quest tab in-game (click the map icon top-left)\n"
            f"  Run: python snip_template.py hud/probes/quest_tab_indicator\n"
            f"  Snip a stable visual element that's ONLY visible when the\n"
            f"  quest tab is open (e.g., a header text like 'Quests' or a\n"
            f"  category background pattern)."
        )
    region = get_roblox_region()
    _get_ocr()  # warm-up OCR model

    print("Reading quest tab...")
    with mss.MSS() as sct:
        shot = sct.grab(region)
        frame = np.array(shot)[:, :, :3].copy()

    is_open, quests = reader.read_quests(frame)
    if not is_open:
        _, conf, _ = reader.is_tab_open(frame)
        print(f"Quest tab NOT detected (template match {conf:.2f}, "
              f"threshold {CONFIDENCE_THRESHOLD}). Open the tab in-game "
              f"and re-run.")
    else:
        print(f"Quest tab OPEN, {len(quests)} quest lines detected:")
        for i, q in enumerate(quests):
            state = "COMPLETE" if q["complete"] else f"{q['current']:,}/{q['target']:,}"
            print(f"  [{i}] {state}  |  {q['raw_line']}  |  "
                  f"progress={q['progress']:.2%}")

        # Save debug panel image for tuning
        debug_dir = Path(__file__).parent / "probes"
        _, _, loc = reader.is_tab_open(frame)
        x, y = loc
        H, W = frame.shape[:2]
        panel_x = max(0, x + PANEL_OFFSET_X)
        panel_y = max(0, y + PANEL_OFFSET_Y)
        panel_x2 = min(W, x + reader.tw + PANEL_EXTRA_WIDTH)
        panel_y2 = min(H, y + reader.th + PANEL_EXTRA_HEIGHT)
        panel = frame[panel_y:panel_y2, panel_x:panel_x2]
        cv2.imwrite(str(debug_dir / "debug_quest_panel.png"), panel)
        print(f"Saved debug_quest_panel.png (panel region OCR was run on)")
