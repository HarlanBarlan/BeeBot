"""
Text triggers — OCR the screen periodically, match against known popup /
dialog phrases, execute scripted responses to escape.

Purpose: bot RL keeps getting stuck in quest dialogs, age verification
popups, "click to continue" prompts, etc. RL could theoretically learn to
dismiss these, but the reward is so sparse (rare occurrence + delayed
outcome) that it takes many hours. Faster to bypass with a rules table.

Rules live in text_triggers.json — add new phrases as you discover new
stuck states. Each rule specifies a match phrase and a scripted action.
"""

import json
import time
from pathlib import Path
import cv2
import pydirectinput

from .pollen_ocr import _get_ocr  # reuse the already-loaded EasyOCR instance

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from robo_input import click_at


TRIGGERS_PATH = Path(__file__).parent / "text_triggers.json"

# OCR only the CENTER region of the Roblox window — most popups appear
# there. Skipping the whole frame keeps OCR cost low.
# Values are fractions of the window size (relative to left, top).
CENTER_REGION_FRAC = {"left": 0.15, "top": 0.15, "right": 0.85, "bottom": 0.75}


class TextTriggers:
    def __init__(self, triggers_path=TRIGGERS_PATH):
        with open(triggers_path) as f:
            raw = json.load(f)
        self.rules = raw["triggers"]
        self.last_fired = {i: 0.0 for i in range(len(self.rules))}
        # OCR reader — lazily initialized (may already be loaded by pollen OCR)
        self._ocr = None

    def _ensure_ocr(self):
        if self._ocr is None:
            self._ocr = _get_ocr()

    def check(self, frame_bgr, roblox_region):
        """OCR the center of the frame, match against rules, execute first
        matching rule if its cooldown has elapsed. Returns the rule dict
        that fired (or None).

        frame_bgr: full BGR screenshot of the Roblox window (H, W, 3)
        roblox_region: dict with left/top/width/height (screen coords)
        """
        # Crop to center region
        h, w = frame_bgr.shape[:2]
        x1 = int(w * CENTER_REGION_FRAC["left"])
        y1 = int(h * CENTER_REGION_FRAC["top"])
        x2 = int(w * CENTER_REGION_FRAC["right"])
        y2 = int(h * CENTER_REGION_FRAC["bottom"])
        crop = frame_bgr[y1:y2, x1:x2]

        self._ensure_ocr()
        results = self._ocr.readtext(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        # results = list of (bbox, text, confidence)
        text_all = " ".join(r[1] for r in results).lower()
        if not text_all:
            return None

        now = time.time()
        for i, rule in enumerate(self.rules):
            phrase = rule["phrase"].lower()
            if phrase not in text_all:
                continue
            # Cooldown check — don't fire the same rule repeatedly
            cd = rule.get("cooldown_sec", 5)
            if now - self.last_fired[i] < cd:
                continue
            # Fire!
            self._execute(rule, results, x1, y1, roblox_region)
            self.last_fired[i] = now
            return rule
        return None

    def _execute(self, rule, ocr_results, crop_offset_x, crop_offset_y, roblox_region):
        action = rule.get("action", "press_key")
        params = rule.get("params", {})
        print(f"[text-triggers] matched '{rule['phrase']}' → {action}({params})")

        if action == "press_key":
            key = params.get("key", "esc")
            pydirectinput.press(key)

        elif action == "click_at":
            # Absolute pixel coords relative to Roblox window
            x_win = params.get("x", 0)
            y_win = params.get("y", 0)
            click_at(roblox_region["left"] + x_win,
                     roblox_region["top"] + y_win)

        elif action == "click_at_frac":
            # Fractional coords 0.0-1.0 relative to Roblox window — resolution-independent
            x_frac = params.get("x", 0.5)
            y_frac = params.get("y", 0.5)
            x_win = int(x_frac * roblox_region["width"])
            y_win = int(y_frac * roblox_region["height"])
            click_at(roblox_region["left"] + x_win,
                     roblox_region["top"] + y_win)

        elif action == "click_at_match":
            # Click the center of the FIRST OCR bounding box that matched the phrase
            phrase = rule["phrase"].lower()
            for bbox, text, _conf in ocr_results:
                if phrase in text.lower():
                    # bbox = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] in crop coordinates
                    xs = [p[0] for p in bbox]
                    ys = [p[1] for p in bbox]
                    cx = (min(xs) + max(xs)) / 2 + crop_offset_x
                    cy = (min(ys) + max(ys)) / 2 + crop_offset_y
                    click_at(roblox_region["left"] + int(cx),
                             roblox_region["top"] + int(cy))
                    return

        elif action == "noop":
            # Detected but nothing to do (e.g., "server closing")
            pass

        else:
            print(f"[text-triggers] unknown action '{action}' — ignoring")
