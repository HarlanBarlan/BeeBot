"""
UI helpers — find a button by template match, click it.

Shared by any bridge that needs to click a Roblox menu or dialog
button. Uses the standard robo_input.click_at() so clicks are properly
routed through SendInput for Roblox to see.
"""

from pathlib import Path
import time
import cv2
import numpy as np
import mss

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.robo_input import click_at
from common.roblox_window import get_roblox_region


def find_and_click(template_path, confidence=0.70, wait_after=0.5, max_attempts=3):
    """Screencap Roblox, template-match a button, click its center.
    Returns True on click, False if the template wasn't found reliably
    after `max_attempts`."""
    template_path = Path(template_path)
    if not template_path.exists():
        print(f"[ui_click] missing template: {template_path}")
        return False
    template = cv2.imread(str(template_path))
    if template is None:
        print(f"[ui_click] failed to load template: {template_path}")
        return False
    th, tw = template.shape[:2]

    for attempt in range(max_attempts):
        try:
            region = get_roblox_region()
        except RuntimeError:
            time.sleep(0.5)
            continue
        with mss.MSS() as sct:
            shot = sct.grab(region)
            frame = np.array(shot)[:, :, :3].copy()
        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val >= confidence:
            x, y = max_loc
            click_x = region["left"] + x + tw // 2
            click_y = region["top"] + y + th // 2
            click_at(click_x, click_y)
            time.sleep(wait_after)
            return True
        time.sleep(0.3)

    print(f"[ui_click] {template_path.name} not found "
          f"(best confidence over {max_attempts} attempts: {max_val:.2f})")
    return False
