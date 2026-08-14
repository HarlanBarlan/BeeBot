"""
The first real bot: watch for token.png, click it when seen.

Loops forever. Every CHECK_INTERVAL seconds, screencaps the Roblox
window, runs template matching. If confidence is high enough, moves
the mouse to the match center and clicks. CLICK_COOLDOWN prevents
spam-clicking the same thing.

SAFETY:
  - Press ESC to quit cleanly.
  - Ctrl+C in the terminal also works.
  - pydirectinput.FAILSAFE is on: mouse to top-left screen corner
    aborts immediately.

Tuning knobs at the top — change these and rerun.
"""

import time
import cv2
import numpy as np
import mss
import pydirectinput
import keyboard

from roblox_window import get_roblox_region
from robo_input import click_at

TEMPLATE_PATH = "token.png"
CONFIDENCE = 0.85         # 0.75-0.95 typical; higher = fewer false hits
CHECK_INTERVAL = 0.2      # seconds between screenshots
CLICK_COOLDOWN = 1.5      # min seconds between clicks
QUIT_KEY = "esc"

pydirectinput.FAILSAFE = True

template = cv2.imread(TEMPLATE_PATH)
if template is None:
    raise SystemExit(f"Couldn't load {TEMPLATE_PATH}. Run snip_template.py first.")
th, tw = template.shape[:2]

print(f"Watching for {TEMPLATE_PATH}. Press ESC to quit.")
print("Starting in 3 seconds — Alt-Tab to Roblox now.")
time.sleep(3)

last_click = 0.0
was_visible = False

with mss.MSS() as sct:
    try:
        while True:
            if keyboard.is_pressed(QUIT_KEY):
                print("ESC pressed — quitting.")
                break

            try:
                region = get_roblox_region()
            except RuntimeError as e:
                print(f"[waiting] {e}")
                time.sleep(1.0)
                continue

            shot = sct.grab(region)
            frame = np.array(shot)[:, :, :3].copy()

            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val >= CONFIDENCE:
                if not was_visible:
                    print(f"Target visible (confidence={max_val:.2f})")
                    was_visible = True

                now = time.time()
                if now - last_click >= CLICK_COOLDOWN:
                    click_x = region["left"] + max_loc[0] + tw // 2
                    click_y = region["top"] + max_loc[1] + th // 2
                    print(f"  Click at ({click_x}, {click_y})")
                    click_at(click_x, click_y)
                    last_click = now
            else:
                if was_visible:
                    print("Target gone.")
                    was_visible = False

            time.sleep(CHECK_INTERVAL)
    except KeyboardInterrupt:
        print("\nInterrupted — quitting.")
