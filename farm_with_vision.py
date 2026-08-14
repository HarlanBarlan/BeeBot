"""
Field farmer + vision monitoring.

Walks the PATTERN and swings the scoop like farm_field.py, but every
VISION_INTERVAL seconds it also screencaps the Roblox window and checks
for every template listed in WATCH_LIST. When one is detected, it logs
'[SEEN] name (conf=...)'.

No reactions yet — the point of this step is proving that vision can
run alongside farming without freezing either one. Once you're happy
with what's being detected, we hook up per-template actions.

SETUP:
  1. Snip templates for things you want to watch for. Rename them:
       snip_template.py       -> creates token.png
     Then rename token.png to something meaningful, e.g. pollen_full.png,
     level_up.png, low_hp.png, rare_token.png. Repeat for each thing.
  2. List them in WATCH_LIST below.
  3. Run this script from inside the field, scoop equipped, as before.

  Press ESC to stop.
"""

import time
import cv2
import numpy as np
import mss
import pydirectinput
import keyboard

from roblox_window import get_roblox_region
from robo_input import move_mouse

pydirectinput.FAILSAFE = True

QUIT_KEY = "esc"
SWING_INTERVAL = 0.35
CYCLE_DELAY = 0.2
VISION_INTERVAL = 1.0   # seconds between vision checks
DEDUPE_WINDOW = 3.0     # don't re-log the same template for this many seconds

PATTERN = [
    ("w", 1.5),
    ("a", 0.6),
    ("s", 1.5),
    ("d", 0.6),
]

# Each entry: {"name": ..., "path": PNG file, "confidence": 0.0-1.0}
# Add more as you snip them.
WATCH_LIST = [
    {"name": "pollen_full", "path": "pollen_full.png", "confidence": 0.85},
    # {"name": "level_up",    "path": "level_up.png",    "confidence": 0.85},
    # {"name": "low_hp",      "path": "low_hp.png",      "confidence": 0.85},
]

# --- load templates ---------------------------------------------------------
templates = []
for entry in WATCH_LIST:
    img = cv2.imread(entry["path"])
    if img is None:
        print(f"[warn] couldn't load {entry['path']} — skipping")
        continue
    templates.append({**entry, "img": img, "size": img.shape[:2]})
if not templates:
    raise SystemExit("No templates loaded. Snip at least one and update WATCH_LIST.")
print(f"Loaded {len(templates)} template(s): {', '.join(t['name'] for t in templates)}")

# --- vision loop state ------------------------------------------------------
last_vision_check = 0.0
last_seen = {t["name"]: 0.0 for t in templates}
_sct = mss.MSS()


def check_vision():
    """Screencap Roblox and check all templates. Log any confident matches."""
    global last_vision_check
    now = time.time()
    if now - last_vision_check < VISION_INTERVAL:
        return
    last_vision_check = now

    try:
        region = get_roblox_region()
    except RuntimeError:
        return

    shot = _sct.grab(region)
    frame = np.array(shot)[:, :, :3].copy()

    for t in templates:
        result = cv2.matchTemplate(frame, t["img"], cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(result)
        if max_val >= t["confidence"]:
            if now - last_seen[t["name"]] >= DEDUPE_WINDOW:
                print(f"[SEEN] {t['name']} (conf={max_val:.2f})")
                last_seen[t["name"]] = now


# --- farm loop --------------------------------------------------------------
def swing_scoop():
    pydirectinput.mouseDown(button="left")
    time.sleep(0.04)
    pydirectinput.mouseUp(button="left")


def hold_and_swing(key, duration):
    pydirectinput.keyDown(key)
    end = time.time() + duration
    next_swing = time.time()
    try:
        while time.time() < end:
            if keyboard.is_pressed(QUIT_KEY):
                return False
            if time.time() >= next_swing:
                swing_scoop()
                next_swing = time.time() + SWING_INTERVAL
            check_vision()
            time.sleep(0.02)
    finally:
        pydirectinput.keyUp(key)
    return True


try:
    region = get_roblox_region()
except RuntimeError as e:
    raise SystemExit(str(e))

print(f"Roblox window: {region['width']}x{region['height']}")
print(f"Farm + vision starting. Press {QUIT_KEY.upper()} to stop.")
print("Starting in 5 seconds — Alt-Tab to Roblox now.")
for i in range(5, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

center_x = region["left"] + region["width"] // 2
center_y = region["top"] + region["height"] // 2
move_mouse(center_x, center_y)
time.sleep(0.1)

try:
    cycle = 0
    while True:
        cycle += 1
        print(f"Cycle {cycle}")
        for key, dur in PATTERN:
            if not hold_and_swing(key, dur):
                raise KeyboardInterrupt
        time.sleep(CYCLE_DELAY)
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    _sct.close()
