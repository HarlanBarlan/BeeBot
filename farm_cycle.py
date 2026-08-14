"""
Full farm cycle: farm the field, detect full pollen, walk to hive,
wait for pollen to convert, walk back, resume farming. Repeat forever.

PREREQUISITES:
  - pollen_full.png template snipped (with snip_template.py pollen_full).
  - Recording of walk field->hive, saved as sunflower_to_hive.json
    (change GO_HOME_PATH below if you named it differently).
  - Recording of walk hive->field, saved as hive_to_sunflower.json
    (change GO_FIELD_PATH below).
  - Character starts standing in the field where the pattern begins,
    with the scoop equipped, cursor centered.

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
from play_path import play

pydirectinput.FAILSAFE = True

QUIT_KEY = "esc"

# --- farm tuning ------------------------------------------------------------
SWING_INTERVAL = 0.35
CYCLE_DELAY = 0.2
PATTERN = [
    ("w", 1.5),
    ("a", 0.6),
    ("s", 1.5),
    ("d", 0.6),
]

# --- vision tuning ----------------------------------------------------------
POLLEN_TEMPLATE = "pollen_full.png"
POLLEN_CONFIDENCE = 0.85
VISION_INTERVAL = 1.0  # seconds between checks

# --- hive-cycle tuning ------------------------------------------------------
GO_HOME_PATH = "sunflower_to_hive"
GO_FIELD_PATH = "hive_to_sunflower"
CONVERT_SECONDS = 5.0  # time to stand on hive pad and convert
# --------------------------------------------------------------------------

template = cv2.imread(POLLEN_TEMPLATE)
if template is None:
    raise SystemExit(f"Couldn't load {POLLEN_TEMPLATE}. Snip it first.")

_sct = mss.MSS()
_last_check = 0.0
_pollen_full = False


def poll_pollen_full():
    """Check if the pollen_full template is on screen. Rate-limited by VISION_INTERVAL."""
    global _last_check, _pollen_full
    now = time.time()
    if now - _last_check < VISION_INTERVAL:
        return _pollen_full
    _last_check = now

    try:
        region = get_roblox_region()
    except RuntimeError:
        return _pollen_full

    shot = _sct.grab(region)
    frame = np.array(shot)[:, :, :3].copy()
    result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    if max_val >= POLLEN_CONFIDENCE:
        _pollen_full = True
    return _pollen_full


def swing_scoop():
    pydirectinput.mouseDown(button="left")
    time.sleep(0.04)
    pydirectinput.mouseUp(button="left")


def hold_and_swing(key, duration):
    """Hold key while swinging. Returns 'done', 'quit', or 'full'."""
    pydirectinput.keyDown(key)
    end = time.time() + duration
    next_swing = time.time()
    try:
        while time.time() < end:
            if keyboard.is_pressed(QUIT_KEY):
                return "quit"
            if time.time() >= next_swing:
                swing_scoop()
                next_swing = time.time() + SWING_INTERVAL
            if poll_pollen_full():
                return "full"
            time.sleep(0.02)
    finally:
        pydirectinput.keyUp(key)
    return "done"


def do_hive_cycle():
    """Walk to hive, convert, walk back. Returns False if aborted."""
    print("[REACT] Pollen full — heading to hive.")
    if not play(GO_HOME_PATH):
        return False
    print(f"[REACT] Converting for {CONVERT_SECONDS}s...")
    time.sleep(CONVERT_SECONDS)
    print("[REACT] Walking back to field.")
    if not play(GO_FIELD_PATH):
        return False
    print("[REACT] Resuming farm.")
    return True


# --- main -------------------------------------------------------------------
try:
    region = get_roblox_region()
except RuntimeError as e:
    raise SystemExit(str(e))

print(f"Roblox: {region['width']}x{region['height']}")
print(f"Full farm cycle. Press {QUIT_KEY.upper()} to stop.")
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
        print(f"=== Farm cycle {cycle} ===")
        _pollen_full = False
        # Farm until pollen is full
        while not _pollen_full:
            for key, dur in PATTERN:
                status = hold_and_swing(key, dur)
                if status == "quit":
                    raise KeyboardInterrupt
                if status == "full":
                    break
            if _pollen_full:
                break
            time.sleep(CYCLE_DELAY)
        # Do the hive round trip
        if not do_hive_cycle():
            raise KeyboardInterrupt
except KeyboardInterrupt:
    print("Stopped by user.")
finally:
    _sct.close()
    for k in ["w", "a", "s", "d", "space", "shift", ",", ".", "i", "o"]:
        try:
            pydirectinput.keyUp(k)
        except Exception:
            pass
