"""
Play back a recorded path.

Usable as a script or imported:
  from play_path import play
  play("sunflower_to_hive")     # blocks until done or ESC

CLI:
  .\.venv\Scripts\python.exe play_path.py go_home
  .\.venv\Scripts\python.exe play_path.py path.json

SAFETY:
  - Press ESC to abort.
  - On exit (normal or aborted) it releases all common keys.
"""

import sys
import time
import json
import pydirectinput
import keyboard

pydirectinput.FAILSAFE = True

STOP_KEY = "esc"
ALL_KEYS = ["w", "a", "s", "d", "space", "shift", ",", ".", "i", "o"]


def _release_all():
    for k in ALL_KEYS:
        try:
            pydirectinput.keyUp(k)
        except Exception:
            pass


def play(path):
    """Play back a recorded path. Returns True if completed, False if aborted.
    Accepts either 'name' or 'name.json'.
    """
    if not path.lower().endswith(".json"):
        path = path + ".json"
    with open(path) as f:
        events = json.load(f)
    if not events:
        return True

    start = time.time()
    try:
        for e in events:
            if keyboard.is_pressed(STOP_KEY):
                return False
            wait = (start + e["t"]) - time.time()
            if wait > 0:
                time.sleep(wait)
            if e["type"] == "down":
                pydirectinput.keyDown(e["name"])
            elif e["type"] == "up":
                pydirectinput.keyUp(e["name"])
        return True
    finally:
        _release_all()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "path"
    print(f"Playing {path}. Press {STOP_KEY.upper()} to abort.")
    print("Starting in 3 seconds...")
    time.sleep(3)
    if play(path):
        print("Playback complete.")
    else:
        print("Aborted.")
