"""
Record gameplay for imitation learning.

Every ~1/FPS seconds:
  - Screencap the Roblox window
  - Downsize to a small JPEG (saves ~10-100x disk vs. full-res PNG)
  - Poll which tracked keys and mouse buttons are currently held
  - Write a labels.jsonl line pairing the frame filename with the action

Each run creates a new folder under data/session_YYYY-MM-DD_HHMMSS/ so
you can run this many times and the sessions stay separate.

Press F8 to stop. Sessions can be as short or long as you want; more
varied data helps more than raw hours.

TIPS FOR GOOD TRAINING DATA:
  - Play NORMALLY. Don't try to be "consistent" or "optimal" — the model
    learns from what you actually do.
  - Play across multiple fields (Sunflower, Dandelion, Blue Flower).
  - Include the full loop: field walk, farming, hive walk, converting,
    checking bees, dodging mobs.
  - A few 15-minute sessions across a couple days beats one 4-hour grind.
"""

import ctypes
from ctypes import wintypes
import time
import json
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import mss
import keyboard

from common.roblox_window import get_roblox_region

_user32 = ctypes.windll.user32


def _cursor_pos():
    """Absolute screen (x, y) of the OS cursor."""
    pt = wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y

FPS = 10
FRAME_WIDTH = 480       # target width; height auto-scales to Roblox aspect
JPEG_QUALITY = 80       # 60-90 typical; higher = larger files, better quality

# Order matters — this defines the model's output layout later.
# Add or remove keys here BEFORE recording data you want to keep.
# Deliberately excluded: ctrl, alt, windows key (system shortcut safety),
# f8 (our stop key), num lock / caps lock / scroll lock / print screen
# (irrelevant to gameplay).
TRACKED_KEYS = (
    # Letters (26)
    ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
     "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    # Number row (10)
    + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # Movement/shift keys
    + ["shift", "space", "enter", "tab", "backspace", "esc"]
    # Roblox camera + common punctuation
    + [",", ".", "/", ";", "'", "[", "]", "\\", "-", "="]
    # Arrows
    + ["up", "down", "left", "right"]
    # Navigation cluster
    + ["page up", "page down", "home", "end", "insert", "delete"]
    # F-keys (F8 excluded — it's our stop key)
    + ["f1", "f2", "f3", "f4", "f5", "f6", "f7",
       "f9", "f10", "f11", "f12"]
)
STOP_KEY = "f8"


def _key_down(vk):
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def poll_action():
    """Return (keys, mouse) — both lists of 0/1."""
    keys = [1 if keyboard.is_pressed(k) else 0 for k in TRACKED_KEYS]
    mouse = [1 if _key_down(0x01) else 0,   # VK_LBUTTON
             1 if _key_down(0x02) else 0]   # VK_RBUTTON
    return keys, mouse


def main():
    session_name = datetime.now().strftime("session_%Y-%m-%d_%H%M%S")
    root = Path("data") / session_name
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_path = root / "labels.jsonl"

    try:
        region = get_roblox_region()
    except RuntimeError as e:
        raise SystemExit(str(e))

    aspect = region["height"] / region["width"]
    target_h = int(FRAME_WIDTH * aspect)
    print(f"Roblox: {region['width']}x{region['height']} -> saving {FRAME_WIDTH}x{target_h}")
    print(f"Session dir: {root}")
    print(f"Recording at {FPS} FPS. Press {STOP_KEY.upper()} to stop.")
    print("Play normally. Starting in 3 seconds...")
    time.sleep(3)

    frame_dt = 1.0 / FPS
    frame_i = 0
    start = time.time()
    next_capture = start
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    with mss.MSS() as sct, open(labels_path, "a") as labels_f:
        try:
            while True:
                if keyboard.is_pressed(STOP_KEY):
                    break
                now = time.time()
                if now < next_capture:
                    time.sleep(min(0.005, next_capture - now))
                    continue

                try:
                    region = get_roblox_region()
                except RuntimeError:
                    time.sleep(0.2)
                    next_capture = time.time()
                    continue

                shot = sct.grab(region)
                frame = np.array(shot)[:, :, :3]
                small = cv2.resize(frame, (FRAME_WIDTH, target_h))
                keys, mouse = poll_action()
                cx, cy = _cursor_pos()
                # Cursor relative to Roblox window; None,None if outside it
                if 0 <= cx - region["left"] < region["width"] and 0 <= cy - region["top"] < region["height"]:
                    rel_cx = cx - region["left"]
                    rel_cy = cy - region["top"]
                else:
                    rel_cx = rel_cy = None

                frame_i += 1
                fname = f"{frame_i:06d}.jpg"
                cv2.imwrite(str(frames_dir / fname), small, encode_params)
                labels_f.write(json.dumps({
                    "frame": fname,
                    "t": round(now - start, 4),
                    "keys": keys,
                    "mouse": mouse,
                    "cursor": [rel_cx, rel_cy],  # relative to Roblox window; null when outside
                }) + "\n")

                if frame_i % 100 == 0:
                    elapsed = now - start
                    print(f"  {frame_i} frames, {elapsed:.1f}s "
                          f"({frame_i/elapsed:.1f} FPS effective)")

                next_capture += frame_dt
                if next_capture < now - frame_dt:
                    # If we've fallen more than a frame behind, skip ahead
                    next_capture = now + frame_dt
        finally:
            elapsed = time.time() - start
            print(f"\nStopped. {frame_i} frames in {elapsed:.1f}s "
                  f"({frame_i/max(elapsed, 0.01):.1f} FPS effective)")
            with open(root / "session.json", "w") as f:
                json.dump({
                    "fps": FPS,
                    "frame_width": FRAME_WIDTH,
                    "frame_height": target_h,
                    "jpeg_quality": JPEG_QUALITY,
                    "tracked_keys": TRACKED_KEYS,
                    "mouse_buttons": ["left", "right"],
                    "frame_count": frame_i,
                    "duration_seconds": round(elapsed, 2),
                    "roblox_native_width": region["width"],
                    "roblox_native_height": region["height"],
                }, f, indent=2)


if __name__ == "__main__":
    main()
