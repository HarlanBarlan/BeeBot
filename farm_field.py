"""
Simple field farmer: walks a small back-and-forth pattern while
swinging the scoop on a timer.

SETUP BEFORE RUNNING:
  - Stand in the middle of a small field (Sunflower is a good first choice).
  - Equip your scoop.
  - Camera in 3rd person, looking down at the flowers.
  - Face the direction you want the pattern to start with (usually forward).

  Press ESC to stop cleanly.

WHAT IT DOES:
  Walks the PATTERN below on repeat. While holding each movement key,
  it fires a left-click every SWING_INTERVAL seconds so the scoop keeps
  swinging. That's it — no vision, no pollen-full detection, no hive
  return. Just farm forever until you stop it or something goes wrong.

TUNING:
  - Change PATTERN to shape a bigger/smaller area, or a different field.
  - SWING_INTERVAL: your scoop's swing cooldown. Faster scoops = lower.
  - CYCLE_DELAY: pause between full pattern loops.
"""

import time
import pydirectinput
import keyboard

from roblox_window import get_roblox_region
from robo_input import move_mouse

pydirectinput.FAILSAFE = True

QUIT_KEY = "esc"
SWING_INTERVAL = 0.35  # seconds between scoop swings
CYCLE_DELAY = 0.2      # brief pause between pattern cycles

# (key, seconds to hold) — one pattern loop
PATTERN = [
    ("w", 1.5),
    ("a", 0.6),
    ("s", 1.5),
    ("d", 0.6),
]


def swing_scoop():
    """Fire one left click at the current cursor position."""
    pydirectinput.mouseDown(button="left")
    time.sleep(0.04)
    pydirectinput.mouseUp(button="left")


def hold_and_swing(key, duration):
    """Hold `key` for `duration` seconds, swinging the scoop periodically.
    Returns False if the user pressed the quit key mid-hold.
    """
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
            time.sleep(0.02)
    finally:
        pydirectinput.keyUp(key)
    return True


try:
    region = get_roblox_region()
except RuntimeError as e:
    raise SystemExit(str(e))

print(f"Roblox window: {region['width']}x{region['height']}")
print(f"Field farmer starting. Press {QUIT_KEY.upper()} to stop.")
print("Starting in 5 seconds — Alt-Tab to Roblox now.")
for i in range(5, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

# Park the cursor near the center of the Roblox window so early clicks
# land in the world, not on any UI element on screen edges.
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
