"""
Proof-of-life test #2: keyboard control (Roblox-compatible).

We use pydirectinput instead of pyautogui because Roblox ignores
pyautogui's virtual-key events. pydirectinput sends hardware scan
codes that the Roblox client actually recognizes.

WARNING: This script will press W-A-S-D in your active window.
Make sure your Roblox character is standing somewhere safe first,
then Alt-Tab to Roblox during the 5-second countdown.

SAFETY:
  pydirectinput inherits pyautogui's FAILSAFE. If anything goes
  wrong, slam your mouse cursor into the TOP-LEFT corner of the
  screen and the script will abort immediately.
"""

import time
import pydirectinput

pydirectinput.FAILSAFE = True
# Small pause between events; pydirectinput's default (0.1s) is fine.

print("Focus your Roblox window! Starting in 5 seconds...")
for i in range(5, 0, -1):
    print(f"  {i}...")
    time.sleep(1)

# Walk forward, right, back, left — one square
for key in ["w", "d", "s", "a"]:
    print(f"Holding {key} for 1.5 seconds")
    pydirectinput.keyDown(key)
    time.sleep(1.5)
    pydirectinput.keyUp(key)

print("Done. If your character walked in a rough square, movement works.")
