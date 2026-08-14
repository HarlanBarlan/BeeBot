"""
Proof-of-life test #1: screen capture.

This waits 3 seconds so you can Alt-Tab to Roblox, then grabs the
top-left 800x600 pixels of your screen and saves them to capture.png.

Run it, then open capture.png to check that we can actually see the game.
"""

import time
import mss

print("Switching to your game window... you have 3 seconds.")
time.sleep(3)

with mss.mss() as sct:
    region = {"top": 0, "left": 0, "width": 800, "height": 600}
    img = sct.grab(region)
    mss.tools.to_png(img.rgb, img.size, output="capture.png")

print("Saved capture.png in the BeeBot folder.")
