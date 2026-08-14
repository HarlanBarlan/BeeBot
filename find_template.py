"""
Look for token.png inside the Roblox window and report where it is.

Coordinates are printed both:
  - relative to the Roblox window (for reasoning about the game)
  - absolute on-screen (for pydirectinput.moveTo / click)
"""

import time
import cv2
import numpy as np
import mss

from roblox_window import get_roblox_region

TEMPLATE_PATH = "token.png"
CONFIDENCE_THRESHOLD = 0.75

template = cv2.imread(TEMPLATE_PATH)
if template is None:
    raise SystemExit(f"Couldn't load {TEMPLATE_PATH}. Run snip_template.py first.")
th, tw = template.shape[:2]

region = get_roblox_region()
print(f"Roblox window: {region['width']}x{region['height']} "
      f"at ({region['left']}, {region['top']})")
print("Searching in 3 seconds...")
time.sleep(3)

with mss.MSS() as sct:
    shot = sct.grab(region)
    frame = np.array(shot)[:, :, :3].copy()

result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(result)

x_rel, y_rel = max_loc
center_rel = (x_rel + tw // 2, y_rel + th // 2)
center_abs = (center_rel[0] + region["left"], center_rel[1] + region["top"])

print(f"Best match confidence: {max_val:.3f}")
print(f"  Center (in Roblox window): {center_rel}")
print(f"  Center (on screen):        {center_abs}")

if max_val >= CONFIDENCE_THRESHOLD:
    print("MATCH found.")
else:
    print("No confident match.")

cv2.rectangle(frame, (x_rel, y_rel), (x_rel + tw, y_rel + th), (0, 255, 0), 3)
cv2.imwrite("debug.png", frame)
print("Saved debug.png with a green box on the best match.")
