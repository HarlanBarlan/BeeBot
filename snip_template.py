r"""
Snip a template image from the Roblox window.

Usage:
  .\.venv\Scripts\python.exe snip_template.py                 -> saves token.png
  .\.venv\Scripts\python.exe snip_template.py pollen_full     -> saves pollen_full.png
  .\.venv\Scripts\python.exe snip_template.py rare_token.png  -> saves rare_token.png

  1. Have Roblox open with the thing you want to detect on screen.
  2. Run the script. It waits 3 seconds so you can Alt-Tab.
  3. Drag a rectangle around the thing to recognize.
  4. ENTER/SPACE to confirm, C to cancel.
"""

import sys
import time
import cv2
import numpy as np
import mss

from roblox_window import get_roblox_region

out_name = sys.argv[1] if len(sys.argv) > 1 else "token"
if not out_name.lower().endswith(".png"):
    out_name += ".png"

region = get_roblox_region()
print(f"Roblox window found: {region['width']}x{region['height']} "
      f"at ({region['left']}, {region['top']})")
print("Snipping in 3 seconds — Alt-Tab to Roblox now.")
time.sleep(3)

with mss.MSS() as sct:
    shot = sct.grab(region)
    frame = np.array(shot)[:, :, :3].copy()

# Shrink preview if the window is huge, so the ROI selector fits
h, w = frame.shape[:2]
scale = min(1.0, 1600 / w)
display = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame

print("Drag a rectangle. ENTER/SPACE to confirm, C to cancel.")
x, y, rw, rh = cv2.selectROI("Snip template", display, showCrosshair=True)
cv2.destroyAllWindows()

if rw == 0 or rh == 0:
    print("Cancelled — nothing saved.")
else:
    x, y, rw, rh = [int(v / scale) for v in (x, y, rw, rh)]
    crop = frame[y:y + rh, x:x + rw]
    cv2.imwrite(out_name, crop)
    print(f"Saved {out_name} ({rw}x{rh} pixels).")
