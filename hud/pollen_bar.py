"""
Pollen bar reader — detect the bar and measure fill %.

BSS's pollen bar sits at the top-center of the screen and fills
left-to-right with amber/yellow as you gather.

Approach — pure visual, no OCR:
  1. User snips the bar + surrounding UI (include the "Pollen" label
     and icon — those are stable pixels that don't change with fill)
     -> hud/probes/pollen_bar_frame.png
  2. Each frame: template-match against captured screen to locate the region
  3. Within the matched region, count amber pixels vs total → fill %

IMPORTANT — snip TIP:
  Don't snip JUST the bar — the interior color changes as you fill up
  and template matching will fail. Include the "Pollen" label / icon /
  frame so the match anchors on stable pixels. The fill measurement
  then works correctly even though the template contains some non-bar
  area (label pixels aren't amber so they don't distort the fill %).
"""

from pathlib import Path
import cv2
import numpy as np

TEMPLATE_PATH = Path(__file__).parent / "probes" / "pollen_bar_frame.png"
# Lower than the default 0.75 — the bar's content varies with fill,
# so we tolerate some template drift as long as the surrounding UI matches.
CONFIDENCE_THRESHOLD = 0.55

# Bar fill is measured by BRIGHTNESS, not color.
# BSS pollen bar segments can be any color (amber/red/blue/white/green/orange
# depending on what pollen you carry). The one thing that's consistent:
# filled portions are BRIGHT, empty portions are DARK.
# Threshold: any grayscale pixel above BRIGHTNESS_THRESHOLD counts as "filled."
# Value picked assuming empty bar background is ~30-50 grayscale
# and any real fill is 100+ regardless of hue.
BRIGHTNESS_THRESHOLD = 80


class PollenBarReader:
    def __init__(self, template_path=TEMPLATE_PATH):
        self.template_path = template_path
        self.template = None
        self.th = self.tw = 0
        if template_path.exists():
            self.template = cv2.imread(str(template_path))
            if self.template is not None:
                self.th, self.tw = self.template.shape[:2]

    def is_ready(self):
        return self.template is not None

    def read(self, frame_bgr):
        """Return (fill_fraction, confidence) or (None, best_confidence) if not found.
        fill_fraction in [0.0, 1.0]."""
        if not self.is_ready():
            return None, 0.0
        result = cv2.matchTemplate(frame_bgr, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < CONFIDENCE_THRESHOLD:
            return None, max_val
        x, y = max_loc
        bar_region = frame_bgr[y:y + self.th, x:x + self.tw]
        # Pure brightness threshold — works regardless of what color the
        # bar happens to be right now (amber/red/blue/white/green/orange).
        gray = cv2.cvtColor(bar_region, cv2.COLOR_BGR2GRAY)
        mask = (gray > BRIGHTNESS_THRESHOLD).astype(np.uint8) * 255
        fill_fraction = float(mask.sum()) / float(mask.size * 255)
        return fill_fraction, max_val


if __name__ == "__main__":
    # CLI smoke test: capture Roblox, read pollen bar, print result.
    # Also saves debug images so you can see what the reader is looking at:
    #   hud/probes/debug_match.png    — the region the template matched (crop of the screen)
    #   hud/probes/debug_mask.png     — the amber-pixel mask (white = counted, black = not)
    #   hud/probes/debug_overlay.png  — the full screencap with a green box on the match
    import sys
    import mss
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from roblox_window import get_roblox_region

    reader = PollenBarReader()
    if not reader.is_ready():
        raise SystemExit(f"Missing {TEMPLATE_PATH}. Snip the pollen bar first.")
    region = get_roblox_region()
    with mss.MSS() as sct:
        shot = sct.grab(region)
        frame = np.array(shot)[:, :, :3].copy()

    # Template match — do it inline so we can save debug images
    result = cv2.matchTemplate(frame, reader.template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    x, y = max_loc
    print(f"Template match confidence: {max_val:.3f} at ({x}, {y})")
    bar_region = frame[y:y + reader.th, x:x + reader.tw]

    gray = cv2.cvtColor(bar_region, cv2.COLOR_BGR2GRAY)
    mask = (gray > BRIGHTNESS_THRESHOLD).astype(np.uint8) * 255
    fill_fraction = float(mask.sum()) / float(mask.size * 255)
    print(f"Fill-pixel fraction: {fill_fraction*100:.1f}% (brightness threshold {BRIGHTNESS_THRESHOLD})")

    # Save debug images
    debug_dir = Path(__file__).parent / "probes"
    cv2.imwrite(str(debug_dir / "debug_match.png"), bar_region)
    cv2.imwrite(str(debug_dir / "debug_mask.png"), mask)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + reader.tw, y + reader.th), (0, 255, 0), 3)
    cv2.imwrite(str(debug_dir / "debug_overlay.png"), overlay)
    print(f"Saved debug images to {debug_dir}/")
    print(f"  debug_match.png    — the region we're measuring")
    print(f"  debug_mask.png     — which pixels count as 'amber'")
    print(f"  debug_overlay.png  — full screen with green box on match")
