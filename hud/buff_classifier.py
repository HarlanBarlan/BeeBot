"""
Buff icon classifier — detect currently-active buffs in the top-left buff strip.

Layout (per user screenshot 2026-08-15):
  - Top-left of screen, directly above the row of side-panel icons
    (egg/quest/bee/badge/settings/robux)
  - Small icons (~40-50 px) in a horizontal strip
  - Each icon has a stack count overlay like "x5", "x4"
  - Some visual fill/decay indicator shows time remaining (higher fill = more time)
  - Types include: Science Enhancement (permanent), Blue Boost, Focus, Haste,
    Red Boost, Bomb combo, and many others

Approach — same progressive-template pattern as the dialogue rescue:
  - Load ALL templates matching hud/probes/buff_*.png (glob)
  - For each frame, template-match each buff type against the strip region
  - Any template that scores above threshold → that buff is active
  - Nearby OCR extracts stack count "xN"
  - Works with 0 templates (returns empty list) OR 40 (returns full state)
  - User snips a new template whenever a new buff type appears they want tracked

For MVP, this returns a list of {name, stacks} dicts. Duration/time-remaining
is deferred — icon fill measurement is a Phase 4+ improvement (bot doesn't
yet need it to make strategic decisions).

Design principle: don't hardcode buff-specific behavior. Bot's reward
function uses aggregate "active buff count" as a proxy — the RL policy
must learn buff-specific correlations from raw observation state. Aligned
with almost-pure-RL vision.
"""

import re
from pathlib import Path
import cv2
import numpy as np

from .pollen_ocr import _get_ocr

# Directory containing per-buff templates. Filename convention:
# hud/probes/buff_<name>.png — e.g., buff_haste.png, buff_focus.png.
# The <name> portion becomes the buff's `name` field in the output.
PROBE_DIR = Path(__file__).parent / "probes"

# Confidence threshold for template match. Slightly lower than pollen/honey
# because buff icons are small and can be partially occluded by stack-count
# overlay text.
CONFIDENCE_THRESHOLD = 0.55

# Multi-scale template matching. Wiki-sourced buff icons are 120-225px but
# in-game icons render at ~40-60px. We try each template at multiple scales
# and take the best-confidence match across all scales. Slower than single-
# scale but necessary for wiki-fetched templates. If user snips their own
# templates at in-game size, the 1.0 scale will match immediately and the
# other scales still get tried but don't hurt.
TEMPLATE_SCALES = [0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.75, 1.0]

# Approximate buff strip region relative to full frame. User can override
# via a hud/probes/buff_strip_region.png template snip — if present, we
# template-match to find the strip's location dynamically. Otherwise fall
# back to these ratios (top-left corner, spanning first ~14% of screen
# width, buffs typically in y=40-100 range on a 1080p window).
STRIP_REGION_TEMPLATE = PROBE_DIR / "buff_strip_region.png"
STRIP_REGION_DEFAULT = {
    "x_start_frac": 0.0,
    "y_start_frac": 0.03,     # skip Roblox top bar
    "x_end_frac": 0.15,       # first ~15% of width
    "y_end_frac": 0.10,       # small vertical band
}

# Match "x5" or "x12" — the stack count overlay next to each buff icon.
STACK_RE = re.compile(r"[xX]\s*(\d+)")


class BuffClassifier:
    """Detects active buffs in the top-left buff strip via template matching.

    Loads one template per buff type from PROBE_DIR. Returns a list of
    {name, stacks} dicts for buffs currently detected. Robust to having
    zero templates loaded (returns empty list; observation shows no buffs).
    """

    def __init__(self, probe_dir=PROBE_DIR):
        self.probe_dir = probe_dir
        self.templates = self._load_templates()
        self.strip_anchor = None
        if STRIP_REGION_TEMPLATE.exists():
            self.strip_anchor = cv2.imread(str(STRIP_REGION_TEMPLATE))

    def _load_templates(self):
        """Return dict of {buff_name: template_image}. Filename buff_X.png
        becomes name 'X'."""
        templates = {}
        for path in sorted(self.probe_dir.glob("buff_*.png")):
            # Skip the strip-region anchor template itself
            if path.name == "buff_strip_region.png":
                continue
            name = path.stem[len("buff_"):]
            img = cv2.imread(str(path))
            if img is not None:
                templates[name] = img
        return templates

    def is_ready(self):
        """We're 'ready' as long as we could enumerate the probes directory —
        having zero buff templates just means we detect no buffs (bot's
        observation gets zeros). Not an error."""
        return True

    def _get_strip_bounds(self, frame_bgr):
        """Return (x1, y1, x2, y2) of the buff strip region in the frame.
        Prefers dynamic detection via strip_anchor template; falls back to
        fixed ratios."""
        H, W = frame_bgr.shape[:2]
        if self.strip_anchor is not None:
            result = cv2.matchTemplate(frame_bgr, self.strip_anchor, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val >= CONFIDENCE_THRESHOLD:
                ah, aw = self.strip_anchor.shape[:2]
                x, y = max_loc
                return x, y, min(W, x + aw), min(H, y + ah)
        # Fallback: fixed ratios
        x1 = int(W * STRIP_REGION_DEFAULT["x_start_frac"])
        y1 = int(H * STRIP_REGION_DEFAULT["y_start_frac"])
        x2 = int(W * STRIP_REGION_DEFAULT["x_end_frac"])
        y2 = int(H * STRIP_REGION_DEFAULT["y_end_frac"])
        return x1, y1, x2, y2

    def read_buffs(self, frame_bgr):
        """Return list of active buffs: [{'name': str, 'stacks': int}, ...].
        Empty list if no buffs detected or no templates loaded.
        """
        if not self.templates:
            return []

        x1, y1, x2, y2 = self._get_strip_bounds(frame_bgr)
        strip = frame_bgr[y1:y2, x1:x2]
        if strip.size == 0:
            return []

        found = []
        for name, template in self.templates.items():
            # Multi-scale template match — try each scale, keep the best
            # confidence + location across all of them.
            best_val = -1.0
            best_loc = None
            best_th = best_tw = 0
            for scale in TEMPLATE_SCALES:
                th_s = max(8, int(template.shape[0] * scale))
                tw_s = max(8, int(template.shape[1] * scale))
                if th_s > strip.shape[0] or tw_s > strip.shape[1]:
                    continue
                scaled = cv2.resize(template, (tw_s, th_s),
                                    interpolation=cv2.INTER_AREA
                                    if scale < 1.0 else cv2.INTER_LINEAR)
                result = cv2.matchTemplate(strip, scaled, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val > best_val:
                    best_val = max_val
                    best_loc = max_loc
                    best_th, best_tw = th_s, tw_s
            if best_val < CONFIDENCE_THRESHOLD:
                continue
            # Match found — extract stack count from region below the icon
            # (BSS displays "xN" as an overlay near the bottom-right of icon)
            mx, my = best_loc
            tw = best_tw
            th = best_th
            max_val = best_val
            # OCR region: bottom-right corner of matched icon + a bit beyond
            ocr_x1 = mx + tw // 2
            ocr_y1 = my + th // 2
            ocr_x2 = min(strip.shape[1], mx + tw + 15)
            ocr_y2 = min(strip.shape[0], my + th + 5)
            ocr_region = strip[ocr_y1:ocr_y2, ocr_x1:ocr_x2]
            stacks = 1  # default if OCR can't read stack count
            if ocr_region.size > 0:
                try:
                    ocr = _get_ocr()
                    results = ocr.readtext(cv2.cvtColor(ocr_region, cv2.COLOR_BGR2RGB))
                    text = " ".join(r[1] for r in results)
                    m = STACK_RE.search(text)
                    if m:
                        stacks = int(m.group(1))
                except Exception:
                    pass
            found.append({
                "name": name,
                "stacks": stacks,
                "confidence": float(max_val),
            })
        return found


if __name__ == "__main__":
    import sys
    import mss
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from roblox_window import get_roblox_region

    classifier = BuffClassifier()
    print(f"Loaded {len(classifier.templates)} buff template(s):")
    for name in classifier.templates:
        print(f"  buff_{name}.png")
    if not classifier.templates:
        print("(NO TEMPLATES loaded — no buffs will be detected.)")
        print(f"Snip buff templates as you encounter them:")
        print(f"  Open BSS with a buff active")
        print(f"  Run: python snip_template.py hud/probes/buff_<name>")
        print(f"  e.g., buff_haste, buff_focus, buff_rage")
        print(f"  Snip tightly around the icon graphic (not the stack count text)")

    region = get_roblox_region()
    with mss.MSS() as sct:
        shot = sct.grab(region)
        frame = np.array(shot)[:, :, :3].copy()

    # Save debug: buff strip region
    x1, y1, x2, y2 = classifier._get_strip_bounds(frame)
    strip = frame[y1:y2, x1:x2]
    debug_path = PROBE_DIR / "debug_buff_strip.png"
    cv2.imwrite(str(debug_path), strip)
    print(f"\nBuff strip region saved to {debug_path} (x=[{x1},{x2}] y=[{y1},{y2}])")
    print("If the strip location looks wrong, snip a hud/probes/buff_strip_region.png "
          "with a stable non-changing area near the buff row (e.g., a corner or the "
          "row of side buttons below the strip).")

    if classifier.templates:
        print()
        buffs = classifier.read_buffs(frame)
        if not buffs:
            print("No buffs detected (either none active or templates need adjustment).")
        else:
            print(f"{len(buffs)} buff(s) detected:")
            for b in buffs:
                print(f"  {b['name']}  x{b['stacks']}  (match {b['confidence']:.2f})")
