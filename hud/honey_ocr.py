"""
Honey OCR — read the current honey balance from the top-of-screen HUD.

Same pattern as pollen_ocr: user snips the honey display region once,
template-matches to locate each frame, EasyOCR extracts the number,
regex + suffix table converts to a real int.

Honey is displayed as a single number (no "current/max" — just a total)
in either short-scale form ("12.5M", "1.7B") or long form with commas
("12,500,000"). We handle both.
"""

import re
from pathlib import Path
import cv2
import numpy as np

# Reuse the suffix table + OCR instance from pollen_ocr
from .pollen_ocr import SUFFIX_TO_MULT, _get_ocr

TEMPLATE_PATH = Path(__file__).parent / "probes" / "honey_display.png"
CONFIDENCE_THRESHOLD = 0.55

# Match a single number (with commas + optional decimal + optional suffix).
# Anchored to a digit start so labels like "Honey" don't accidentally match.
HONEY_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*([kKMmBbTtqQsSoNdU]?)")


class HoneyOCRReader:
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
        """Return (honey_value, match_confidence) or (None, best_conf)."""
        if not self.is_ready():
            return None, 0.0
        result = cv2.matchTemplate(frame_bgr, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < CONFIDENCE_THRESHOLD:
            return None, max_val

        x, y = max_loc
        honey_region = frame_bgr[y:y + self.th, x:x + self.tw]

        ocr = _get_ocr()
        results = ocr.readtext(cv2.cvtColor(honey_region, cv2.COLOR_BGR2RGB))
        text_all = " ".join(r[1] for r in results)

        m = HONEY_RE.search(text_all)
        if not m:
            return None, max_val

        num_str, suffix = m.groups()
        try:
            honey = float(num_str.replace(",", "")) * SUFFIX_TO_MULT.get(suffix, 1)
        except ValueError:
            return None, max_val

        return honey, max_val


if __name__ == "__main__":
    import sys
    import mss
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from roblox_window import get_roblox_region

    reader = HoneyOCRReader()
    if not reader.is_ready():
        raise SystemExit(f"Missing {TEMPLATE_PATH}. Snip the honey display first.")
    region = get_roblox_region()
    print("Reading honey...")
    _get_ocr()  # trigger any first-time model load
    with mss.MSS() as sct:
        shot = sct.grab(region)
        frame = np.array(shot)[:, :, :3].copy()

    # Debug: show raw OCR text before parsing
    result = cv2.matchTemplate(frame, reader.template, cv2.TM_CCOEFF_NORMED)
    _, _, _, ml = cv2.minMaxLoc(result)
    x, y = ml
    honey_region = frame[y:y + reader.th, x:x + reader.tw]
    ocr_results = _get_ocr().readtext(cv2.cvtColor(honey_region, cv2.COLOR_BGR2RGB))
    text_all = " ".join(r[1] for r in ocr_results)
    print(f"OCR raw: '{text_all}'")

    honey, conf = reader.read(frame)
    if honey is None:
        print(f"Could not read honey (template match: {conf:.2f})")
    else:
        print(f"Honey: {honey:,.0f}  (match {conf:.2f})")

    debug_dir = Path(__file__).parent / "probes"
    cv2.imwrite(str(debug_dir / "debug_honey_region.png"), honey_region)
    print(f"Saved debug_honey_region.png")
