"""
Pollen OCR — read the "current / max" text on the pollen bar and
compute exact fill % via math.

Way more reliable than color/brightness detection: BSS shows the
numbers directly on the bar (e.g. "1.2M / 3.5M"), we just OCR them
and divide.

Approach:
  1. Template-match to locate the pollen bar region (uses the same
     pollen_bar_frame.png template you already snipped)
  2. Run EasyOCR on that region
  3. Parse the strings for "X [suffix] / Y [suffix]" pattern
  4. Return fill_fraction = X / Y

Short-scale suffix table used by BSS: k, M, B, T, q, Q, s, S, o, N, d.
"""

import re
from pathlib import Path
import cv2
import numpy as np

TEMPLATE_PATH = Path(__file__).parent / "probes" / "pollen_bar_frame.png"
CONFIDENCE_THRESHOLD = 0.55

# Short-scale suffix multipliers (BSS convention)
SUFFIX_TO_MULT = {
    "": 1,
    "k": 1e3,  "K": 1e3,
    "M": 1e6,  "m": 1e6,
    "B": 1e9,  "b": 1e9,
    "T": 1e12, "t": 1e12,
    "q": 1e15,
    "Q": 1e18,
    "s": 1e21,
    "S": 1e24,
    "o": 1e27,
    "N": 1e30,
    "d": 1e33,
    "U": 1e36,
}

# Match "1.23M" or "12k" or "500" — a number (int or decimal) + optional suffix
NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)\s*([kKMmBbTtqQsSoNdU]?)")

# Match the full BSS pollen readout: "X/Y" or "X / Y" where each side
# can be:
#   - Full form with commas as thousand separators: "3,231" or "87,500"
#   - Short-scale with optional decimal + suffix: "1.2M" or "3.5B"
# Captures: (current_number_str, current_suffix, max_number_str, max_suffix)
POLLEN_PAIR_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*([kKMmBbTtqQsSoNdU]?)"
    r"\s*/\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*([kKMmBbTtqQsSoNdU]?)"
)

# Lazy-loaded EasyOCR reader — slow first init, cached across calls
_ocr_reader = None
def _get_ocr():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        # gpu=True uses the CUDA torch install we already have
        _ocr_reader = easyocr.Reader(['en'], gpu=True, verbose=False)
    return _ocr_reader


def parse_bss_number(text):
    """'1.23M' -> 1_230_000. Returns None if no valid number found."""
    m = NUMBER_RE.search(text)
    if not m:
        return None
    num_str, suffix = m.group(1), m.group(2)
    try:
        base = float(num_str)
    except ValueError:
        return None
    mult = SUFFIX_TO_MULT.get(suffix, 1)
    return base * mult


class PollenOCRReader:
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
        """Return (fill_fraction, current, max, match_confidence) or (None, None, None, best_conf)."""
        if not self.is_ready():
            return None, None, None, 0.0
        # Guard: OpenCV crashes if template > frame. Happens on smaller-
        # resolution windows (e.g., RDP session vs main desktop).
        fh, fw = frame_bgr.shape[:2]
        if self.th > fh or self.tw > fw:
            return None, None, None, 0.0
        result = cv2.matchTemplate(frame_bgr, self.template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < CONFIDENCE_THRESHOLD:
            return None, None, None, max_val

        x, y = max_loc
        bar_region = frame_bgr[y:y + self.th, x:x + self.tw]

        ocr = _get_ocr()
        # EasyOCR wants RGB; we have BGR from cv2
        results = ocr.readtext(cv2.cvtColor(bar_region, cv2.COLOR_BGR2RGB))
        text_all = " ".join(r[1] for r in results)

        # Find the "current/max" pair — commas as thousand separators are
        # kept inside each number until we strip them below.
        m = POLLEN_PAIR_RE.search(text_all)
        if not m:
            return None, None, None, max_val

        current_raw, current_suffix, max_raw, max_suffix = m.groups()
        try:
            current = float(current_raw.replace(",", "")) * SUFFIX_TO_MULT.get(current_suffix, 1)
            maximum = float(max_raw.replace(",", "")) * SUFFIX_TO_MULT.get(max_suffix, 1)
        except ValueError:
            return None, None, None, max_val

        if maximum <= 0:
            return None, None, None, max_val

        return current / maximum, current, maximum, max_val


if __name__ == "__main__":
    import sys
    import mss
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from roblox_window import get_roblox_region

    reader = PollenOCRReader()
    if not reader.is_ready():
        raise SystemExit(f"Missing {TEMPLATE_PATH}. Snip the pollen bar first.")
    region = get_roblox_region()
    print("Initializing EasyOCR (first call downloads models, ~1 min first time)...")
    _get_ocr()
    print("Capturing screen and reading...")
    with mss.MSS() as sct:
        shot = sct.grab(region)
        frame = np.array(shot)[:, :, :3].copy()
    # For debugging: also show raw OCR text before/after normalization
    result = cv2.matchTemplate(frame, reader.template, cv2.TM_CCOEFF_NORMED)
    _, _, _, ml = cv2.minMaxLoc(result)
    x, y = ml
    bar_region = frame[y:y + reader.th, x:x + reader.tw]
    ocr_results = _get_ocr().readtext(cv2.cvtColor(bar_region, cv2.COLOR_BGR2RGB))
    text_all = " ".join(r[1] for r in ocr_results)
    print(f"OCR raw: '{text_all}'")
    m = POLLEN_PAIR_RE.search(text_all)
    if m:
        print(f"Regex match: current='{m.group(1)}{m.group(2)}'  max='{m.group(3)}{m.group(4)}'")
    else:
        print("Regex did not match the pollen pair pattern.")

    fill, cur, mx, conf = reader.read(frame)
    if fill is None:
        print(f"Could not read pollen (template match: {conf:.2f})")
    else:
        print(f"Pollen: {cur:,.0f} / {mx:,.0f} = {fill*100:.2f}%  (match {conf:.2f})")

    # Also save the region for reference
    debug_dir = Path(__file__).parent / "probes"
    result = cv2.matchTemplate(frame, reader.template, cv2.TM_CCOEFF_NORMED)
    _, _, _, ml = cv2.minMaxLoc(result)
    x, y = ml
    cv2.imwrite(str(debug_dir / "debug_ocr_region.png"),
                frame[y:y + reader.th, x:x + reader.tw])
    print(f"Saved debug_ocr_region.png so you can see what OCR is reading.")
