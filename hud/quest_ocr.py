"""
Quest tracker OCR — read active quest progress from the pop-out quest panel.

Unlike pollen/honey which are ALWAYS visible in the HUD, the quest tab is a
menu that must be OPENED by clicking the map icon in the top-left. Bot has
to learn to open it periodically to check progress. When it's open, we
OCR the panel and extract each quest's name + progress.

Panel layout (per user screenshot 2026-08-15):
  - Anchored on left side of screen, ~20-25% of width
  - Grouped by category header (bear name, event name, or quest chain name)
  - Each quest = descriptive text + progress bar + numeric "N/M" counter
  - Complete quests show "Complete!" instead of "N/M"

Approach:
  1. Template match on a stable visual signature of the OPEN panel (user
     snips hud/probes/quest_tab_indicator.png while panel is visible)
  2. If detected → OCR the left-side region (bounded relative to template)
  3. Parse text lines into structured {name, current, target, complete} dicts

The tab-open detector doubles as the "should reward fire" gate — quest
progress rewards only fire when we can actually SEE the current state.

If template file is missing, returns (False, []) — feature disabled but
doesn't crash. User can snip the template later without code changes.
"""

import re
from pathlib import Path
import cv2
import numpy as np

from .pollen_ocr import _get_ocr, SUFFIX_TO_MULT, parse_bss_number

TEMPLATE_PATH = Path(__file__).parent / "probes" / "quest_tab_indicator.png"
CONFIDENCE_THRESHOLD = 0.55

# When template matches, OCR this region relative to the template's top-left
# corner. Defaults assume the indicator is near the top of the panel and the
# panel extends down for ~800px. User can tune these if the panel geometry
# differs on their setup.
PANEL_OFFSET_X = -20          # extend a bit left of the template
PANEL_OFFSET_Y = -10          # extend a bit above
PANEL_EXTRA_WIDTH = 400       # width from template's left edge outward
PANEL_EXTRA_HEIGHT = 900      # panel extends this far down from template

# ---------------------------------------------------------------------------
# Alternative detection: color-based panel-open check.
#
# Template matching fails when the panel has NO stable visual elements —
# every quest text, progress bar, and category changes with the active
# quest set. But the PANEL BACKGROUND is a solid tan/beige color that's
# constant whenever the panel is open. If we sample pixels where the panel
# background would be, we can detect open-state by checking if those pixels
# look like the panel color vs the game world (grass green, sky blue, etc.).
#
# Sample points are relative to window dimensions (fraction of width/height)
# so they scale with different Roblox window sizes. Points chosen to be
# WITHIN the panel bounds AND avoid text/progress-bar rows (target the tan
# background between quest entries).
#
# Panel colors — a family of characteristic colors that appear in the quest
# panel regardless of scroll position. Single-color detection failed because
# at any FIXED pixel position, scroll shifts what's there (sometimes
# background, sometimes progress bar, sometimes header). Multi-color
# detection: sample MANY points across the panel region; if a pixel matches
# ANY of these characteristic colors, count it as a "panel pixel". Panel
# is open if enough sample points are panel pixels.
#
# Colors below were derived from user's --sample-colors runs on 2026-08-15:
# these appeared consistently across scroll positions in the panel region.
# BGR order (matches cv2 frames).
PANEL_COLORS_BGR = [
    (247, 240, 229),   # light beige — panel background
    (222, 195, 150),   # slightly darker tan — panel section bg
    (160, 136, 99),    # brownish — quest text row bg
    (85, 108, 244),    # saturated red — progress bar (uncompleted)
    (96, 255, 110),    # saturated green — progress bar (completed)
    (62, 61, 91),      # dark blue-navy — category header bg
    (53, 42, 27),      # dark brown — dividers / borders
]
PANEL_COLOR_TOLERANCE = 30              # per-color distance in BGR space
# Legacy single-color config still supported — if user set it, we ADD their
# color to the family (they know their setup best).
PANEL_BG_COLOR_BGR = (247, 240, 229)    # kept for backwards-compat
PANEL_COLOR_CONFIG = Path(__file__).parent / "probes" / "quest_panel_bg_color.txt"
# Sample points — dense grid across the left panel region. More points =
# more robust to scroll (some points will always hit background or
# characteristic colored elements). 30 points at 5 x-values × 6 y-values.
PANEL_COLOR_SAMPLE_POINTS = [(xf, yf)
    for xf in (0.02, 0.05, 0.09, 0.12, 0.15)
    for yf in (0.20, 0.32, 0.44, 0.56, 0.68, 0.80)]
# Need at least N sample points to match a panel color. With 30 points and
# a panel that occupies ~40% of the sampled area (rest is quest content
# with different colors), ~30% match rate is a natural threshold.
PANEL_MIN_MATCHING_SAMPLES = 9          # 30% of 30 samples

# Panel region for OCR when using color-based detection (no template anchor)
COLOR_PANEL_X_START_FRAC = 0.00
COLOR_PANEL_X_END_FRAC = 0.18
COLOR_PANEL_Y_START_FRAC = 0.15         # skip top HUD + buff strip
COLOR_PANEL_Y_END_FRAC = 0.90           # go down to just above the hotbar

# Parse "N/M" progress where N and M can have commas, decimals, and short-
# scale suffixes (like the pollen/honey OCR). Matches "702/1,500" or
# "24/250,000" or "1.2M / 3.5M" — same syntax as POLLEN_PAIR_RE.
PROGRESS_RE = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*([kKMmBbTtqQsSoNdU]?)"
    r"\s*/\s*"
    r"(\d[\d,]*(?:\.\d+)?)\s*([kKMmBbTtqQsSoNdU]?)"
)

# Recognize a "Complete!" state so we can flag completed quests distinctly.
# BSS uses this exact string; matching case-insensitively for OCR robustness.
COMPLETE_RE = re.compile(r"complete\s*!", re.IGNORECASE)


class QuestOCRReader:
    """Detects quest-tab-open state and OCRs active quest progress."""

    def __init__(self, template_path=TEMPLATE_PATH):
        self.template_path = template_path
        self.template = None
        self.th = self.tw = 0
        if template_path.exists():
            self.template = cv2.imread(str(template_path))
            if self.template is not None:
                self.th, self.tw = self.template.shape[:2]
        # Panel colors to check — start with the family defaults and add
        # any user-configured extra color from the legacy config file
        self.panel_colors = list(PANEL_COLORS_BGR)
        if PANEL_COLOR_CONFIG.exists():
            try:
                nums = PANEL_COLOR_CONFIG.read_text().strip().split()
                if len(nums) >= 3:
                    user_color = (int(nums[0]), int(nums[1]), int(nums[2]))
                    if user_color not in self.panel_colors:
                        self.panel_colors.append(user_color)
            except (ValueError, IOError):
                pass
        # Pre-compute numpy version for fast distance calc
        self._panel_colors_np = np.array(self.panel_colors, dtype=np.float32)

    def is_ready(self):
        """Reader is always ready — color detection needs no template.
        Template match is a bonus if the user snipped one."""
        return True

    def is_tab_open(self, frame_bgr):
        """Return (is_open, confidence, match_location).

        Two-tier detection:
        1. Template match if user snipped `quest_tab_indicator.png` AND
           it matches above threshold — most precise
        2. Color-based fallback: sample left-side pixels and check if
           they match the panel's tan background color. Robust to quest
           content changes (everything text-based moves, background stays).

        match_location is (x, y) when template-matched; None when
        color-detected (in which case downstream code should use the
        color-based panel region).
        """
        # Template path — most precise if available. Only usable if template
        # actually fits in the frame; on smaller windows (e.g., RDP session)
        # the template snipped on a bigger display can be larger than the
        # frame, and OpenCV crashes if we pass a template > image. Silently
        # fall through to color detection in that case.
        if self.template is not None:
            fh, fw = frame_bgr.shape[:2]
            if self.th <= fh and self.tw <= fw:
                result = cv2.matchTemplate(frame_bgr, self.template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                if max_val >= CONFIDENCE_THRESHOLD:
                    return True, max_val, max_loc
            else:
                # Warn once — template snipped at wrong resolution
                if not getattr(self, "_size_warned", False):
                    print(f"[quest_ocr] template ({self.tw}x{self.th}) larger "
                          f"than frame ({fw}x{fh}) — falling back to color "
                          f"detection. Re-snip quest_tab_indicator.png at "
                          f"current resolution to fix.")
                    self._size_warned = True

        # Color-based fallback — check each sample point against ALL known
        # panel colors. A pixel counts as "panel-like" if it matches ANY
        # of the panel-characteristic colors within tolerance.
        H, W = frame_bgr.shape[:2]
        matching = 0
        for xf, yf in PANEL_COLOR_SAMPLE_POINTS:
            x = int(W * xf)
            y = int(H * yf)
            if x >= W or y >= H:
                continue
            pixel = frame_bgr[y, x].astype(np.float32)
            # Distance to every panel color; if any is within tolerance, match
            dists = np.linalg.norm(self._panel_colors_np - pixel, axis=1)
            if dists.min() < PANEL_COLOR_TOLERANCE:
                matching += 1
        if matching >= PANEL_MIN_MATCHING_SAMPLES:
            # Return a synthetic "confidence" as matching/total ratio
            return True, matching / len(PANEL_COLOR_SAMPLE_POINTS), None

        return False, 0.0, None

    def sample_pixel_colors(self, frame_bgr):
        """Diagnostic — return list of (x, y, bgr_color) at each sample point.
        Use this to figure out the correct PANEL_BG_COLOR_BGR for your setup:
        open the panel, run this, look at the printed colors, pick one that
        matches the panel background."""
        H, W = frame_bgr.shape[:2]
        samples = []
        for xf, yf in PANEL_COLOR_SAMPLE_POINTS:
            x = int(W * xf)
            y = int(H * yf)
            if x < W and y < H:
                pixel = tuple(int(v) for v in frame_bgr[y, x])
                samples.append((x, y, pixel))
        return samples

    def read_quests(self, frame_bgr):
        """Return (is_open, [quest_dicts]) — OCR every visible quest.

        Each dict: {
            'raw_line': str,       # OCR'd text line that produced this entry
            'current': int|None,   # progress value (None if 'Complete!')
            'target': int|None,    # target value  (None if 'Complete!')
            'progress': float,     # 0..1 fraction (1.0 for complete)
            'complete': bool,
        }
        Returns (False, []) if tab isn't open or reader isn't ready.

        Quest NAMES aren't parsed here — they'd require line-context reasoning
        (each name is on a separate line ABOVE the progress line). The env-side
        reward tracker keys quests by their raw progress line text as a
        stable per-quest identifier, which works because BSS's progress line
        format is deterministic per quest.
        """
        is_open, conf, loc = self.is_tab_open(frame_bgr)
        if not is_open:
            return False, []

        # Compute panel bounding box — from template match if we have one,
        # else from color-region fractions
        H, W = frame_bgr.shape[:2]
        if loc is not None:
            x, y = loc
            panel_x = max(0, x + PANEL_OFFSET_X)
            panel_y = max(0, y + PANEL_OFFSET_Y)
            panel_x2 = min(W, x + self.tw + PANEL_EXTRA_WIDTH)
            panel_y2 = min(H, y + self.th + PANEL_EXTRA_HEIGHT)
        else:
            # Color-based detection — use fraction-of-window bounds
            panel_x = int(W * COLOR_PANEL_X_START_FRAC)
            panel_y = int(H * COLOR_PANEL_Y_START_FRAC)
            panel_x2 = int(W * COLOR_PANEL_X_END_FRAC)
            panel_y2 = int(H * COLOR_PANEL_Y_END_FRAC)
        panel = frame_bgr[panel_y:panel_y2, panel_x:panel_x2]

        # OCR the panel — EasyOCR handles multi-line automatically
        ocr = _get_ocr()
        results = ocr.readtext(cv2.cvtColor(panel, cv2.COLOR_BGR2RGB))
        # results = list of (bbox, text, confidence) tuples where
        # bbox is [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4 corners, panel-relative)

        # BSS quest layout: description text is on line ABOVE its progress
        # line. OCR returns each text region separately, so we have to
        # associate them ourselves via bounding-box proximity.
        # Split results into "progress" (contain N/M or Complete!) and
        # "description" (everything else).
        progress_entries = []   # list of (bbox_y_center, bbox_x_center, text, current, target, complete)
        description_entries = []  # list of (bbox_y_center, bbox_x_center, text)

        for bbox, text, _text_conf in results:
            ys = [p[1] for p in bbox]
            xs = [p[0] for p in bbox]
            y_center = (min(ys) + max(ys)) / 2
            x_center = (min(xs) + max(xs)) / 2

            m = PROGRESS_RE.search(text)
            if m:
                cur_str, cur_suffix, tgt_str, tgt_suffix = m.groups()
                current = _parse_num(cur_str, cur_suffix)
                target = _parse_num(tgt_str, tgt_suffix)
                if current is not None and target is not None and target > 0:
                    progress_entries.append((y_center, x_center, text, current, target, False))
                continue
            if COMPLETE_RE.search(text):
                progress_entries.append((y_center, x_center, text, None, None, True))
                continue
            # Not a progress line — treat as candidate description
            description_entries.append((y_center, x_center, text))

        # For each progress entry, find the description line whose y_center
        # is CLOSEST (nearby, either above or slightly below) — with
        # preference for above. Description above ties the quest name to
        # its progress bar.
        quests = []
        for py, px, ptext, current, target, is_complete in progress_entries:
            best_desc = None
            best_dist = float("inf")
            for dy, dx, dtext in description_entries:
                y_dist = py - dy   # positive if description is above progress
                # Prefer descriptions above (y_dist > 0), penalize below
                if y_dist < 0:
                    weighted = abs(y_dist) * 3
                else:
                    weighted = y_dist
                # Add small penalty for x misalignment (should be similar x range)
                weighted += abs(px - dx) * 0.5
                if weighted < best_dist:
                    best_dist = weighted
                    best_desc = dtext
            # Quest key = description text (stable identifier across reads
            # of the same quest) plus target as tiebreaker for the rare
            # case two quests share a description ('Collect 60,000 pollen'
            # for both Cactus and Rose has different desc text so they
            # don't collide even without the target).
            quest_key = best_desc.strip() if best_desc else _strip_progress(ptext)
            if not quest_key:
                # Fallback: if no description found, use target as key
                # (still allows delta tracking within a single quest)
                quest_key = f"target={target}"
            quests.append({
                "raw_line": ptext,
                "description": best_desc,
                "quest_key": quest_key,
                "current": current,
                "target": target,
                "progress": (1.0 if is_complete else
                             (min(1.0, current / target) if target else 0.0)),
                "complete": is_complete,
            })

        return True, quests


def _strip_progress(text):
    """Remove 'N/M' progress + 'Complete!' from OCR text to produce a
    stable per-quest identifier. e.g.:
      'Collect 1000 Blue Pollen. 100/1000' -> 'Collect 1000 Blue Pollen.'
      'Defeat 4 Scorpions. 0/4'            -> 'Defeat 4 Scorpions.'
      'Discover 10 Bee Types. Complete!'   -> 'Discover 10 Bee Types.'
    """
    text = PROGRESS_RE.sub("", text)
    text = COMPLETE_RE.sub("", text)
    return text.strip()


def _parse_num(num_str, suffix):
    """Parse '3,231' or '1.5M' → int. Returns None on failure."""
    if not num_str:
        return None
    try:
        base = float(num_str.replace(",", ""))
        mult = SUFFIX_TO_MULT.get(suffix, 1)
        return int(base * mult)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    import argparse
    import sys
    import mss
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from roblox_window import get_roblox_region

    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-colors", action="store_true",
                    help="Sample pixel colors at the color-detection sample "
                         "points and print them. Use this to figure out the "
                         "correct PANEL_BG_COLOR_BGR value for your setup: "
                         "open the panel in-game, run with this flag, pick "
                         "one of the printed colors, write it to "
                         "hud/probes/quest_panel_bg_color.txt as 'B G R'.")
    args = ap.parse_args()

    reader = QuestOCRReader()
    region = get_roblox_region()

    with mss.MSS() as sct:
        shot = sct.grab(region)
        frame = np.array(shot)[:, :, :3].copy()

    if args.sample_colors:
        print(f"Pixel colors at {len(PANEL_COLOR_SAMPLE_POINTS)} sample "
              f"points (BGR order):")
        print(f"MATCH column: which panel color family (if any) matched, "
              f"or '-' for game-world pixel")
        for x, y, bgr in reader.sample_pixel_colors(frame):
            pixel = np.array(bgr, dtype=np.float32)
            dists = np.linalg.norm(reader._panel_colors_np - pixel, axis=1)
            best_idx = int(dists.argmin())
            best_dist = float(dists[best_idx])
            if best_dist < PANEL_COLOR_TOLERANCE:
                match = f"panel_color[{best_idx}] {reader.panel_colors[best_idx]} (d={best_dist:.1f})"
            else:
                match = f"- (nearest {reader.panel_colors[best_idx]}, d={best_dist:.1f})"
            print(f"  ({x:4d}, {y:4d}): B={bgr[0]:3d} G={bgr[1]:3d} R={bgr[2]:3d}  {match}")
        print(f"\nPanel colors checked ({len(reader.panel_colors)}):")
        for i, c in enumerate(reader.panel_colors):
            print(f"  [{i}] BGR {c}")
        print(f"Tolerance: {PANEL_COLOR_TOLERANCE}")
        print(f"Need {PANEL_MIN_MATCHING_SAMPLES}+ matches (of "
              f"{len(PANEL_COLOR_SAMPLE_POINTS)} sample points) to consider tab open")
        sys.exit(0)

    _get_ocr()  # warm-up OCR model
    print("Reading quest tab...")
    is_open, quests = reader.read_quests(frame)
    if not is_open:
        _, conf, _ = reader.is_tab_open(frame)
        print(f"Quest tab NOT detected (template match {conf:.2f}, "
              f"threshold {CONFIDENCE_THRESHOLD}). Open the tab in-game "
              f"and re-run.")
    else:
        print(f"Quest tab OPEN, {len(quests)} quest lines detected:")
        for i, q in enumerate(quests):
            state = "COMPLETE" if q["complete"] else f"{q['current']:,}/{q['target']:,}"
            desc = q.get("description", "(no description found)")
            print(f"  [{i}] {state}  {q['progress']:>7.2%}  "
                  f"desc: {desc}  |  raw: {q['raw_line']}")
            print(f"       quest_key: {q['quest_key']}")

        # Save debug panel image for tuning — bounds depend on detection mode
        debug_dir = Path(__file__).parent / "probes"
        _, _, loc = reader.is_tab_open(frame)
        H, W = frame.shape[:2]
        if loc is not None:
            x, y = loc
            panel_x = max(0, x + PANEL_OFFSET_X)
            panel_y = max(0, y + PANEL_OFFSET_Y)
            panel_x2 = min(W, x + reader.tw + PANEL_EXTRA_WIDTH)
            panel_y2 = min(H, y + reader.th + PANEL_EXTRA_HEIGHT)
        else:
            # Color-based detection — use the same fraction bounds as read_quests
            panel_x = int(W * COLOR_PANEL_X_START_FRAC)
            panel_y = int(H * COLOR_PANEL_Y_START_FRAC)
            panel_x2 = int(W * COLOR_PANEL_X_END_FRAC)
            panel_y2 = int(H * COLOR_PANEL_Y_END_FRAC)
        panel = frame[panel_y:panel_y2, panel_x:panel_x2]
        cv2.imwrite(str(debug_dir / "debug_quest_panel.png"), panel)
        print(f"Saved debug_quest_panel.png (panel region OCR was run on)")
