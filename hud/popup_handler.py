"""
Intrusive popup handler — detects and dismisses known Roblox-level modals
that block gameplay but aren't part of BSS strategy.

Examples: age confirmation dialog, Roblox "purchase confirmation" dialogs,
"leave game?" prompts, connection error notifications.

Philosophy: fully within the "session plumbing" category (like ClipCursor,
dxcam, dialogue rescue). Auto-dismissing these doesn't teach the bot HOW
to play — it removes blockers to the bot playing at all. Bot's RL policy
never sees the popup because we close it fast enough to bypass the reward
signal being affected. Consistent with the almost-pure-RL vision.

Templates in `hud/probes/`:
  popup_<name>.png         — DETECTOR: any stable visual signature of the
                             popup being visible (e.g., its title text
                             "Age Confirmation" or a border pattern)
  popup_<name>_close.png   — CLICK TARGET: the button that dismisses it
                             (e.g., the X, "Cancel", "Decline", "No")

Both files must exist for a popup type to be watched. Missing either =
that popup is skipped silently.

Add a new popup type: snip the two templates. No code changes.
"""

from pathlib import Path
import time
import cv2
import numpy as np

PROBE_DIR = Path(__file__).parent / "probes"

# Confidence threshold for detecting the popup is present. Lower than
# dialogue rescue's 0.65 because popups are visually stable across
# sessions (they're OS-level UI, not procedurally-generated game content).
DETECT_THRESHOLD = 0.65

# Confidence threshold for the close-button click target. Slightly higher
# — we're about to CLICK, don't want to click the wrong pixel.
CLICK_THRESHOLD = 0.55

# How many rapid clicks to fire on the close button. Some popups need
# multiple clicks to fully dismiss (confirmation chains, etc). 5 is
# plenty for typical single-button popups.
DISMISS_CLICK_BURST = 5


def _list_popup_pairs(probe_dir=PROBE_DIR):
    """Return list of (name, detector_path, close_path) for every popup
    type where BOTH the detector and close-button templates exist."""
    pairs = []
    for detector_path in sorted(probe_dir.glob("popup_*.png")):
        name = detector_path.stem[len("popup_"):]
        if name.endswith("_close"):
            continue   # skip close-button templates in the outer loop
        close_path = probe_dir / f"popup_{name}_close.png"
        if close_path.exists():
            pairs.append((name, detector_path, close_path))
    return pairs


class PopupHandler:
    """Detects known intrusive popups and dismisses them via template-
    matched click. Runs at env's chosen cadence (see POPUP_CHECK_EVERY_N_STEPS)."""

    def __init__(self, probe_dir=PROBE_DIR):
        self.probe_dir = probe_dir
        # Load once at init. If user snips a new template mid-session, they
        # need to restart training to pick it up (rare enough not to matter).
        self.popups = []   # list of dicts: name, detector, close, detector_shape, close_shape
        for name, detector_path, close_path in _list_popup_pairs(probe_dir):
            detector = cv2.imread(str(detector_path))
            close = cv2.imread(str(close_path))
            if detector is None or close is None:
                continue
            self.popups.append({
                "name": name,
                "detector": detector,
                "close": close,
                "detector_shape": detector.shape[:2],
                "close_shape": close.shape[:2],
            })

    def is_ready(self):
        return len(self.popups) > 0

    def check_and_dismiss(self, frame_bgr, region, click_fn):
        """Scan for each known popup. If any is detected, find its close
        button in the frame and click it (bursted). Returns the name of
        the dismissed popup, or None if no popup detected.

        `click_fn` is a callable that takes (screen_x, screen_y) and
        performs the click (env-provided so we don't couple to pydirectinput
        here directly).
        `region` is the Roblox window region dict — used to convert
        frame-relative coordinates to screen-absolute for the click.
        """
        if not self.popups:
            return None
        fh, fw = frame_bgr.shape[:2]

        for popup in self.popups:
            # 1. Detect if popup is present
            dh, dw = popup["detector_shape"]
            if dh > fh or dw > fw:
                continue   # template larger than frame — skip
            result = cv2.matchTemplate(frame_bgr, popup["detector"], cv2.TM_CCOEFF_NORMED)
            _, det_conf, _, _ = cv2.minMaxLoc(result)
            if det_conf < DETECT_THRESHOLD:
                continue

            # 2. Popup IS present. Find the close button.
            ch, cw = popup["close_shape"]
            if ch > fh or cw > fw:
                # No way to click the close button — bail
                print(f"[popup] '{popup['name']}' detected (conf {det_conf:.2f}) "
                      f"but close-button template too big for frame — skipping")
                continue
            close_result = cv2.matchTemplate(frame_bgr, popup["close"], cv2.TM_CCOEFF_NORMED)
            _, close_conf, _, close_loc = cv2.minMaxLoc(close_result)
            if close_conf < CLICK_THRESHOLD:
                print(f"[popup] '{popup['name']}' detected (conf {det_conf:.2f}) "
                      f"but couldn't locate close button (best conf {close_conf:.2f}) — "
                      f"consider re-snipping popup_{popup['name']}_close.png")
                continue

            # 3. Click the close button (screen-absolute coords)
            cx = region["left"] + close_loc[0] + cw // 2
            cy = region["top"] + close_loc[1] + ch // 2
            click_fn(cx, cy)
            print(f"[popup] dismissed '{popup['name']}' (detector conf {det_conf:.2f}, "
                  f"close-button conf {close_conf:.2f}) — clicked at ({cx}, {cy})")
            return popup["name"]

        return None


if __name__ == "__main__":
    import sys
    import mss
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from roblox_window import get_roblox_region

    handler = PopupHandler()
    print(f"Loaded {len(handler.popups)} popup handler(s):")
    for p in handler.popups:
        print(f"  {p['name']}: detector={p['detector_shape']}, close={p['close_shape']}")
    if not handler.popups:
        print("(No popup templates found. Snip pairs of hud/probes/popup_<name>.png "
              "and hud/probes/popup_<name>_close.png to enable.)")
        sys.exit(0)

    region = get_roblox_region()
    with mss.MSS() as sct:
        shot = sct.grab(region)
        frame = np.array(shot)[:, :, :3].copy()

    # Dummy click function — just prints what would be clicked
    def dry_run_click(x, y):
        print(f"  [dry-run] would click at screen ({x}, {y})")

    result = handler.check_and_dismiss(frame, region, dry_run_click)
    if result is None:
        print("\nNo popup currently detected in the frame.")
    else:
        print(f"\nWould have dismissed: {result}")
