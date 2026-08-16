"""
Find the Roblox client window and return its screen bounds.

Also makes this Python process DPI-aware so window coordinates and
screen-capture coordinates agree on high-DPI displays (Windows scaling
at 125% / 150% otherwise gives wrong pixel positions).
"""

import ctypes
import pygetwindow as gw

# Must run before any other window/screen code.
try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor DPI aware
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass


def get_roblox_region():
    """Return {top, left, width, height} for the Roblox game client window.

    Raises RuntimeError if Roblox isn't open or is minimized.
    """
    all_wins = gw.getAllWindows()

    # Prefer an exact "Roblox" title (the game client). Fall back to any
    # window with "roblox" in it but excluding Studio and empty stubs.
    exact = [w for w in all_wins
             if w.title.strip() == "Roblox" and w.width > 100 and w.height > 100]
    if exact:
        win = exact[0]
    else:
        fuzzy = [w for w in all_wins
                 if "roblox" in w.title.lower()
                 and "studio" not in w.title.lower()
                 and w.width > 100 and w.height > 100]
        if not fuzzy:
            raise RuntimeError(
                "No Roblox window found. Make sure Roblox is open, "
                "logged into a game, and not minimized."
            )
        win = fuzzy[0]

    if win.isMinimized:
        raise RuntimeError("Roblox window is minimized. Restore it and try again.")

    return {"top": win.top, "left": win.left, "width": win.width, "height": win.height}
