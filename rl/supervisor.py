"""
Roblox / BSS supervisor.

Detects whether the Roblox client is running with Bee Swarm loaded, and
auto-launches / relaunches BSS if it isn't. Used by rl/env.py to keep
training going through crashes, disconnects, and Roblox client updates.

Requires:
  - Roblox client installed on the machine
  - User already logged in (Fredrick account) to the client
  - BSS accessible from this account (public game — should be)
"""

import os
import time
import subprocess
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from roblox_window import get_roblox_region


BSS_PLACE_ID = 1537690962           # public place ID from roblox.com/games/1537690962
LAUNCH_TIMEOUT_SECONDS = 180        # allow 3 min for client to load, join server, spawn
LOAD_POLL_INTERVAL_SEC = 5.0
POST_LAUNCH_GRACE_SEC = 10.0        # wait after firing URL before we start polling


def is_roblox_running():
    """Return True if the Roblox client process is running."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq RobloxPlayerBeta.exe"],
            stderr=subprocess.DEVNULL,
        ).decode(errors="ignore")
        return "RobloxPlayerBeta.exe" in out
    except Exception:
        return False


def is_bss_loaded():
    """Return True if a Roblox game window is present and reasonably sized
    (>800×600 filters out the small launcher window before you're in-game)."""
    try:
        region = get_roblox_region()
    except RuntimeError:
        return False
    return region["width"] >= 800 and region["height"] >= 600


def launch_bss():
    """Launch BSS. Non-blocking.

    Uses the browser-based launch URL rather than the raw `roblox://`
    scheme. Roblox's `roblox://placeID=...` scheme just opens the tray
    launcher without game context — the client sits idle then exits.
    The `roblox.com/games/start?placeId=...` URL hits Roblox's servers,
    generates the auth token, and hands the client a proper join
    package. This is what clicking Play in a browser does.

    Falls back to the roblox:// scheme if browser launch fails."""
    web_url = f"https://www.roblox.com/games/start?placeId={BSS_PLACE_ID}"
    scheme_urls = [
        f"roblox://experiences/start?placeId={BSS_PLACE_ID}",
        f"roblox://placeID={BSS_PLACE_ID}",
    ]

    # Prefer browser-based launch — Roblox needs the server-generated
    # auth token to actually join a game
    try:
        os.startfile(web_url)
        print(f"[supervisor] launched via browser: {web_url}")
        return True
    except Exception as e:
        print(f"[supervisor] browser launch failed: {e}")

    # Fallback: raw URL scheme (may just open tray without joining)
    for url in scheme_urls:
        try:
            os.startfile(url)
            print(f"[supervisor] launched via scheme (may not join): {url}")
            return True
        except Exception as e:
            print(f"[supervisor] scheme launch failed for {url}: {e}")
    return False


def wait_for_bss(timeout=LAUNCH_TIMEOUT_SECONDS):
    """Block until BSS is loaded or timeout. Returns True if loaded."""
    print(f"[supervisor] waiting up to {timeout:.0f}s for BSS to load…")
    time.sleep(POST_LAUNCH_GRACE_SEC)
    start = time.time()
    while time.time() - start < timeout:
        if is_bss_loaded():
            elapsed = time.time() - start
            print(f"[supervisor] BSS loaded after {elapsed:.0f}s")
            return True
        time.sleep(LOAD_POLL_INTERVAL_SEC)
    print(f"[supervisor] TIMEOUT — BSS did not load within {timeout:.0f}s")
    return False


def ensure_bss_running(max_attempts=3):
    """Full recovery: check status, launch if needed, wait for load.
    Returns True on success, False after all attempts fail."""
    for attempt in range(1, max_attempts + 1):
        if is_bss_loaded():
            return True
        print(f"[supervisor] BSS not detected (attempt {attempt}/{max_attempts})")
        if not launch_bss():
            print(f"[supervisor] launch itself failed — likely Roblox URL handler not registered")
            time.sleep(5)
            continue
        if wait_for_bss():
            # Give it a few more seconds to fully settle (spawn animation, HUD init)
            time.sleep(5)
            return True
    print(f"[supervisor] gave up after {max_attempts} attempts")
    return False


if __name__ == "__main__":
    # CLI: check state, launch if needed
    if is_bss_loaded():
        print("BSS is already loaded.")
    else:
        print("BSS is not loaded. Attempting to launch…")
        if ensure_bss_running():
            print("Successfully launched and loaded BSS.")
        else:
            print("Failed to launch BSS after multiple attempts.")
