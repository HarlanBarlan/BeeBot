"""
Roblox-friendly input helpers.

Why this exists:
  Roblox uses raw mouse input to track the cursor. It ignores plain
  SetCursorPos "teleports" — the cursor visually moves but Roblox's
  internal hover state doesn't update, so the next click still lands
  at whatever position Roblox last thought the cursor was.

  The fix is to use SendInput with MOUSEEVENTF_MOVE + MOUSEEVENTF_ABSOLUTE
  + MOUSEEVENTF_VIRTUALDESK. This both positions the cursor AND fires a
  real motion event that raw-input listeners like Roblox will see.

  For clicks we still use pydirectinput's mouseDown/mouseUp because those
  send proper DirectInput-style button events Roblox accepts. We add small
  pauses between move / down / up so Roblox has time to react to each.
"""

import ctypes
from ctypes import wintypes
import time
import pydirectinput

pydirectinput.FAILSAFE = True

POST_MOVE_DELAY = 0.05   # give Roblox time to update hover state
CLICK_HOLD_DELAY = 0.05  # click duration


# --- SendInput plumbing -----------------------------------------------------

INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class _INPUT_UNION(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT)]


class INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _INPUT_UNION)]


def _virtual_desktop_bounds():
    gm = ctypes.windll.user32.GetSystemMetrics
    return gm(SM_XVIRTUALSCREEN), gm(SM_YVIRTUALSCREEN), gm(SM_CXVIRTUALSCREEN), gm(SM_CYVIRTUALSCREEN)


def move_mouse(x, y):
    """Move cursor to absolute screen (x, y) via SendInput.
    Generates a real motion event Roblox will register.
    """
    vx, vy, vw, vh = _virtual_desktop_bounds()
    # Normalize to 0..65535 across the virtual desktop
    nx = int((x - vx) * 65535 / max(vw - 1, 1))
    ny = int((y - vy) * 65535 / max(vh - 1, 1))
    mi = MOUSEINPUT(
        dx=nx, dy=ny, mouseData=0,
        dwFlags=MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
        time=0, dwExtraInfo=None,
    )
    inp = INPUT(type=INPUT_MOUSE)
    inp.mi = mi
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def click_at(x, y):
    """Move to (x, y), pause, then perform a real left click."""
    move_mouse(x, y)
    time.sleep(POST_MOVE_DELAY)
    pydirectinput.mouseDown(button="left")
    time.sleep(CLICK_HOLD_DELAY)
    pydirectinput.mouseUp(button="left")


# --- camera control ---------------------------------------------------------
# In Roblox you look around either by (a) holding shift-lock (Shift key,
# WASD then moves relative to camera) or (b) holding right mouse button
# and dragging. These helpers support both — the model picks by outputting
# the appropriate button state + cursor motion.

def right_mouse_down():
    pydirectinput.mouseDown(button="right")


def right_mouse_up():
    pydirectinput.mouseUp(button="right")


def drag_mouse(from_x, from_y, to_x, to_y, steps=10, step_delay=0.01):
    """Move cursor from (from_x, from_y) to (to_x, to_y) in N steps.
    Generates real motion events so Roblox raw-input listeners see the
    drag. Use with right_mouse_down/up around the call to rotate camera.
    """
    for i in range(1, steps + 1):
        t = i / steps
        x = int(from_x + (to_x - from_x) * t)
        y = int(from_y + (to_y - from_y) * t)
        move_mouse(x, y)
        time.sleep(step_delay)


def rotate_camera(from_xy, dx, dy, steps=10):
    """Convenience: right-click-drag from a starting cursor position
    by (dx, dy) pixels. Used to rotate camera without shift lock.
    """
    fx, fy = from_xy
    right_mouse_down()
    try:
        drag_mouse(fx, fy, fx + dx, fy + dy, steps=steps)
    finally:
        right_mouse_up()
