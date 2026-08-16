r"""
Record a movement path.

Captures W-A-S-D + Space + Shift + , . I O key events with timestamps
while you play. Save to a JSON file, replay with play_path.py.

Usage:
  .\.venv\Scripts\python.exe record_path.py               -> saves to path.json
  .\.venv\Scripts\python.exe record_path.py go_home       -> saves to go_home.json
  .\.venv\Scripts\python.exe record_path.py field_to_hive.json

  1. Run the script. You'll get a 3-second countdown.
  2. Alt-Tab to Roblox and walk the path you want to record.
  3. Press F8 to stop recording.

Only real up/down transitions are recorded — Windows key-repeat events
are filtered so a 2-second hold becomes one down + one up (not sixty
downs). Any keys still held when you press F8 get a synthetic "up"
appended at F8's timestamp, so playback never gets stuck holding
something forever.
"""

import sys
import time
import json
import keyboard

STOP_KEY = "f8"
# Movement keys + Roblox camera keys (, . rotate, I O zoom)
CAPTURE_KEYS = {"w", "a", "s", "d", "space", "shift", ",", ".", "i", "o"}

out_name = sys.argv[1] if len(sys.argv) > 1 else "path"
if not out_name.lower().endswith(".json"):
    out_name += ".json"

events = []
start_time = [None]
key_state = {k: False for k in CAPTURE_KEYS}  # False = up, True = down


def on_event(e):
    if e.name not in CAPTURE_KEYS:
        return
    is_down = (e.event_type == "down")
    # Ignore auto-repeat and duplicate events — only real transitions.
    if key_state[e.name] == is_down:
        return
    key_state[e.name] = is_down
    if start_time[0] is None:
        start_time[0] = e.time
    events.append({
        "t": round(e.time - start_time[0], 4),
        "name": e.name,
        "type": e.event_type,
    })


print(f"Ready to record to {out_name}.")
print("Alt-Tab to Roblox in 3 seconds, then walk your path.")
print(f"Press {STOP_KEY.upper()} when done.")
print("Captured keys: WASD, Space, Shift, , . I O (camera).")
time.sleep(3)
print("RECORDING...")

keyboard.hook(on_event)
keyboard.wait(STOP_KEY)
keyboard.unhook_all()

# Close any keys that were still held when F8 was pressed.
stop_ts = time.time()
if start_time[0] is None:
    start_time[0] = stop_ts
stop_offset = round(stop_ts - start_time[0], 4)
dangling = [k for k, held in key_state.items() if held]
for k in dangling:
    events.append({"t": stop_offset, "name": k, "type": "up"})
if dangling:
    print(f"Auto-released {len(dangling)} still-held key(s): {', '.join(dangling)}")

with open(out_name, "w") as f:
    json.dump(events, f, indent=2)

print(f"Saved {len(events)} events to {out_name}.")
if events:
    print(f"  Duration: {events[-1]['t']:.1f} seconds")
