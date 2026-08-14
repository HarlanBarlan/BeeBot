# BeeBot

A hobby project to build a hybrid computer-vision bot that plays Bee Swarm Simulator.

## Setup (already done)

- Python 3.14 via `py` launcher
- Virtual environment at `.venv\`
- Libraries: opencv-python, pyautogui, pydirectinput, mss, numpy, pillow, keyboard

**Roblox input note:** pyautogui's key events don't register in Roblox — use `pydirectinput` (drop-in replacement API). Symptom of using the wrong one: keys type into text fields but the character doesn't move.

## How to run a script

Open PowerShell in this folder and use the venv's Python directly:

```
.\.venv\Scripts\python.exe hello.py
```

Or activate the venv once per terminal session so plain `python` works:

```
.\.venv\Scripts\Activate.ps1
python hello.py
```

If activation is blocked, run this once (as your normal user):

```
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

## Scripts so far

- `hello.py` — sanity check, prints a message.
- `test_capture.py` — waits 3s, screenshots the top-left 800x600 of your screen to `capture.png`.
- `test_move.py` — waits 3s, then presses W-A-S-D for 1s each. **Panic button:** slam mouse to top-left corner of screen to abort.

## Roadmap

1. Prove capture + movement (these first two scripts).
2. Screenshot a Bee Swarm token, save as `token.png`, use OpenCV template matching to detect it on screen.
3. Timer-based farm loop: swing scoop N times, then walk back to hive.
4. Add visual reactions: detect low HP tint, detect monster spawn, run away.
