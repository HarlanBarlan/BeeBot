r"""
Live inference — run the trained BeeBot model against Roblox.

Loop: screencap Roblox window -> CNN forward pass -> decode to keys/mouse/cursor
-> issue only transitions (avoid spamming re-presses of already-held keys).

Usage:
  .\.venv\Scripts\python.exe play.py                          # default: best.pt, temp=1.0
  .\.venv\Scripts\python.exe play.py --ckpt models/beebot_latest.pt
  .\.venv\Scripts\python.exe play.py --temperature 0.7        # more decisive
  .\.venv\Scripts\python.exe play.py --temperature 1.5        # more curious/exploratory
  .\.venv\Scripts\python.exe play.py --temperature 0          # pure argmax (deterministic)

SAFETY:
  - Press ESC any time to stop cleanly. All held keys/buttons are released.
  - pydirectinput FAILSAFE on: slam the mouse to top-left corner to abort.
  - Cursor moves are filtered by --cursor_min_delta so tiny jitter every
    frame doesn't flood the input pipeline.

CURIOSITY:
  - Stochastic sampling means the bot samples from the model's action
    distribution rather than always argmax. Higher temperature = more
    exploration. Lower = more deterministic. Serves the vision goal of
    "not stubborn, tries new things."
"""

import time
import argparse
from pathlib import Path
import numpy as np
import cv2
import mss
import torch
import pydirectinput
import keyboard

from common.roblox_window import get_roblox_region
from common.robo_input import move_mouse
from .model import BeeBotCNN, BeeBotLSTM
from .dataset import MODEL_INPUT_W, MODEL_INPUT_H, GAME_RELEVANT_KEYS
from hud.reader import HudReader
from bridges.orchestrator import Orchestrator

pydirectinput.FAILSAFE = True

QUIT_KEY = "esc"
STATUS_EVERY_N_FRAMES = 50

# GAME_RELEVANT_KEYS is imported from dataset.py — see the definition there.
# It's the same whitelist used during training to mask non-game keys, kept
# as an inference-time safety net in case a checkpoint was trained without
# the mask.


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    tracked_keys = ckpt["tracked_keys"]
    model_type = ckpt.get("model_type", "BeeBotCNN")
    if model_type == "BeeBotLSTM":
        hidden = ckpt.get("hidden_size", 256)
        model = BeeBotLSTM(n_keys=len(tracked_keys), hidden_size=hidden)
    else:
        model = BeeBotCNN(n_keys=len(tracked_keys))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, tracked_keys, model_type


def prepare_frame(bgra_np):
    """mss returns BGRA (H,W,4). Convert to model input tensor (1,3,H,W) float32 [0,1]."""
    rgb = cv2.cvtColor(bgra_np[:, :, :3], cv2.COLOR_BGR2RGB)
    small = cv2.resize(rgb, (MODEL_INPUT_W, MODEL_INPUT_H))
    small = small.astype(np.float32) / 255.0
    small = np.transpose(small, (2, 0, 1))
    return torch.from_numpy(small).unsqueeze(0)


def sample_binary(logits, temperature):
    """Per-element sampling. temperature=0 = argmax (deterministic), 1.0 = model dist,
    >1.0 = more exploratory (probs pulled toward 0.5).
    Returns numpy bool array of same shape as logits.
    """
    if temperature <= 0:
        return (logits > 0).cpu().numpy()
    scaled = logits / temperature
    probs = torch.sigmoid(scaled)
    rands = torch.rand_like(probs)
    return (rands < probs).cpu().numpy()


def release_all(held_keys, held_mouse):
    for k in list(held_keys):
        try: pydirectinput.keyUp(k)
        except Exception: pass
    for mb in list(held_mouse):
        try: pydirectinput.mouseUp(button=mb)
        except Exception: pass
    held_keys.clear()
    held_mouse.clear()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="models/beebot_best.pt")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="0=argmax, 1.0=model dist, >1.0=curious")
    p.add_argument("--fps", type=float, default=10.0,
                   help="Inference tick rate. Match training FPS (10) unless testing.")
    p.add_argument("--cursor_min_delta", type=int, default=8,
                   help="Ignore cursor moves smaller than this many pixels total.")
    p.add_argument("--dry_run", action="store_true",
                   help="Print decisions but don't send inputs. For debugging.")
    p.add_argument("--no_bridges", action="store_true",
                   help="Disable orchestrator/bridge scripts. Model runs alone.")
    p.add_argument("--hud_interval", type=float, default=0.5,
                   help="How often (seconds) to read the HUD. OCR is expensive.")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[play] device: {device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise SystemExit(f"Missing checkpoint: {ckpt_path}. Train first with train.py or train_lstm.py.")
    model, tracked_keys, model_type = load_model(ckpt_path, device)
    print(f"[play] loaded {ckpt_path} ({model_type}, {len(tracked_keys)} tracked keys)")
    is_lstm = (model_type == "BeeBotLSTM")
    lstm_hidden = model.init_hidden(batch_size=1, device=device) if is_lstm else None

    # HUD reader + orchestrator (bridges)
    hud = HudReader()
    hud.status()
    orch = None if args.no_bridges else Orchestrator()
    hud_state = {}
    last_hud_read = 0.0

    try:
        region = get_roblox_region()
    except RuntimeError as e:
        raise SystemExit(str(e))
    print(f"[play] Roblox: {region['width']}x{region['height']} at ({region['left']},{region['top']})")
    print(f"[play] temperature={args.temperature}  fps={args.fps}  dry_run={args.dry_run}")
    print(f"[play] Focus Roblox — starting in 5 seconds. Press ESC to stop.")
    for i in range(5, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    tick = 1.0 / args.fps
    held_keys = set()
    held_mouse = set()
    last_cursor = None
    frame_i = 0
    started = time.time()

    with mss.MSS() as sct:
        try:
            next_tick = time.time()
            while True:
                if keyboard.is_pressed(QUIT_KEY):
                    print("[play] ESC pressed — stopping")
                    break

                now = time.time()
                if now < next_tick:
                    time.sleep(min(0.003, next_tick - now))
                    continue

                try:
                    region = get_roblox_region()
                except RuntimeError:
                    time.sleep(0.5)
                    next_tick = time.time()
                    continue

                shot = sct.grab(region)
                frame = np.array(shot)

                # Rate-limited HUD reads (OCR is ~50-100ms per call)
                if time.time() - last_hud_read >= args.hud_interval:
                    frame_bgr = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_RGB2BGR) if frame.shape[2] == 3 else frame[:, :, :3][:, :, ::-1].copy()
                    # mss returns BGRA, so channel[:3] is BGR already
                    new_state = hud.read(frame[:, :, :3].copy())
                    if new_state:
                        hud_state.update(new_state)
                    last_hud_read = time.time()

                # Let the orchestrator run bridges if any preconditions are met.
                # If a bridge fires, the character moves/interacts via scripts,
                # and the LSTM hidden state becomes stale — reset it.
                if orch is not None and not args.dry_run:
                    if orch.tick(hud_state):
                        if is_lstm:
                            lstm_hidden = model.init_hidden(batch_size=1, device=device)
                        continue  # skip model inference this tick

                tensor = prepare_frame(frame).to(device)

                with torch.no_grad():
                    if is_lstm:
                        out, lstm_hidden = model(tensor, lstm_hidden)
                    else:
                        out = model(tensor)

                key_logits = out["keys"][0]
                mouse_logits = out["mouse"][0]
                cursor_norm = out["cursor"][0].cpu().numpy()

                key_desired = sample_binary(key_logits, args.temperature)
                mouse_desired = sample_binary(mouse_logits, args.temperature)

                # Whitelist filter — force non-game keys to always "off"
                # regardless of what the model predicted. Prevents typing
                # artifacts learned from noisy training data.
                if GAME_RELEVANT_KEYS is not None:
                    for idx, name in enumerate(tracked_keys):
                        if name not in GAME_RELEVANT_KEYS:
                            key_desired[idx] = False

                # Resolve contradictory movement keys — pressing W+S or A+D
                # in Roblox cancels out (character stands still). Pick the
                # key with the higher logit; drop the loser.
                for opp_a, opp_b in [("w", "s"), ("a", "d")]:
                    try:
                        idx_a = tracked_keys.index(opp_a)
                        idx_b = tracked_keys.index(opp_b)
                    except ValueError:
                        continue
                    if key_desired[idx_a] and key_desired[idx_b]:
                        if key_logits[idx_a].item() >= key_logits[idx_b].item():
                            key_desired[idx_b] = False
                        else:
                            key_desired[idx_a] = False

                # Cursor absolute screen coords
                cx_win = int(cursor_norm[0] * region["width"])
                cy_win = int(cursor_norm[1] * region["height"])
                cx_scr = region["left"] + cx_win
                cy_scr = region["top"] + cy_win

                # Compute transitions (always) so status prints reflect what the model wants.
                # Actual key/mouse events only fire when not in dry_run.
                key_transitions = []
                for idx, name in enumerate(tracked_keys):
                    want = bool(key_desired[idx])
                    have = name in held_keys
                    if want and not have:
                        key_transitions.append(("down", name))
                        held_keys.add(name)
                    elif have and not want:
                        key_transitions.append(("up", name))
                        held_keys.discard(name)

                mouse_transitions = []
                for idx, mb in enumerate(["left", "right"]):
                    want = bool(mouse_desired[idx])
                    have = mb in held_mouse
                    if want and not have:
                        mouse_transitions.append(("down", mb))
                        held_mouse.add(mb)
                    elif have and not want:
                        mouse_transitions.append(("up", mb))
                        held_mouse.discard(mb)

                cursor_moved = (
                    last_cursor is None
                    or (abs(cx_scr - last_cursor[0]) + abs(cy_scr - last_cursor[1])) >= args.cursor_min_delta
                )

                if not args.dry_run:
                    for act, name in key_transitions:
                        try:
                            if act == "down": pydirectinput.keyDown(name)
                            else: pydirectinput.keyUp(name)
                        except Exception:
                            pass
                    for act, mb in mouse_transitions:
                        try:
                            if act == "down": pydirectinput.mouseDown(button=mb)
                            else: pydirectinput.mouseUp(button=mb)
                        except Exception:
                            pass
                    if cursor_moved:
                        move_mouse(cx_scr, cy_scr)

                if cursor_moved:
                    last_cursor = (cx_scr, cy_scr)

                frame_i += 1
                if frame_i % STATUS_EVERY_N_FRAMES == 0:
                    elapsed = time.time() - started
                    eff_fps = frame_i / elapsed
                    keys_str = ",".join(sorted(held_keys)) or "(none)"
                    mouse_str = ",".join(sorted(held_mouse)) or "(none)"
                    pollen = hud_state.get("pollen_fill")
                    pollen_str = f"{pollen*100:.0f}%" if pollen is not None else "?"
                    honey = hud_state.get("honey")
                    honey_str = f"{honey:,.0f}" if honey is not None else "?"
                    print(f"[t={frame_i:5d} {eff_fps:.1f}fps  pollen={pollen_str}  honey={honey_str}] "
                          f"keys={keys_str}  mouse={mouse_str}  "
                          f"cursor=({cx_win},{cy_win})")

                next_tick += tick
                if next_tick < now - tick:
                    next_tick = now + tick
        finally:
            release_all(held_keys, held_mouse)
            if orch is not None:
                orch.print_stats()
            print("[play] released all inputs, exiting")


if __name__ == "__main__":
    main()
