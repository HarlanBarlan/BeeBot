"""
Gymnasium environment wrapping Bee Swarm Simulator for RL.

Each step:
  1. Execute the last action (key/mouse presses via robo_input)
  2. Wait one tick (1/FPS seconds)
  3. Capture new screen frame
  4. Read HUD state (pollen, honey via OCR)
  5. Compute reward
  6. Return observation dict

Reset:
  Currently a no-op — Roblox has no episode reset. Training runs one
  continuous "episode" until the user stops it. Reward is stationary
  (per-tick honey delta) so no episode boundary is needed.

Observation:
  - image: (3, H, W) float32 in [0,1] — same preprocessing as imitation
  - hud: dict of scalar HUD values (pollen_fill, honey, etc.)

Action:
  - keys: multi-binary vector over GAME_RELEVANT_KEYS
  - mouse_buttons: multi-binary over [left, right]
  - cursor: (x_norm, y_norm) in [0,1] within Roblox window

For SB3 compat we flatten this into a single Box/Discrete space where
needed — see the ACTION_SPACE definition below.

NOTE: this is a SINGLE-ENV, SLOW-STEPPING environment. Each step is
1/FPS seconds real-time. PPO with 1 env at 10 FPS = ~600 samples/min.
Training will take days to see real improvement. That's expected for
a hobby-scale Roblox RL bot.
"""

import ctypes
from collections import deque
from ctypes import wintypes
import math
import time
from pathlib import Path
import numpy as np
import mss
import cv2
import gymnasium as gym
from gymnasium import spaces

# Try to use dxcam for faster screen capture (GPU-accelerated via DirectX
# Desktop Duplication API). Falls back to mss if dxcam isn't installed or
# can't create a camera in this session (e.g. some RDP configurations).
try:
    import dxcam
    _dxcam_available = True
except ImportError:
    _dxcam_available = False


class _RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG),
                ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


# Exclude the top strip of the window from cursor movement.
# - In windowed mode this covers the Windows title bar (avoid clicking X to close)
# - In borderless mode this avoids Roblox's top-bar dropdown that appears when
#   the mouse touches the very top edge (which covers the HUD)
CURSOR_TOP_EXCLUSION_PX = 50
# Also exclude a small strip along the bottom (chat bar edges) and sides
CURSOR_BOTTOM_EXCLUSION_PX = 15
CURSOR_SIDE_EXCLUSION_PX = 5

# Cursor is INCREMENTAL (delta from current position), not absolute.
# Reason: PPO's Gaussian policy over a Box space samples heavily into the
# clamped bounds — with absolute cursor that pins the cursor at the edges
# ~60% of the time. Delta encoding treats those extreme samples as "large
# movement" instead of "sit at edge", which lets the cursor move through
# the interior naturally.
# 0.5 in the action = no movement this step; 0 or 1 = full-speed left/right (up/down).
MAX_CURSOR_DELTA_PX = 80


def clip_cursor_to_region(region):
    """Confine the OS cursor to Roblox's window, minus edge exclusion zones
    that trigger bad game behavior (title bar clicks, top-bar dropdown)."""
    rect = _RECT(
        region["left"] + CURSOR_SIDE_EXCLUSION_PX,
        region["top"] + CURSOR_TOP_EXCLUSION_PX,
        region["left"] + region["width"] - CURSOR_SIDE_EXCLUSION_PX,
        region["top"] + region["height"] - CURSOR_BOTTOM_EXCLUSION_PX,
    )
    ctypes.windll.user32.ClipCursor(ctypes.byref(rect))


def release_cursor_clip():
    """Undo clip_cursor_to_region. Always call on exit."""
    ctypes.windll.user32.ClipCursor(None)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from roblox_window import get_roblox_region
from robo_input import move_mouse
from dataset import MODEL_INPUT_W, MODEL_INPUT_H, GAME_RELEVANT_KEYS
from hud.reader import HudReader
try:
    from hud.text_triggers import TextTriggers
    _text_triggers_available = True
except ImportError:
    _text_triggers_available = False
from .reward import MultiTimescaleReward
from .supervisor import ensure_bss_running, is_bss_loaded
import pydirectinput
import keyboard as kb_lib


# How often to check that BSS is still loaded (seconds). Every check is cheap
# (window title lookup) so this can be frequent without overhead.
BSS_HEALTH_CHECK_INTERVAL_SEC = 30.0

# How often to run text triggers (OCR whole popup region). Full OCR takes
# 100-400ms so we don't do it every step — every 5 sec is plenty for popups.
TEXT_TRIGGER_INTERVAL_SEC = 5.0


TICK_HZ = 10
TICK_SECONDS = 1.0 / TICK_HZ
QUIT_KEY = "esc"

# Hold this key to temporarily freeze the bot so you can drive manually.
# While held: bot releases all keys/mouse, releases the cursor clip, and
# skips action-apply. Training keeps running (rollout still collects the
# capture, reward, and observation) — the bot just stops issuing inputs.
# Useful for nudging the bot when it gets stuck somewhere unproductive,
# without stopping/restarting the training session.
PAUSE_KEY = "f8"

# Rate-limit HUD OCR — it's the slowest thing per step (~100-500ms each read).
# HUD state doesn't change that fast anyway. Every 10 steps = ~1 sec, plenty
# for return-to-hive detection and reward tracking.
HUD_READ_EVERY_N_STEPS = 10

# Rate-limit window lookup — pygetwindow enumerates ALL system windows every
# call (~100-500ms on a busy machine). Roblox window rarely moves. Cache
# for a few seconds between refreshes.
REGION_REFRESH_EVERY_N_STEPS = 50    # ~5 sec at 10 fps

# Artificial episode length for SB3's Monitor wrapper. Our env has no
# natural episode end (game runs forever), so we mark truncated=True every
# N steps. This gives SB3 something to average over for ep_rew_mean (the
# master learning-progress metric). Doesn't reset game state — bot keeps
# playing across "episodes" seamlessly.
EPISODE_LENGTH_STEPS = 1024

# Dialogue-escape safety net — if pollen/honey haven't budged for this many
# steps (~60 sec at 10 fps), the bot is likely stuck in a bear quest dialogue.
# BSS dialogues close by clicking IN the dialogue box repeatedly to advance
# through text. Auto-click at center of screen every N stalled steps —
# progresses any open dialogue, harmless otherwise (in-world click = tool swing).
# This doesn't teach bot to avoid bears; it just prevents indefinite stalls.
STALL_STEPS_BEFORE_DIALOGUE_CLICK = 600
DIALOGUE_CLICK_INTERVAL_STEPS = 30    # ~3 sec between rescue clicks

# After a successful dialogue-rescue click, suppress the bot's E key for a
# while so it can walk out of the NPC's E-range before its policy re-presses
# E and re-opens the dialogue. Without this we've seen the rescue fire
# hundreds of times in a row (dialogue closes → bot presses E → dialogue
# opens → repeat) for over an hour of wasted training.
POST_RESCUE_E_SUPPRESS_STEPS = 50     # ~10 sec at 5 fps — enough for the
                                      # policy's WASD to walk the bot out

# How many rapid-fire clicks per rescue firing. BSS bear dialogues have
# multiple text lines and each click only advances ONE line — a single click
# per firing leaves the bot stuck in a slow chain. Community consensus:
# 5-15 clicks per exchange. Bursting through in one shot resolves most
# dialogues fully instead of doing them one line every 6 seconds.
DIALOGUE_RESCUE_CLICK_BURST = 10
DIALOGUE_RESCUE_CLICK_DELAY = 0.05    # seconds between burst clicks
                                      # (~200ms total for a 10-click burst)

# Only expose game-relevant keys in the action space (matches imitation training)
ACTION_KEYS = sorted(GAME_RELEVANT_KEYS)
N_KEYS = len(ACTION_KEYS)
N_MOUSE = 2  # [left, right]


# --- HUD observation vector ---------------------------------------------------
# The HUD scalar vector is the second half of the Dict observation. It gives
# the critic (and policy) ground-truth game state instead of forcing them to
# visually parse the HUD from downsized pixels — the single biggest lever
# for fixing sparse-reward value-function fitting (Andrychowicz 2020).
#
# Each dim is bounded roughly to [-1, 1] via the normalizers below. When an
# OCR read is stale or missing, the value is 0 and the *_valid flag is 0 so
# the policy can learn to distinguish "unknown" from "actually zero".
#
# Keep the vector SMALL (~8 dims): our on-policy sample rate is ~5 fps, so
# over-parameterizing the HUD MLP on many inputs overfits fast.

HUD_DIM = 8
HUD_INDEX = {
    "pollen_fill":                0,   # 0..1 (direct)
    "honey_log_norm":             1,   # log(1+honey) / log(1+HONEY_LOG_CAP), clipped
    "pollen_delta_60s":           2,   # (pollen_fill_now - pollen_fill_60s_ago), typ [-1,1]
    "honey_delta_60s_log_norm":   3,   # log(1+Δhoney)/log(1+HONEY_DELTA_LOG_CAP), clipped
    "time_since_honey_gain":      4,   # clip(sec, 300) / 300
    "time_since_pollen_change":   5,   # clip(sec, 60) / 60
    "pollen_valid":               6,   # 1 if OCR read within HUD_STALE_SEC else 0
    "honey_valid":                7,   # 1 if OCR read within HUD_STALE_SEC else 0
}

HUD_HISTORY_SECONDS = 60.0           # window over which deltas are computed
HUD_STALE_SEC = 30.0                 # a read older than this counts as "no data"
HONEY_LOG_CAP = 1e12                 # 1 trillion honey — endgame ceiling
HONEY_DELTA_LOG_CAP = 1e10           # 10 billion Δhoney/min — well above any real burst
POLLEN_CHANGE_EPSILON = 0.005        # fill delta below this doesn't count as "changed"


class BSSEnv(gym.Env):
    """Bee Swarm Simulator as a Gym environment."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        # Observation: Dict of pixel frame + scalar HUD vector.
        # The HUD vector is critical for value-function fitting under sparse
        # reward — the critic gets ground-truth pollen/honey/rate scalars
        # instead of having to visually parse HUD text from downsized pixels.
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0.0, high=1.0,
                shape=(3, MODEL_INPUT_H, MODEL_INPUT_W),
                dtype=np.float32,
            ),
            # HUD dims are normalized to roughly [-1, 1]; use wider bounds
            # to avoid clipping if OCR spikes momentarily.
            "hud": spaces.Box(
                low=-2.0, high=2.0,
                shape=(HUD_DIM,),
                dtype=np.float32,
            ),
        })
        # Action: multi-binary for keys + mouse buttons, plus (cursor_x, cursor_y)
        # SB3 needs a single Box; we flatten into a length-(N_KEYS + N_MOUSE + 2) vector
        # where the first N_KEYS+N_MOUSE entries are 0/1 sigmoid outputs and the last 2
        # are cursor coords in [0,1].
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(N_KEYS + N_MOUSE + 2,),
            dtype=np.float32,
        )
        # Screen capture: prefer dxcam (GPU-accelerated), fall back to mss.
        # dxcam grabs frames directly from GPU framebuffer via DirectX
        # Desktop Duplication — much faster than mss's CPU-side path.
        self._dxcam = None
        if _dxcam_available:
            try:
                self._dxcam = dxcam.create(output_color="BGR")
                print("[env] using dxcam for screen capture (GPU-accelerated)")
            except Exception as e:
                print(f"[env] dxcam init failed ({e}) — falling back to mss")
                self._dxcam = None
        if self._dxcam is None:
            print("[env] using mss for screen capture (CPU-side)")
        self._sct = mss.MSS()      # always available as fallback

        self._hud = HudReader()
        # Text-trigger popup dismissal — auto-escapes known dialogs so bot
        # doesn't spend hours stuck in a quest screen.
        # Optional: if hud/text_triggers.py isn't present, feature is skipped.
        self._text_triggers = None
        if _text_triggers_available:
            try:
                self._text_triggers = TextTriggers()
                print(f"[env] text triggers loaded ({len(self._text_triggers.rules)} rules)")
            except Exception as e:
                print(f"[env] text triggers disabled: {e}")
                self._text_triggers = None
        self._last_text_trigger_check = 0.0
        self._reward = MultiTimescaleReward()
        self._held_keys = set()
        self._held_mouse = set()
        self._region = None
        self._last_cursor = None
        self._last_health_check = 0.0
        self._step_count = 0
        self._session_start_honey = None
        self._last_status_print_step = 0
        self._last_status_print_ts = time.time()
        self._cached_hud = {}          # last successful HUD read, reused between OCR calls
        self._stalled_since_step = None
        self._last_rescue_click_step = 0
        self._last_seen_honey = None
        self._last_seen_pollen = None
        self._steps_in_episode = 0
        # HUD history for delta features: rolling buffers of (timestamp, value).
        # Maxlen sized for HUD_HISTORY_SECONDS at TICK_HZ, doubled for safety.
        history_maxlen = int(HUD_HISTORY_SECONDS * TICK_HZ * 2)
        self._pollen_history = deque(maxlen=history_maxlen)
        self._honey_history = deque(maxlen=history_maxlen)
        # Wall-clock timestamps of last observed change / gain
        self._last_pollen_change_ts = None
        self._last_honey_gain_ts = None
        # Last OCR-read timestamps (for validity flags)
        self._last_pollen_read_ts = None
        self._last_honey_read_ts = None
        # Manual-pause state (F8 hold) — tracked so we only print on transitions
        self._paused = False
        # Step index up to which the E key is suppressed (set after dialogue
        # rescue clicks to prevent immediate re-trigger of the same dialogue)
        self._e_suppress_until_step = -1
        pydirectinput.FAILSAFE = True
        # Kill pydirectinput's default 100ms sleep after every call. That
        # global PAUSE was silently adding 500-1000ms per env step because
        # each keyDown/keyUp/mouseDown sleeps 100ms after by default.
        pydirectinput.PAUSE = 0

        # Auto-launch BSS if not already loaded when the env is created
        if not is_bss_loaded():
            print("[env] BSS not loaded on startup — launching")
            ensure_bss_running()

    # --- Gym API -----------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Release anything we were holding, reset reward tracking
        self._release_all()
        self._reward.reset()
        self._steps_in_episode = 0
        # Verify BSS is loaded — recover if crashed
        if not is_bss_loaded():
            print("[env] BSS not loaded during reset — attempting recovery")
            ensure_bss_running()
        obs, _ = self._capture()
        # Re-clip cursor to Roblox after each reset — window may have moved
        if self._region is not None:
            clip_cursor_to_region(self._region)
        return obs, {}

    def step(self, action):
        # Per-phase timing for diagnostics (printed every 100 steps)
        _t0 = time.time()

        # Manual pause: hold PAUSE_KEY to freeze bot inputs so a human can
        # nudge/reposition the character. Rollout keeps running (obs, reward,
        # value estimates still collected) — the bot just isn't the one
        # driving.
        pause_held = kb_lib.is_pressed(PAUSE_KEY)
        if pause_held and not self._paused:
            print(f"[env t={self._step_count}] MANUAL PAUSE (holding {PAUSE_KEY.upper()}) — "
                  f"bot released; drive manually")
            self._paused = True
            self._release_all()
            release_cursor_clip()
        elif not pause_held and self._paused:
            print(f"[env t={self._step_count}] resuming bot control")
            self._paused = False

        # Apply action (skip while paused so user has full control)
        if not self._paused:
            self._apply_action(action)
        _t_action = time.time() - _t0
        _t0 = time.time()

        # Wait one tick before reading next state
        time.sleep(TICK_SECONDS)
        _t_sleep = time.time() - _t0
        _t0 = time.time()

        # Periodic health check — if BSS crashed/disconnected, recover
        now = time.time()
        if now - self._last_health_check >= BSS_HEALTH_CHECK_INTERVAL_SEC:
            self._last_health_check = now
            if not is_bss_loaded():
                print("[env] health check: BSS gone — recovering")
                self._release_all()
                ensure_bss_running()
                # Reset LSTM-ish state via reward — fresh episode after recovery
                self._reward.reset()

        # Capture new observation and HUD (also refreshes self._region)
        obs, hud = self._capture()
        _t_capture = time.time() - _t0

        # Text-trigger check (rate-limited — OCR is expensive)
        if (self._text_triggers is not None
                and self._region is not None
                and now - self._last_text_trigger_check >= TEXT_TRIGGER_INTERVAL_SEC):
            self._last_text_trigger_check = now
            try:
                # Reuse the last-captured raw frame if we have it
                frame_for_ocr = getattr(self, "_last_captured_frame", None)
                if frame_for_ocr is not None:
                    self._text_triggers.check(frame_for_ocr, self._region)
            except Exception as e:
                print(f"[env] text trigger check failed: {e}")

        if self._step_count % 100 == 99:
            print(f"[timing] step {self._step_count}: action={_t_action*1000:.0f}ms "
                  f"sleep={_t_sleep*1000:.0f}ms capture={_t_capture*1000:.0f}ms")

        # Re-apply cursor clip each step — Windows silently releases it on
        # focus changes (alt-tab, system notifications, Roblox internal
        # window ops). Cheap to re-apply. Skip while paused so the user's
        # mouse can leave the Roblox window freely.
        if self._region is not None and not self._paused:
            clip_cursor_to_region(self._region)

        reward = self._reward.compute(hud, obs_frame=obs)

        # Dialogue-escape safety: if pollen and honey haven't changed for
        # STALL_STEPS_BEFORE_DIALOGUE_CLICK, click at center of screen to
        # advance any open dialogue. Not a punishment — bot still gets to
        # choose whether to engage bears. Just prevents indefinite stall.
        current_honey = hud.get("honey")
        current_pollen = hud.get("pollen_fill")
        state_changed = False
        if current_honey is not None and self._last_seen_honey is not None:
            if current_honey != self._last_seen_honey:
                state_changed = True
        if current_pollen is not None and self._last_seen_pollen is not None:
            if abs(current_pollen - self._last_seen_pollen) > 0.005:
                state_changed = True
        if current_honey is not None:
            self._last_seen_honey = current_honey
        if current_pollen is not None:
            self._last_seen_pollen = current_pollen

        if state_changed or self._stalled_since_step is None:
            self._stalled_since_step = self._step_count
        elif (self._step_count - self._stalled_since_step > STALL_STEPS_BEFORE_DIALOGUE_CLICK
              and self._step_count - self._last_rescue_click_step > DIALOGUE_CLICK_INTERVAL_STEPS
              and self._region is not None
              and not self._paused):
            self._try_dialogue_rescue_click(obs=None)
            self._last_rescue_click_step = self._step_count

        # Periodic human-readable status so you can see progress at a glance
        self._step_count += 1
        honey = hud.get("honey")
        # Session-start reference: use the SMALLEST honey reading seen in the
        # first 60 seconds. OCR sometimes misreads a huge phantom value on the
        # first frame before HUD is stable — waiting a moment then taking the
        # min avoids locking into that phantom as the baseline.
        if honey is not None and self._step_count < 600:  # first 60s at 10 FPS
            if self._session_start_honey is None or honey < self._session_start_honey:
                self._session_start_honey = honey
        if self._step_count - self._last_status_print_step >= 300:  # every ~30s at 10 FPS
            elapsed = time.time() - self._last_status_print_ts
            steps_since = self._step_count - self._last_status_print_step
            live_fps = steps_since / max(elapsed, 0.01)
            self._last_status_print_step = self._step_count
            self._last_status_print_ts = time.time()
            pollen = hud.get("pollen_fill")
            pollen_str = f"{pollen*100:.0f}%" if pollen is not None else "?"
            honey_str = f"{honey:,.0f}" if honey is not None else "?"
            gained = ""
            if honey is not None and self._session_start_honey is not None:
                delta = honey - self._session_start_honey
                gained = f"  (+{delta:,.0f} this session)"
            print(f"[env t={self._step_count:6d} {live_fps:.1f}fps] pollen={pollen_str}  honey={honey_str}{gained}  "
                  f"total_reward={self._reward.total_reward_this_episode:.2f}")

        # We do NOT terminate on ESC here — that would just start a new
        # episode. Stopping training is handled by StopOnKeyCallback in
        # train_ppo.py, which halts the training loop entirely.
        terminated = False
        # Artificial episode boundary every EPISODE_LENGTH_STEPS so SB3's
        # Monitor wrapper can compute ep_rew_mean. Doesn't reset game state
        # (reset() releases keys + resets reward accumulator, but the game
        # continues from where the bot is).
        self._steps_in_episode += 1
        truncated = self._steps_in_episode >= EPISODE_LENGTH_STEPS

        info = {"hud": hud, "total_reward": self._reward.total_reward_this_episode}
        return obs, reward, terminated, truncated, info

    def close(self):
        self._release_all()
        release_cursor_clip()
        try:
            self._sct.close()
        except Exception:
            pass
        if self._dxcam is not None:
            try:
                self._dxcam.release()
            except Exception:
                pass

    # --- internals ----------------------------------------------------------

    def _empty_observation(self):
        """Zero-filled observation matching the Dict space — used when
        Roblox window can't be located (bot is between windows, alt-tabbed,
        etc.). Returned along with the cached HUD dict."""
        return {
            "image": np.zeros(
                (3, MODEL_INPUT_H, MODEL_INPUT_W), dtype=np.float32
            ),
            "hud": np.zeros((HUD_DIM,), dtype=np.float32),
        }

    def _capture(self):
        # Refresh Roblox region only every N steps — pygetwindow.getAllWindows
        # is slow and window rarely moves during training
        if self._region is None or self._step_count % REGION_REFRESH_EVERY_N_STEPS == 0:
            try:
                self._region = get_roblox_region()
            except RuntimeError:
                time.sleep(0.5)
                return self._empty_observation(), self._cached_hud

        frame = None
        # Try dxcam first (fast, GPU-accelerated)
        if self._dxcam is not None:
            left = self._region["left"]
            top = self._region["top"]
            right = left + self._region["width"]
            bottom = top + self._region["height"]
            try:
                frame = self._dxcam.grab(region=(left, top, right, bottom))
                # dxcam returns None if no new frame available since last grab —
                # very common at high poll rates. If None, use cached frame or fall back.
                if frame is None:
                    frame = getattr(self, "_last_captured_frame", None)
            except Exception:
                frame = None

        # Fallback to mss if dxcam failed or returned nothing
        if frame is None:
            shot = self._sct.grab(self._region)
            frame = np.array(shot)[:, :, :3]  # BGRA -> BGR

        # Cache latest frame so a subsequent None from dxcam still gets recent pixels
        self._last_captured_frame = frame

        # OCR is expensive (~100-500ms each read) so only run every N steps.
        # Between reads, return the last cached values — pollen/honey don't
        # change faster than 1 Hz in practice anyway.
        if self._step_count % HUD_READ_EVERY_N_STEPS == 0:
            new_hud = self._hud.read(frame.copy())
            if new_hud:
                self._cached_hud.update(new_hud)
        hud = self._cached_hud

        # Downsize + normalize + CHW for observation
        small = cv2.resize(frame, (MODEL_INPUT_W, MODEL_INPUT_H))
        small_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_obs = np.transpose(small_rgb, (2, 0, 1))

        # Update HUD history + compute scalar HUD vector for policy/critic
        now = time.time()
        self._update_hud_history(hud, now)
        hud_vec = self._compute_hud_vector(hud, now)

        return {"image": img_obs, "hud": hud_vec}, hud

    def _update_hud_history(self, hud, now):
        """Append current HUD values to rolling history + update change/gain
        timestamps. Called each capture (rate-limited by HUD OCR cadence,
        which itself is every HUD_READ_EVERY_N_STEPS steps)."""
        pollen = hud.get("pollen_fill")
        if pollen is not None:
            self._pollen_history.append((now, float(pollen)))
            self._last_pollen_read_ts = now
            # Update "changed" timestamp only on meaningful change
            if (self._last_seen_pollen is None
                    or abs(pollen - self._last_seen_pollen) > POLLEN_CHANGE_EPSILON):
                self._last_pollen_change_ts = now
        honey = hud.get("honey")
        if honey is not None:
            self._honey_history.append((now, float(honey)))
            self._last_honey_read_ts = now
            # Honey only ever increases in normal play; treat any gain as an event
            if self._last_seen_honey is None or honey > self._last_seen_honey:
                self._last_honey_gain_ts = now

    def _compute_hud_vector(self, hud, now):
        """Build the fixed-size HUD scalar vector from cached HUD state +
        history buffers. Everything is normalized to roughly [-1, 1] and
        clipped. Missing readings return 0 for the value dim and 0 for the
        corresponding *_valid flag."""
        vec = np.zeros((HUD_DIM,), dtype=np.float32)

        pollen = hud.get("pollen_fill")
        if pollen is not None:
            vec[HUD_INDEX["pollen_fill"]] = float(np.clip(pollen, 0.0, 1.0))

        honey = hud.get("honey")
        if honey is not None:
            honey = max(0.0, float(honey))
            vec[HUD_INDEX["honey_log_norm"]] = float(np.clip(
                math.log1p(honey) / math.log1p(HONEY_LOG_CAP), 0.0, 1.0
            ))

        # Deltas over HUD_HISTORY_SECONDS
        vec[HUD_INDEX["pollen_delta_60s"]] = self._history_delta(
            self._pollen_history, now, HUD_HISTORY_SECONDS
        )
        honey_delta = self._history_delta(
            self._honey_history, now, HUD_HISTORY_SECONDS
        )
        # Honey delta is huge in magnitude; log-normalize and clip
        # (drop sign — honey only goes up in normal play; a negative delta
        # is either OCR noise or a session reset, both of which shouldn't
        # feed the policy negative "learn to lose honey" signals)
        honey_delta = max(0.0, honey_delta)
        vec[HUD_INDEX["honey_delta_60s_log_norm"]] = float(np.clip(
            math.log1p(honey_delta) / math.log1p(HONEY_DELTA_LOG_CAP), 0.0, 1.0
        ))

        # Time-since-events, clipped and normalized
        if self._last_honey_gain_ts is not None:
            since = min(now - self._last_honey_gain_ts, 300.0)
            vec[HUD_INDEX["time_since_honey_gain"]] = float(since / 300.0)
        else:
            vec[HUD_INDEX["time_since_honey_gain"]] = 1.0   # unknown -> "long time ago"
        if self._last_pollen_change_ts is not None:
            since = min(now - self._last_pollen_change_ts, 60.0)
            vec[HUD_INDEX["time_since_pollen_change"]] = float(since / 60.0)
        else:
            vec[HUD_INDEX["time_since_pollen_change"]] = 1.0

        # Validity flags — the policy can learn "if valid=0, ignore this dim"
        if self._last_pollen_read_ts is not None and now - self._last_pollen_read_ts < HUD_STALE_SEC:
            vec[HUD_INDEX["pollen_valid"]] = 1.0
        if self._last_honey_read_ts is not None and now - self._last_honey_read_ts < HUD_STALE_SEC:
            vec[HUD_INDEX["honey_valid"]] = 1.0

        return vec

    @staticmethod
    def _history_delta(history, now, window_sec):
        """Return (latest_value - earliest_value_within_window). If there's
        no data in the window, returns 0.0."""
        if not history:
            return 0.0
        latest_ts, latest_val = history[-1]
        # Find earliest entry within the window
        earliest_val = None
        for ts, val in history:
            if now - ts <= window_sec:
                earliest_val = val
                break
        if earliest_val is None:
            return 0.0
        return latest_val - earliest_val

    def _apply_action(self, action):
        # Split action vector
        key_probs = action[:N_KEYS]
        mouse_probs = action[N_KEYS:N_KEYS + N_MOUSE]
        cursor = action[N_KEYS + N_MOUSE:N_KEYS + N_MOUSE + 2]

        # Threshold at 0.5 for on/off (PPO learns a distribution; threshold is a
        # bit crude but simple — a Bernoulli action head would be cleaner)
        for i, name in enumerate(ACTION_KEYS):
            want = key_probs[i] > 0.5
            # Post-rescue E suppression: force E off for a window after a
            # dialogue-rescue click so the bot can escape the NPC's E-range
            # before its policy re-triggers the dialogue.
            if name == "e" and self._step_count < self._e_suppress_until_step:
                want = False
            have = name in self._held_keys
            if want and not have:
                try: pydirectinput.keyDown(name)
                except Exception: pass
                self._held_keys.add(name)
            elif have and not want:
                try: pydirectinput.keyUp(name)
                except Exception: pass
                self._held_keys.discard(name)

        for i, mb in enumerate(["left", "right"]):
            want = mouse_probs[i] > 0.5
            have = mb in self._held_mouse
            if want and not have:
                try: pydirectinput.mouseDown(button=mb)
                except Exception: pass
                self._held_mouse.add(mb)
            elif have and not want:
                try: pydirectinput.mouseUp(button=mb)
                except Exception: pass
                self._held_mouse.discard(mb)

        # Cursor as DELTA from current position (not absolute).
        # cursor[i] in [0, 1]: 0.5 = no movement, extremes = full-speed shift.
        if self._region is not None:
            # Initialize cursor to window center on first step
            if self._last_cursor is None:
                cx = self._region["left"] + self._region["width"] // 2
                cy = self._region["top"] + self._region["height"] // 2
            else:
                # Map [0,1] action to [-1,+1] delta scale, then to pixels
                dx = (float(cursor[0]) - 0.5) * 2.0 * MAX_CURSOR_DELTA_PX
                dy = (float(cursor[1]) - 0.5) * 2.0 * MAX_CURSOR_DELTA_PX
                cx = self._last_cursor[0] + int(dx)
                cy = self._last_cursor[1] + int(dy)

            # Enforce exclusion zones — cursor never lands on title bar,
            # top-bar dropdown trigger, or extreme edges. Deltas that would
            # push out just get truncated at the boundary and the cursor
            # will "bounce" back into the interior on the next step.
            cx = max(self._region["left"] + CURSOR_SIDE_EXCLUSION_PX,
                     min(self._region["left"] + self._region["width"] - CURSOR_SIDE_EXCLUSION_PX, cx))
            cy = max(self._region["top"] + CURSOR_TOP_EXCLUSION_PX,
                     min(self._region["top"] + self._region["height"] - CURSOR_BOTTOM_EXCLUSION_PX, cy))

            # Only actually move if delta is meaningful (avoid jittery micro-moves)
            if self._last_cursor is None or abs(cx - self._last_cursor[0]) + abs(cy - self._last_cursor[1]) >= 4:
                move_mouse(cx, cy)
                self._last_cursor = (cx, cy)

    def _release_all(self):
        for k in list(self._held_keys):
            try: pydirectinput.keyUp(k)
            except Exception: pass
        for mb in list(self._held_mouse):
            try: pydirectinput.mouseUp(button=mb)
            except Exception: pass
        self._held_keys.clear()
        self._held_mouse.clear()

    def _try_dialogue_rescue_click(self, obs=None):
        """When bot's been stalled long enough that we suspect it's stuck in
        a bear dialogue, template-match on the 'click to continue' text and
        click that spot to advance the dialogue.

        User snips the template once: `snip_template.py bridges/probes/dialogue_continue`
        while a dialogue is open. Template should be tight around the "click
        to continue" text or similar dialogue-advance element.

        If template file is missing, prints a helpful note (once) and skips.
        """
        template_path = Path(__file__).parent.parent / "bridges" / "probes" / "dialogue_continue.png"
        if not template_path.exists():
            if not getattr(self, "_dialogue_template_warned", False):
                print(f"[env] stall rescue skipped — no dialogue template at {template_path}. "
                      f"Snip one with: snip_template.py bridges/probes/dialogue_continue")
                self._dialogue_template_warned = True
            return

        template = cv2.imread(str(template_path))
        if template is None:
            return
        th, tw = template.shape[:2]

        # Grab current frame (small overhead, only fires on stall)
        try:
            shot = self._sct.grab(self._region)
            frame = np.array(shot)[:, :, :3]
        except Exception:
            return

        result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        if max_val < 0.65:
            # No dialogue visible — don't click randomly. Bot is stuck on
            # something else (wall, weird visual state); RL learns from that.
            return

        x, y = max_loc
        cx = self._region["left"] + x + tw // 2
        cy = self._region["top"] + y + th // 2
        move_mouse(cx, cy)
        time.sleep(0.03)
        # Rapid-fire burst — BSS dialogues have multiple text lines and each
        # click only advances one line. Burst clicks through the whole chain
        # in one shot instead of doing it one line every 6 seconds.
        for _ in range(DIALOGUE_RESCUE_CLICK_BURST):
            pydirectinput.mouseDown(button="left")
            time.sleep(DIALOGUE_RESCUE_CLICK_DELAY / 2)
            pydirectinput.mouseUp(button="left")
            time.sleep(DIALOGUE_RESCUE_CLICK_DELAY / 2)
        # Force-release E if held, then suppress E for a window so the
        # bot can escape the NPC's proximity before its policy re-presses
        # E and re-opens the dialogue we just closed.
        if "e" in self._held_keys:
            try: pydirectinput.keyUp("e")
            except Exception: pass
            self._held_keys.discard("e")
        self._e_suppress_until_step = self._step_count + POST_RESCUE_E_SUPPRESS_STEPS
        print(f"[env t={self._step_count}] dialogue-rescue: burst {DIALOGUE_RESCUE_CLICK_BURST} "
              f"clicks at ({cx},{cy}) (match conf {max_val:.2f}) — "
              f"E suppressed for {POST_RESCUE_E_SUPPRESS_STEPS} steps")
