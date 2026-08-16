"""
PyTorch Dataset for BeeBot imitation training.

Reads every session under data/session_* and yields (image, labels) pairs.

Labels for each frame:
  - keys: (75,) float tensor, 1 if key held, 0 if not
  - mouse_buttons: (2,) float tensor for [left, right]
  - cursor_xy: (2,) float tensor, normalized to [0,1] within Roblox window
  - cursor_valid: (1,) float, 1 if cursor was inside Roblox window else 0

Loss should mask cursor_xy by cursor_valid so out-of-window frames don't
poison the regression.
"""

import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

MODEL_INPUT_W = 240
MODEL_INPUT_H = 135

# Whitelist of keys that actually do something in Bee Swarm Simulator.
# Everything else in the recording is treated as noise (typing artifacts,
# accidental presses) and its label is masked to 0 before training.
# The model still has an output logit for every tracked key, but the loss
# only sees positives for game-relevant keys — non-game outputs collapse
# to "always off" naturally.
# NOTE: play.py has an identical set. If you change one, change both.
GAME_RELEVANT_KEYS = {
    "w", "a", "s", "d",
    "space", "shift",
    "e",                                  # interact
    "1", "2", "3", "4", "5", "6",         # hotbar
    ",", ".", "i", "o",                   # camera keys
    "enter",                              # chat / dialog confirm
    # NOTE: backspace deliberately removed — was capturing typing during
    # hive conversions in training. If a dialog ever needs closing, ESC works.
}


class GameplayDataset(Dataset):
    def __init__(self, data_root="data", split="train", val_frac=0.1):
        self.records = []          # list of (frame_path, keys, mouse, cursor, valid)
        self.tracked_keys = None   # confirmed identical across sessions

        for session_dir in sorted(Path(data_root).glob("session_*")):
            session_json = session_dir / "session.json"
            labels_path = session_dir / "labels.jsonl"
            if not session_json.exists() or not labels_path.exists():
                print(f"[skip] {session_dir.name}: missing session.json or labels.jsonl")
                continue

            with open(session_json) as f:
                meta = json.load(f)
            keys_in_session = meta["tracked_keys"]
            native_w = meta["roblox_native_width"]
            native_h = meta["roblox_native_height"]

            if self.tracked_keys is None:
                self.tracked_keys = keys_in_session
            elif keys_in_session != self.tracked_keys:
                print(f"[skip] {session_dir.name}: tracked_keys mismatch — "
                      f"was recorded with a different TRACKED_KEYS list")
                continue

            # Precompute a mask: 1.0 if this key is game-relevant, else 0.0.
            # We multiply labels by this mask when loading records so the
            # model never sees positive examples for non-game keys.
            mask = [1 if k in GAME_RELEVANT_KEYS else 0 for k in keys_in_session]

            with open(labels_path) as f:
                for line in f:
                    row = json.loads(line)
                    frame_path = session_dir / "frames" / row["frame"]
                    keys = [v * m for v, m in zip(row["keys"], mask)]
                    mouse = row["mouse"]
                    cx, cy = row.get("cursor", [None, None])
                    if cx is None or cy is None:
                        cursor = [0.5, 0.5]
                        valid = 0.0
                    else:
                        cursor = [cx / native_w, cy / native_h]
                        valid = 1.0
                    self.records.append((frame_path, keys, mouse, cursor, valid))

        if not self.records:
            raise RuntimeError(f"No data found under {data_root}/session_*")

        # Deterministic split: last val_frac of records go to validation
        n = len(self.records)
        split_idx = int(n * (1 - val_frac))
        if split == "train":
            self.records = self.records[:split_idx]
        elif split == "val":
            self.records = self.records[split_idx:]
        else:
            raise ValueError(f"unknown split: {split}")

        self.n_keys = len(self.tracked_keys)
        n_active = sum(1 for k in self.tracked_keys if k in GAME_RELEVANT_KEYS)
        n_masked = self.n_keys - n_active
        print(f"[dataset] {split}: {len(self.records)} frames, "
              f"{self.n_keys} tracked keys ({n_active} game, {n_masked} masked as noise)")

        # Compute class-balance stats for pos_weight in BCE loss.
        # Only meaningful on the training split.
        if split == "train":
            key_pos = [0] * self.n_keys
            mouse_pos = [0, 0]
            for _, keys, mouse, _, _ in self.records:
                for i, v in enumerate(keys):
                    if v: key_pos[i] += 1
                for i, v in enumerate(mouse):
                    if v: mouse_pos[i] += 1
            total = len(self.records)
            self.pos_counts_keys = key_pos
            self.pos_counts_mouse = mouse_pos
            self.total_frames = total
            # Print a few of the most imbalanced classes for visibility
            key_ratios = [(self.tracked_keys[i], key_pos[i] / max(total, 1))
                          for i in range(self.n_keys)]
            key_ratios.sort(key=lambda kv: kv[1], reverse=True)
            top5 = key_ratios[:5]
            print(f"[dataset] top-5 most-held keys: "
                  + ", ".join(f"{k}={r:.1%}" for k, r in top5))
            print(f"[dataset] mouse: left={mouse_pos[0]/total:.1%}, "
                  f"right={mouse_pos[1]/total:.1%}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        frame_path, keys, mouse, cursor, valid = self.records[i]
        img = _load_frame(frame_path)
        return {
            "image": torch.from_numpy(img),
            "keys": torch.tensor(keys, dtype=torch.float32),
            "mouse": torch.tensor(mouse, dtype=torch.float32),
            "cursor": torch.tensor(cursor, dtype=torch.float32),
            "cursor_valid": torch.tensor([valid], dtype=torch.float32),
        }


def _load_frame(frame_path):
    """Load an image, resize to model input, normalize to [0,1] CHW float32."""
    img = cv2.imread(str(frame_path))
    if img is None:
        raise RuntimeError(f"failed to read {frame_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (MODEL_INPUT_W, MODEL_INPUT_H))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return img


class GameplaySequenceDataset(Dataset):
    """Sequences of consecutive frames from single sessions — for LSTM training.

    Windows never cross session boundaries. Windows within a session can
    overlap (controlled by `stride`). Each item returns:
      image: (T, C, H, W)
      keys: (T, n_keys)
      mouse: (T, 2)
      cursor: (T, 2)
      cursor_valid: (T, 1)
    """

    def __init__(self, data_root="data", split="train", val_frac=0.1,
                 seq_len=30, stride=15):
        self.seq_len = seq_len
        self.tracked_keys = None
        self.sessions_records = []  # list of per-session records

        for session_dir in sorted(Path(data_root).glob("session_*")):
            session_json = session_dir / "session.json"
            labels_path = session_dir / "labels.jsonl"
            if not session_json.exists() or not labels_path.exists():
                print(f"[skip] {session_dir.name}: missing metadata")
                continue

            with open(session_json) as f:
                meta = json.load(f)
            keys_in_session = meta["tracked_keys"]
            native_w = meta["roblox_native_width"]
            native_h = meta["roblox_native_height"]

            if self.tracked_keys is None:
                self.tracked_keys = keys_in_session
            elif keys_in_session != self.tracked_keys:
                print(f"[skip] {session_dir.name}: tracked_keys mismatch")
                continue

            mask = [1 if k in GAME_RELEVANT_KEYS else 0 for k in keys_in_session]

            records = []
            with open(labels_path) as f:
                for line in f:
                    row = json.loads(line)
                    frame_path = session_dir / "frames" / row["frame"]
                    keys = [v * m for v, m in zip(row["keys"], mask)]
                    mouse = row["mouse"]
                    cx, cy = row.get("cursor", [None, None])
                    if cx is None or cy is None:
                        cursor = [0.5, 0.5]
                        valid = 0.0
                    else:
                        cursor = [cx / native_w, cy / native_h]
                        valid = 1.0
                    records.append((frame_path, keys, mouse, cursor, valid))
            self.sessions_records.append(records)

        if not self.sessions_records:
            raise RuntimeError(f"No data found under {data_root}/session_*")

        # Build window index: (session_idx, start_frame_idx)
        self.windows = []
        for sidx, records in enumerate(self.sessions_records):
            if len(records) < seq_len:
                continue
            for start in range(0, len(records) - seq_len + 1, stride):
                self.windows.append((sidx, start))

        if not self.windows:
            raise RuntimeError(f"No windows built (seq_len={seq_len} too long for all sessions?)")

        # Train/val split by windows — deterministic (last val_frac)
        n = len(self.windows)
        split_idx = int(n * (1 - val_frac))
        if split == "train":
            self.windows = self.windows[:split_idx]
        elif split == "val":
            self.windows = self.windows[split_idx:]
        else:
            raise ValueError(f"unknown split: {split}")

        self.n_keys = len(self.tracked_keys)
        total_frames = sum(len(r) for r in self.sessions_records)
        n_active = sum(1 for k in self.tracked_keys if k in GAME_RELEVANT_KEYS)
        n_masked = self.n_keys - n_active
        print(f"[seq-dataset] {split}: {len(self.windows)} windows of {seq_len} frames "
              f"(stride {stride}). {n_active} game keys, {n_masked} masked.")

        # Class balance for pos_weight — computed on the training split
        if split == "train":
            key_pos = [0] * self.n_keys
            mouse_pos = [0, 0]
            samples = 0
            for sidx, start in self.windows:
                for record in self.sessions_records[sidx][start:start + seq_len]:
                    _, keys, mouse, _, _ = record
                    for i, v in enumerate(keys):
                        if v: key_pos[i] += 1
                    for i, v in enumerate(mouse):
                        if v: mouse_pos[i] += 1
                    samples += 1
            self.pos_counts_keys = key_pos
            self.pos_counts_mouse = mouse_pos
            self.total_frames = samples
            ratios = sorted(
                [(self.tracked_keys[i], key_pos[i] / max(samples, 1))
                 for i in range(self.n_keys) if key_pos[i] > 0],
                key=lambda kv: kv[1], reverse=True,
            )[:5]
            print(f"[seq-dataset] top-5 held keys: "
                  + ", ".join(f"{k}={r:.1%}" for k, r in ratios))
            print(f"[seq-dataset] mouse: left={mouse_pos[0]/samples:.1%}, "
                  f"right={mouse_pos[1]/samples:.1%}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        sidx, start = self.windows[i]
        records = self.sessions_records[sidx][start:start + self.seq_len]
        images = np.stack([_load_frame(r[0]) for r in records])  # (T, C, H, W)
        keys = np.array([r[1] for r in records], dtype=np.float32)
        mouse = np.array([r[2] for r in records], dtype=np.float32)
        cursor = np.array([r[3] for r in records], dtype=np.float32)
        valid = np.array([[r[4]] for r in records], dtype=np.float32)
        return {
            "image": torch.from_numpy(images),
            "keys": torch.from_numpy(keys),
            "mouse": torch.from_numpy(mouse),
            "cursor": torch.from_numpy(cursor),
            "cursor_valid": torch.from_numpy(valid),
        }
