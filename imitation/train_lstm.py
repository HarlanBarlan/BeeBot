r"""
Train BeeBotLSTM (Phase 2b — CNN + LSTM memory).

Usage:
  .\.venv\Scripts\python.exe train_lstm.py

Sequence-based training: model sees 30-frame windows from single sessions
and learns to smooth actions over time. BPTT unrolls gradients through
the full window.

Outputs:
  models/beebot_lstm_epoch<N>.pt
  models/beebot_lstm_latest.pt
  models/beebot_lstm_best.pt

Same loss structure as Phase 2a: weighted BCE on keys/mouse + masked MSE
on cursor. Loss is computed per-frame, averaged across the sequence.
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import GameplaySequenceDataset
from .model import BeeBotLSTM

# --- hyperparams ------------------------------------------------------------
SEQ_LEN = 30                # frames per window (3 seconds at 10 FPS)
STRIDE = 15                 # overlap between windows (50%)
BATCH_SIZE = 8              # each batch item = 30 frames, memory scales with T
EPOCHS = 12
LR = 5e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 2
MODELS_DIR = Path("models")
POS_WEIGHT_MAX = 50.0
GRAD_CLIP = 1.0             # BPTT can produce large grads — clip for stability
# ----------------------------------------------------------------------------


def compute_pos_weight(pos_counts, total, cap):
    weights = []
    for pos in pos_counts:
        if pos <= 0:
            weights.append(1.0)
        else:
            neg = total - pos
            w = neg / pos
            weights.append(max(1.0, min(w, cap)))
    return torch.tensor(weights, dtype=torch.float32)


def compute_loss(out, batch, bce_keys, bce_mouse, mse):
    # out["keys"]: (B, T, n_keys); batch["keys"]: (B, T, n_keys)
    keys_loss = bce_keys(out["keys"], batch["keys"])
    mouse_loss = bce_mouse(out["mouse"], batch["mouse"])
    cursor_err = mse(out["cursor"], batch["cursor"])         # (B, T, 2)
    valid = batch["cursor_valid"]                            # (B, T, 1)
    cursor_loss = (cursor_err * valid).sum() / valid.sum().clamp(min=1)
    return keys_loss + mouse_loss + cursor_loss, {
        "keys": keys_loss.item(),
        "mouse": mouse_loss.item(),
        "cursor": cursor_loss.item(),
    }


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def main():
    print(f"[train-lstm] device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[train-lstm] GPU: {torch.cuda.get_device_name(0)}")

    train_set = GameplaySequenceDataset(split="train", seq_len=SEQ_LEN, stride=STRIDE)
    val_set = GameplaySequenceDataset(split="val", seq_len=SEQ_LEN, stride=STRIDE)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = BeeBotLSTM(n_keys=train_set.n_keys).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train-lstm] model: {n_params:,} params")

    keys_pw = compute_pos_weight(train_set.pos_counts_keys, train_set.total_frames, POS_WEIGHT_MAX).to(DEVICE)
    mouse_pw = compute_pos_weight(train_set.pos_counts_mouse, train_set.total_frames, POS_WEIGHT_MAX).to(DEVICE)
    print(f"[train-lstm] key pos_weight range: [{keys_pw.min().item():.2f}, {keys_pw.max().item():.2f}]")
    print(f"[train-lstm] mouse pos_weight: left={mouse_pw[0].item():.2f}, right={mouse_pw[1].item():.2f}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce_keys = nn.BCEWithLogitsLoss(pos_weight=keys_pw)
    bce_mouse = nn.BCEWithLogitsLoss(pos_weight=mouse_pw)
    mse = nn.MSELoss(reduction="none")

    MODELS_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        train_total = 0.0
        seen = 0
        for batch in train_loader:
            batch = to_device(batch, DEVICE)
            opt.zero_grad()
            out, _ = model(batch["image"])  # fresh hidden per window during training
            loss, parts = compute_loss(out, batch, bce_keys, bce_mouse, mse)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            bs = batch["image"].size(0)
            train_total += loss.item() * bs
            seen += bs
        train_avg = train_total / max(seen, 1)

        model.eval()
        val_total = 0.0
        val_seen = 0
        parts_sum = {"keys": 0.0, "mouse": 0.0, "cursor": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, DEVICE)
                out, _ = model(batch["image"])
                loss, parts = compute_loss(out, batch, bce_keys, bce_mouse, mse)
                bs = batch["image"].size(0)
                val_total += loss.item() * bs
                val_seen += bs
                for k, v in parts.items():
                    parts_sum[k] += v * bs
        val_avg = val_total / max(val_seen, 1)
        parts_avg = {k: v / max(val_seen, 1) for k, v in parts_sum.items()}

        elapsed = time.time() - t0
        print(f"[epoch {epoch:2d}] "
              f"train={train_avg:.4f}  val={val_avg:.4f}  "
              f"(keys={parts_avg['keys']:.3f}  mouse={parts_avg['mouse']:.3f}  "
              f"cursor={parts_avg['cursor']:.4f})  {elapsed:.1f}s")

        ckpt = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_avg,
            "tracked_keys": train_set.tracked_keys,
            "model_type": "BeeBotLSTM",
            "hidden_size": model.hidden_size,
            "seq_len": SEQ_LEN,
        }
        torch.save(ckpt, MODELS_DIR / f"beebot_lstm_epoch{epoch:02d}.pt")
        torch.save(ckpt, MODELS_DIR / "beebot_lstm_latest.pt")
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(ckpt, MODELS_DIR / "beebot_lstm_best.pt")
            print(f"          -> new best val loss, saved beebot_lstm_best.pt")


if __name__ == "__main__":
    main()
