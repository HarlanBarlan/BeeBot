r"""
Train BeeBotCNN on your recorded gameplay.

Usage:
  .\.venv\Scripts\python.exe train.py

Outputs a checkpoint per epoch under models/beebot_epoch<N>.pt plus
models/beebot_latest.pt on every save so play.py always has one to load.

Loss = BCE(keys) + BCE(mouse) + MSE(cursor), where cursor loss is masked
by cursor_valid so out-of-window frames don't corrupt the regression.
"""

import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GameplayDataset
from model import BeeBotCNN

# --- hyperparams ------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
MODELS_DIR = Path("models")
# Cap pos_weight so extremely rare keys (e.g. F9 pressed 3 times) don't
# get weighted at 25000× and dominate the loss. 50× is generous but sane.
POS_WEIGHT_MAX = 50.0
# ----------------------------------------------------------------------------


def compute_pos_weight(pos_counts, total, cap):
    """pos_weight[i] = (negatives_i / positives_i), clamped to [1.0, cap].
    - Below 1.0 would DOWN-weight positives — wrong for majority classes
      like mouse-left when it's held 54% of the time. Clamp min=1.0 so we
      never penalize positives less than negatives.
    - Above `cap` blows up loss for extreme rarities; clamp max=cap."""
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
    keys_loss = bce_keys(out["keys"], batch["keys"])
    mouse_loss = bce_mouse(out["mouse"], batch["mouse"])
    # Mask cursor loss by validity so frames with cursor outside window
    # don't pull the regression toward (0.5, 0.5)
    cursor_err = mse(out["cursor"], batch["cursor"])       # (B, 2)
    valid = batch["cursor_valid"]                          # (B, 1)
    cursor_loss = (cursor_err * valid).sum() / valid.sum().clamp(min=1)
    return keys_loss + mouse_loss + cursor_loss, {
        "keys": keys_loss.item(),
        "mouse": mouse_loss.item(),
        "cursor": cursor_loss.item(),
    }


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def main():
    print(f"[train] device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(0)}")

    train_set = GameplayDataset(split="train")
    val_set = GameplayDataset(split="val")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=True)

    model = BeeBotCNN(n_keys=train_set.n_keys).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model: {n_params:,} params")

    # Weighted BCE to counter class imbalance — rare positives (e.g. mouse
    # clicks at ~5% frequency) get up-weighted so the model actually predicts
    # them instead of always saying "off". Capped at POS_WEIGHT_MAX to
    # prevent extreme-rarity keys from blowing up the loss.
    keys_pw = compute_pos_weight(train_set.pos_counts_keys, train_set.total_frames, POS_WEIGHT_MAX).to(DEVICE)
    mouse_pw = compute_pos_weight(train_set.pos_counts_mouse, train_set.total_frames, POS_WEIGHT_MAX).to(DEVICE)
    print(f"[train] key pos_weight range: [{keys_pw.min().item():.2f}, {keys_pw.max().item():.2f}]")
    print(f"[train] mouse pos_weight: left={mouse_pw[0].item():.2f}, right={mouse_pw[1].item():.2f}")

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    bce_keys = nn.BCEWithLogitsLoss(pos_weight=keys_pw)
    bce_mouse = nn.BCEWithLogitsLoss(pos_weight=mouse_pw)
    mse = nn.MSELoss(reduction="none")

    MODELS_DIR.mkdir(exist_ok=True)
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        # ----- train -----
        model.train()
        t0 = time.time()
        train_total = 0.0
        seen = 0
        for batch in train_loader:
            batch = to_device(batch, DEVICE)
            opt.zero_grad()
            out = model(batch["image"])
            loss, parts = compute_loss(out, batch, bce_keys, bce_mouse, mse)
            loss.backward()
            opt.step()
            train_total += loss.item() * batch["image"].size(0)
            seen += batch["image"].size(0)
        train_avg = train_total / max(seen, 1)

        # ----- val -----
        model.eval()
        val_total = 0.0
        val_seen = 0
        parts_sum = {"keys": 0.0, "mouse": 0.0, "cursor": 0.0}
        with torch.no_grad():
            for batch in val_loader:
                batch = to_device(batch, DEVICE)
                out = model(batch["image"])
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

        # Save checkpoint
        ckpt = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_avg,
            "tracked_keys": train_set.tracked_keys,
        }
        torch.save(ckpt, MODELS_DIR / f"beebot_epoch{epoch:02d}.pt")
        torch.save(ckpt, MODELS_DIR / "beebot_latest.pt")
        if val_avg < best_val_loss:
            best_val_loss = val_avg
            torch.save(ckpt, MODELS_DIR / "beebot_best.pt")
            print(f"          -> new best val loss, saved beebot_best.pt")


if __name__ == "__main__":
    main()
