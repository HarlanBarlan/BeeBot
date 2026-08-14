"""
BeeBotCNN — small feed-forward CNN for Phase 2a imitation learning.

Input:  RGB frame (3 x 135 x 240)
Output: {
    "keys":   logits for each of N tracked keys (BCE loss)
    "mouse":  logits for [left, right] mouse buttons (BCE)
    "cursor": (x, y) normalized to [0, 1] within Roblox window (MSE, masked)
}

Dropout is intentionally on the higher side so the model doesn't over-fit
your exact play patterns — the vision says imitation should teach WHAT to
do, not the PERFECT execution. RL fine-tune later covers execution.
"""

import torch
import torch.nn as nn


def _cnn_backbone():
    """Shared CNN backbone used by both feed-forward and LSTM models."""
    return nn.Sequential(
        nn.Conv2d(3, 32, 5, stride=2, padding=2),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(32, 64, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((4, 8)),
    )


BACKBONE_OUT_DIM = 128 * 4 * 8  # = 4096


class BeeBotCNN(nn.Module):
    """Phase 2a: feed-forward CNN. No memory — every frame independent."""

    def __init__(self, n_keys=75, n_mouse=2, dropout=0.35):
        super().__init__()
        self.backbone = _cnn_backbone()
        self.trunk = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BACKBONE_OUT_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.head_keys = nn.Linear(256, n_keys)
        self.head_mouse = nn.Linear(256, n_mouse)
        self.head_cursor = nn.Linear(256, 2)

    def forward(self, x):
        h = self.trunk(self.backbone(x))
        return {
            "keys": self.head_keys(h),
            "mouse": self.head_mouse(h),
            "cursor": torch.sigmoid(self.head_cursor(h)),
        }


class BeeBotLSTM(nn.Module):
    """Phase 2b: CNN backbone + LSTM over recent frames.
    Adds temporal context so the model can:
      - Remember "I've been walking this direction for a while"
      - Stay still when converting at hive (once it's still, keep still)
      - Fewer per-frame jitters — LSTM smooths action across time

    Training: forward expects (B, T, C, H, W) sequence; returns per-timestep
    outputs. Loss is per-frame across the sequence.

    Inference: forward can accept (B, C, H, W) for single-step + previous
    hidden state; returns per-frame outputs + updated hidden state. Caller
    is responsible for maintaining hidden across frames.
    """

    def __init__(self, n_keys=75, n_mouse=2, hidden_size=256, dropout=0.35):
        super().__init__()
        self.hidden_size = hidden_size
        self.backbone = _cnn_backbone()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BACKBONE_OUT_DIM, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_size, batch_first=True)
        self.head_keys = nn.Linear(hidden_size, n_keys)
        self.head_mouse = nn.Linear(hidden_size, n_mouse)
        self.head_cursor = nn.Linear(hidden_size, 2)

    def forward(self, x, hidden=None):
        """
        x: (B, T, C, H, W) for training/full sequence, OR (B, C, H, W) for single-step inference.
        hidden: optional (h, c) tuple from previous step. If None, LSTM starts fresh.
        Returns (outputs_dict, new_hidden).
        """
        single_step = (x.dim() == 4)
        if single_step:
            x = x.unsqueeze(1)  # (B, 1, C, H, W)

        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        feat = self.backbone(x_flat)           # (B*T, 128, 4, 8)
        emb = self.encoder(feat)                # (B*T, 512)
        emb = emb.view(B, T, -1)                # (B, T, 512)

        lstm_out, new_hidden = self.lstm(emb, hidden)  # (B, T, hidden)

        keys = self.head_keys(lstm_out)         # (B, T, n_keys)
        mouse = self.head_mouse(lstm_out)       # (B, T, 2)
        cursor = torch.sigmoid(self.head_cursor(lstm_out))  # (B, T, 2)

        if single_step:
            return {
                "keys": keys.squeeze(1),
                "mouse": mouse.squeeze(1),
                "cursor": cursor.squeeze(1),
            }, new_hidden
        return {"keys": keys, "mouse": mouse, "cursor": cursor}, new_hidden

    def init_hidden(self, batch_size=1, device="cpu"):
        """Return zero (h, c) tuple to seed a fresh episode at inference."""
        h = torch.zeros(1, batch_size, self.hidden_size, device=device)
        c = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return (h, c)
