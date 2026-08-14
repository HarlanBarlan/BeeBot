"""
Custom PPO policy pieces for BSS:

- BSSCnnFeatures: SB3 feature extractor using the same CNN backbone as
  our imitation model. Enables warm-starting the CNN from the LSTM
  checkpoint (saves 10-20+ hours of "learn what a flower looks like"
  training).

- load_backbone_from_lstm_ckpt: helper that loads matching CNN backbone
  weights from a Phase 2b LSTM checkpoint into this extractor.
"""

from pathlib import Path
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from model import _cnn_backbone, BACKBONE_OUT_DIM


class BSSCnnFeatures(BaseFeaturesExtractor):
    """CNN feature extractor for pixel observations.
    Architecture matches the imitation model's backbone so we can warm-start
    from a Phase 2b LSTM checkpoint's CNN weights.
    """

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        self.cnn = _cnn_backbone()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BACKBONE_OUT_DIM, features_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs):
        # obs comes in as (B, 3, H, W), already normalized to [0, 1] by env
        return self.head(self.cnn(obs))


def load_backbone_from_lstm_ckpt(policy, ckpt_path):
    """Load the CNN backbone weights from a saved LSTM imitation checkpoint
    into a PPO policy's feature extractor.

    Only the `backbone.*` layers are copied — the encoder / LSTM / heads
    are RL-specific and stay initialized fresh (RL will learn its own
    policy from those visual features).

    Returns number of tensors loaded, or 0 if the checkpoint doesn't
    contain matching keys.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print(f"[warm-start] no checkpoint at {ckpt_path} — skipping warm-start")
        return 0

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    src = ckpt.get("model_state", ckpt)
    # Extract only the backbone layers, stripping the "backbone." prefix
    backbone_state = {
        k[len("backbone."):]: v
        for k, v in src.items()
        if k.startswith("backbone.")
    }
    if not backbone_state:
        print(f"[warm-start] no 'backbone.*' keys in {ckpt_path} — skipping")
        return 0

    extractor = policy.features_extractor
    try:
        extractor.cnn.load_state_dict(backbone_state, strict=True)
    except RuntimeError as e:
        print(f"[warm-start] shape mismatch — skipping: {e}")
        return 0

    print(f"[warm-start] loaded {len(backbone_state)} CNN backbone tensors from {ckpt_path.name}")
    return len(backbone_state)
