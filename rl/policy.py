"""
Custom PPO policy pieces for BSS:

- BSSCnnFeatures: legacy pixel-only feature extractor (kept for reference /
  checkpoint compat only — no longer used by train_ppo.py).

- BSSMultiInputFeatures: current feature extractor. Takes a Dict observation
  {image: pixels, hud: scalar vector}, runs image through CNN (warm-startable
  from imitation LSTM backbone), runs hud through small MLP, concatenates
  and projects to features_dim.

  Rationale (from 2026-08-14 research pass): giving the critic ground-truth
  scalar HUD state instead of forcing it to visually parse pixels is the
  single highest-leverage change for sparse-reward continuous-play RL.
  Andrychowicz 2020 "What Matters in On-Policy RL" ranks accurate state
  features among top-5 wins across ~50 design axes.

- load_backbone_from_lstm_ckpt: helper that loads matching CNN backbone
  weights from a Phase 2b LSTM checkpoint into either extractor's CNN.
"""

from pathlib import Path
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from imitation.model import _cnn_backbone, BACKBONE_OUT_DIM


class BSSCnnFeatures(BaseFeaturesExtractor):
    """LEGACY pixel-only extractor. Kept so old checkpoints can still be
    loaded for archival / comparison. New training uses BSSMultiInputFeatures.
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
        return self.head(self.cnn(obs))


class BSSMultiInputFeatures(BaseFeaturesExtractor):
    """Dict-observation extractor: CNN(image) + MLP(hud) -> concat -> project.

    Architecture rationale:
    - CNN path matches the imitation backbone so warm-start works unchanged
      (`load_backbone_from_lstm_ckpt` loads self.cnn from the imitation ckpt).
    - HUD path is a tiny 2-layer MLP — HUD vector is only ~8 dims, so
      capacity there should be small (over-parameterizing on 8 inputs
      overfits fast on our low sample rate).
    - Late-concat + linear projection to features_dim keeps the
      downstream SB3 policy/value heads unchanged.

    features_dim ends up as (CNN_HEAD_DIM + HUD_HEAD_DIM) -> project(features_dim).
    """

    CNN_HEAD_DIM = 512
    HUD_HEAD_DIM = 32

    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)
        assert "image" in observation_space.spaces and "hud" in observation_space.spaces, \
            "BSSMultiInputFeatures expects Dict({image, hud}) observation space"

        self.cnn = _cnn_backbone()
        self.cnn_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BACKBONE_OUT_DIM, self.CNN_HEAD_DIM),
            nn.ReLU(inplace=True),
        )

        hud_input_dim = observation_space["hud"].shape[0]
        self.hud_mlp = nn.Sequential(
            nn.Linear(hud_input_dim, self.HUD_HEAD_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(self.HUD_HEAD_DIM, self.HUD_HEAD_DIM),
            nn.ReLU(inplace=True),
        )

        self.combine = nn.Linear(self.CNN_HEAD_DIM + self.HUD_HEAD_DIM, features_dim)

    def forward(self, obs):
        # SB3 gives us obs as a dict of tensors, images already normalized to [0,1]
        img = obs["image"]
        hud = obs["hud"]
        img_feat = self.cnn_head(self.cnn(img))
        hud_feat = self.hud_mlp(hud)
        return self.combine(torch.cat([img_feat, hud_feat], dim=-1))


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
