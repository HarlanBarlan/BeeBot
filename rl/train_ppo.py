r"""
PPO training for the RL fine-tune (Phase 3a).

Uses stable-baselines3 PPO. Runs one live env against Roblox — every
step is 1/FPS seconds real-time, so training is measured in HOURS, not
minutes. Bot needs to be:
  - Roblox open, logged into the alt account (Fredrick)
  - Standing somewhere useful (a field is fine — model reactions decide)
  - No user interaction with keyboard/mouse during training (bot drives)
  - Press ESC to stop cleanly

Usage:
  .\.venv\Scripts\python.exe -m rl.train_ppo

Outputs:
  models/beebot_ppo_latest.zip  — SB3 checkpoint (loadable)
  logs/tensorboard/             — TB logs (run tensorboard --logdir logs/tensorboard)

STATUS: this is Phase 3a scaffold. Training from scratch (no warm-start
from imitation LSTM yet — that integration is complex with SB3's policy
interface and is a Phase 3a.5 task). Bot will initially act random-ish
and slowly learn what farming looks like via reward signal.
"""

import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
import keyboard as kb_lib

from .env import BSSEnv, release_cursor_clip
from .policy import BSSMultiInputFeatures, load_backbone_from_lstm_ckpt


class StopOnKeyCallback(BaseCallback):
    """Halt training (not just the episode) when the user presses a key.
    SB3's `_on_step` returning False stops the outer learn() loop."""
    def __init__(self, key="esc", verbose=0):
        super().__init__(verbose)
        self.key = key

    def _on_step(self):
        if kb_lib.is_pressed(self.key):
            print(f"\n[rl] {self.key.upper()} pressed — stopping training after this rollout")
            return False
        return True


MODELS_DIR = Path("models")
LOGS_DIR = Path("logs/tensorboard")
TOTAL_TIMESTEPS = 100_000       # ~2.5 hours at 10 FPS
SAVE_EVERY_N_STEPS = 5000
LEARNING_RATE = 3e-4
# Bigger batch: GPU is at 15% because it's underused. 256 gets better GPU
# utilization during backprop and improves gradient signal stability.
# Cap at n_steps/2 so we have at least 2 minibatches per epoch.
BATCH_SIZE = 256
N_STEPS = 1024                  # bigger rollout too — feeds PPO more experience
                                # per update; better learning per iteration

# Warm-start CNN backbone from the imitation LSTM checkpoint if it exists.
# Set to None to disable and train the CNN from scratch.
WARM_START_CKPT = MODELS_DIR / "beebot_lstm_best.pt"


def main():
    print("[rl] setting up env — Roblox needs to be open and character in a field")
    print("[rl] press ESC any time to stop cleanly (bot will release keys and save)")
    time.sleep(3)

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)

    env = BSSEnv()

    # Multi-input feature extractor: CNN(image) + MLP(hud scalars) -> concat.
    # See rl/policy.py docstring for the rationale — TL;DR the value function
    # can't reliably parse pollen%/honey from downsized pixels while ALSO
    # fitting sparse reward, so we feed the critic the ground-truth HUD
    # scalars alongside the pixels.
    policy_kwargs = dict(
        features_extractor_class=BSSMultiInputFeatures,
        features_extractor_kwargs=dict(features_dim=512),
    )

    # GPU chosen: our real bottleneck was pydirectinput.PAUSE (fixed elsewhere),
    # not device choice. CPU was maxing out the machine and lagging the host.
    # GPU offloads CNN work to the 1080 Ti, freeing CPU for other tasks + your
    # main account's Roblox. If FPS drops significantly with GPU, flip back
    # to "cpu" — it means single-env inference overhead outweighs CNN speedup.
    device = "cuda"

    # If a previous PPO checkpoint exists, RESUME from it (preserves learning
    # across sessions and machines). Otherwise create a fresh PPO and warm-
    # start its CNN backbone from the imitation LSTM.
    #
    # IMPORTANT: checkpoints saved before the HUD-in-observation change
    # (2026-08-14) used a Box observation space and MlpPolicy — those will
    # fail to load into the new MultiInputPolicy. Rename or delete the old
    # checkpoint before first run:
    #   mv models/beebot_ppo_latest.zip models/beebot_ppo_pretraining.zip
    # A pretraining archive is preserved so we can compare metrics later.
    resume_path = MODELS_DIR / "beebot_ppo_latest.zip"
    if resume_path.exists():
        print(f"[resume] loading existing PPO checkpoint: {resume_path}")
        try:
            model = PPO.load(
                str(resume_path),
                env=env,
                device=device,
                tensorboard_log=str(LOGS_DIR),
            )
        except (ValueError, RuntimeError, KeyError) as e:
            # Common cause: obs space changed between checkpoint save and now.
            # Fall through to fresh-start so training can proceed; the old
            # checkpoint is preserved on disk for manual recovery.
            print(f"[resume] CHECKPOINT INCOMPATIBLE with current env "
                  f"(likely obs-space change): {e}")
            print(f"[resume] falling back to fresh PPO. If you want to keep the "
                  f"old checkpoint's learning, rename it now and restart:")
            print(f"    mv {resume_path} {MODELS_DIR}/beebot_ppo_pretraining.zip")
            resume_path = None
    if not resume_path or not resume_path.exists():
        print("[fresh] creating new PPO with LSTM CNN warm-start")
        model = PPO(
            "MultiInputPolicy",           # Dict obs {image, hud} -> features extractor
            env,
            policy_kwargs=policy_kwargs,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            ent_coef=0.02,                # entropy bonus — will lower this in a later
                                          # sequential step once critic is healthy
            verbose=1,
            tensorboard_log=str(LOGS_DIR),
            device=device,
        )
        # Warm-start the CNN backbone from imitation checkpoint if available.
        # Only the CNN backbone weights transfer — the HUD MLP, combine layer,
        # and policy/value heads all start fresh. That's fine: the imitation
        # model never saw HUD scalars either.
        if WARM_START_CKPT is not None:
            load_backbone_from_lstm_ckpt(model.policy, WARM_START_CKPT)

    checkpoint_cb = CheckpointCallback(
        save_freq=SAVE_EVERY_N_STEPS,
        save_path=str(MODELS_DIR),
        name_prefix="beebot_ppo",
    )
    stop_cb = StopOnKeyCallback(key="esc")

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=[checkpoint_cb, stop_cb])
    except KeyboardInterrupt:
        print("[rl] KeyboardInterrupt — saving and stopping")
    finally:
        model.save(str(MODELS_DIR / "beebot_ppo_latest"))
        env.close()
        release_cursor_clip()  # belt-and-suspenders in case env.close didn't run
        print("[rl] stopped — saved beebot_ppo_latest.zip")


if __name__ == "__main__":
    main()
