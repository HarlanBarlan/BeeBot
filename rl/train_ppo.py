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
from stable_baselines3.common.monitor import Monitor
import keyboard as kb_lib

from .env import BSSEnv, release_cursor_clip
from .policy import BSSMultiInputFeatures, load_backbone_from_lstm_ckpt


# Freeze the CNN backbone for the first N PPO steps so early gradient noise
# doesn't destroy the imitation-learned visual features. Research pass
# (Andrychowicz 2020, PIRLNav 2023) explicitly recommends freeze-then-
# unfreeze for BC-warm-started policies. When we unfreeze, the CNN
# starts training too but at the same LR as the rest of the network.
CNN_FREEZE_STEPS = 200_000


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


class CnnFreezeCallback(BaseCallback):
    """Freeze the CNN backbone for the first `unfreeze_at` timesteps, then
    unfreeze it. Protects the imitation-warm-started visual features from
    early PPO gradient noise.

    Uses self.num_timesteps (SB3's global step counter across resumes) —
    unfreezes based on total accumulated steps, not per-session.
    """
    def __init__(self, unfreeze_at=CNN_FREEZE_STEPS, verbose=0):
        super().__init__(verbose)
        self.unfreeze_at = unfreeze_at
        self._cnn = None
        self._frozen = False
        self._unfroze = False

    def _on_training_start(self):
        # Grab the CNN reference from the (already-loaded) policy
        try:
            self._cnn = self.model.policy.features_extractor.cnn
        except AttributeError:
            print("[cnn-freeze] policy has no features_extractor.cnn — skipping freeze")
            return
        # Only freeze if we haven't already passed the unfreeze point
        # (relevant when resuming from a checkpoint saved past the threshold)
        if self.model.num_timesteps < self.unfreeze_at:
            for p in self._cnn.parameters():
                p.requires_grad = False
            self._frozen = True
            print(f"[cnn-freeze] CNN backbone FROZEN for first {self.unfreeze_at:,} steps "
                  f"(currently at {self.model.num_timesteps:,})")
        else:
            print(f"[cnn-freeze] already past {self.unfreeze_at:,} steps "
                  f"(at {self.model.num_timesteps:,}) — CNN remains trainable")
            self._unfroze = True

    def _on_step(self):
        if not self._unfroze and self._frozen and self.num_timesteps >= self.unfreeze_at:
            for p in self._cnn.parameters():
                p.requires_grad = True
            self._unfroze = True
            print(f"[cnn-freeze] step {self.num_timesteps:,} — CNN backbone UNFROZEN")
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
# Big rollouts. At ~5 fps, 1024 steps was only ~3.4 min of gameplay —
# often ONE conversion event per rollout, which is not enough for the
# critic to fit sparse-reward targets. 4096 = ~14 min = 3-4 conversion
# events per rollout, giving the value function real data to learn from.
# (Research pass on sparse-reward PPO explicitly recommended this bump.)
N_STEPS = 4096
# Entropy bonus. Dropped from 0.02 -> 0.005 because the higher value was
# actively destroying the imitation warm-start — entropy_loss was
# climbing (more exploration) over the run instead of falling, and the
# policy was wandering into dead map areas that the imitation model would
# not have gone to. Standard warm-start range is 0.001-0.005.
ENT_COEF = 0.005

# Warm-start CNN backbone from the imitation LSTM checkpoint if it exists.
# Set to None to disable and train the CNN from scratch.
WARM_START_CKPT = MODELS_DIR / "beebot_lstm_best.pt"


def main():
    print("[rl] setting up env — Roblox needs to be open and character in a field")
    print("[rl] press ESC any time to stop cleanly (bot will release keys and save)")
    time.sleep(3)

    MODELS_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)

    # Wrap env in Monitor for SB3's per-episode reward tracking.
    # PPO auto-wraps in DummyVecEnv internally, but only auto-adds Monitor
    # if the env isn't already a VecEnv — since we're not pre-wrapping in
    # DummyVecEnv anymore, we still explicitly Monitor-wrap so ep_rew_mean
    # / ep_len_mean show up in the training logs reliably.
    #
    # VecNormalize was tried on 2026-08-14 and removed after showing
    # regression vs the pre-VecNormalize run (ep_rew_mean +3→+1.7 became
    # -1.98→-1.57, EV -1.27→+0.156 became -4.24). Rolling std of returns
    # is unstable for our sparse-reward env — value function chases a
    # moving target. Kept the Monitor addition (real fix), reverted the
    # VecNormalize part.
    env = BSSEnv()
    env = Monitor(env)

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
            # Force the current hyperparameters onto the loaded model —
            # otherwise the checkpoint's original ent_coef / n_steps stay
            # in effect and our recent tuning changes are ignored.
            model.ent_coef = ENT_COEF
            model.n_steps = N_STEPS
            print(f"[resume] applied current hyperparameters: "
                  f"ent_coef={ENT_COEF}, n_steps={N_STEPS}")
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
            ent_coef=ENT_COEF,
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
    freeze_cb = CnnFreezeCallback(unfreeze_at=CNN_FREEZE_STEPS)

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS,
                    callback=[checkpoint_cb, stop_cb, freeze_cb])
    except KeyboardInterrupt:
        print("[rl] KeyboardInterrupt — saving and stopping")
    finally:
        model.save(str(MODELS_DIR / "beebot_ppo_latest"))
        env.close()
        release_cursor_clip()  # belt-and-suspenders in case env.close didn't run
        print("[rl] stopped — saved beebot_ppo_latest.zip")


if __name__ == "__main__":
    main()
