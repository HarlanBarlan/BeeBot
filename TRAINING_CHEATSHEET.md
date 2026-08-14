# Training Output Cheat Sheet

Everything printed to the terminal during `python -m rl.train_ppo`, what it means, and what to do about it.

---

## Startup (once per training run)

```
[rl] setting up env — Roblox needs to be open and character in a field
[rl] press ESC any time to stop cleanly (bot will release keys and save)
Using cpu device                                    ← device PPO runs on
Wrapping the env with a `Monitor` wrapper           ← SB3 boilerplate
Wrapping the env in a DummyVecEnv.                  ← SB3 boilerplate
[warm-start] loaded 8 CNN backbone tensors from beebot_lstm_best.pt   ← imitation weights loaded
Logging to logs\tensorboard\PPO_N                   ← TB folder for this run
```

**Watch for:**
- `Using cuda device` OR `Using cpu device` — should match what you set in train_ppo.py
- `[warm-start] loaded 8 CNN backbone tensors` — imitation warm-start worked. If it says "no matching keys" or "no checkpoint", warm-start failed and training starts from scratch (~10x slower).
- `PPO_N` — the number increments per run (PPO_1, PPO_2...). Old runs preserved.

**Ignore:**
- `UserWarning: You are trying to run PPO on the GPU...` — misleading, we use CNN via custom features extractor
- `SyntaxWarning: "\\."` — old docstring warning, harmless

---

## Timing lines (every 100 env steps)

```
[timing] step 99: action=5ms sleep=100ms capture=41ms
```

**Fields:**
- `action` — time spent applying keys/mouse per step. **Healthy: under 20ms.** If 500-1000ms, `pydirectinput.PAUSE` is wrong (fixed with `pydirectinput.PAUSE = 0`).
- `sleep` — intentional 100ms tick delay. Should always be ~100ms.
- `capture` — screen grab + region lookup + HUD (when it runs). **Healthy: 30-60ms.** If 200ms+ every step, region lookup is slow OR HUD OCR isn't rate-limited.

**Total step time = action + sleep + capture + SB3 forward pass overhead.** Expected ~150-250ms per step = 4-7 fps.

---

## Env status lines (every ~30 sec)

```
[env t=  1500 4.5fps] pollen=42%  honey=15,234  (+2,847 this session)  total_reward=4.32
```

| Field | Meaning | Healthy signal |
|---|---|---|
| `t=` | Step count since env started | Should climb steadily |
| `X.Xfps` | Live steps/sec (last 30 sec) | **Target 5-10 fps**. Lower = something slow. Higher than 10 = tick sleep broken |
| `pollen=X%` | HUD's pollen bar reading | Should vary as bot gathers/converts. `?` = OCR failed |
| `honey=X` | HUD's honey reading | Should stay stable or grow. `?` = OCR failed |
| `(+X this session)` | **KEY METRIC.** Honey earned since env init | Positive & growing = bot productive. `0` for hours = not learning. |
| `total_reward=X` | Cumulative shaped reward | Positive trend = healthy |

---

## PPO iteration table (every 512 steps)

```
| time/              |         |
|    fps             | 3       |
|    iterations      | 5       |
|    time_elapsed    | 1029    |
|    total_timesteps | 2560    |
```

- `fps` — SB3's rolling average (rounds down; usually shows a bit lower than env's live fps)
- `iterations` — count of PPO gradient updates. Meaningful learning starts around 20-40.
- `time_elapsed` — total seconds
- `total_timesteps` — cumulative env.step calls

After iteration 2+:

```
| train/                  |         |
|    approx_kl            | 0.033   |
|    clip_fraction        | 0.249   |
|    clip_range           | 0.2     |
|    entropy_loss         | -31.2   |
|    explained_variance   | -1.49   |
|    learning_rate        | 0.0003  |
|    loss                 | -0.72   |
|    n_updates            | 10      |
|    policy_gradient_loss | -0.0609 |
|    std                  | 1       |
|    value_loss           | 0.035   |
```

| Metric | Healthy range | What it tells you |
|---|---|---|
| `approx_kl` | **0.005 - 0.05** | Policy update magnitude. Near 0 = frozen. Over 0.1 = diverging |
| `clip_fraction` | **0.05 - 0.4** | Fraction of updates hitting PPO safety clip. Higher = clip limiting learning |
| `clip_range` | 0.2 (fixed) | PPO clip parameter |
| `entropy_loss` | Starts near -30, rises slowly | Negative entropy of action distribution. Less negative over time = policy committing |
| `explained_variance` | **Should climb to 0.3+** after 30+ iters | How well value function predicts returns. Near 0 = predicting mean. Negative = worse than mean |
| `learning_rate` | 0.0003 (fixed) | Optimizer step size |
| `loss` | Trends down | Combined actor+critic loss |
| `n_updates` | ~10 per iteration | Total gradient steps |
| `policy_gradient_loss` | Small negative | Actor loss. Direction > magnitude |
| `std` | Starts at 1, shrinks | Action distribution spread. Shrinks as policy commits |
| `value_loss` | High early, drops over hours | Critic loss. Spikes = value function reacting to reward outliers |

After rollout episodes finish:

```
| rollout/       |      |
|   ep_len_mean  | 512  |
|   ep_rew_mean  | 2.87 |
```

- `ep_len_mean` — episode length. Our env has no episode end, so this equals `n_steps` (512).
- **`ep_rew_mean`** — **MASTER LEARNING METRIC.** Should trend up over hours of training.

---

## Supervisor / recovery output (rare)

```
[env] health check: BSS gone — recovering
[supervisor] BSS not detected (attempt 1/3) — launching
[supervisor] launched via browser: https://www.roblox.com/games/start?placeId=1537690962
[supervisor] waiting up to 180s for BSS to load…
[supervisor] BSS loaded after 42s
```

Fires when Roblox window disappears mid-training. Gives up after 3 failed attempts. If gives up, training continues but bot can't play — need manual relaunch.

---

## What "good" looks like after 2-4 hours

- `fps` steady at 4-8
- `[env t=...]` shows `(+X this session)` growing continuously
- `total_reward` climbing (not diving negative for extended stretches)
- `ep_rew_mean` visibly climbing across iterations
- `explained_variance` climbing above 0.1 after 20+ iterations
- `entropy_loss` slowly rising (getting less negative)
- No repeated supervisor recovery attempts

## What "bad" looks like

| Symptom | Likely cause | Fix |
|---|---|---|
| fps < 3 for extended periods | Env bottleneck | Check `[timing]` for slow phase |
| `(+X this session)` stuck at 0 for 30+ min | Bot not in field | Manually rescue |
| `honey` jumps massively then normalizes | OCR outlier | Fixed with `MAX_HONEY_DELTA_PER_TICK` guard |
| `total_reward` diving fast into negatives | Stall penalty dominating | Bot really stuck — rescue |
| `value_loss` explodes to thousands | Reward outlier poisoned value function | Reduce learning rate or wait for buffer to recycle |
| `ep_rew_mean` flat for 100+ iterations | Learning stalled | Bump entropy coefficient, add exploration |
| `approx_kl` near 0 | Policy frozen | Check entropy coefficient not zero |
| `explained_variance` stays negative | Value function broken | Something wrong with reward |
| Constant supervisor recovery | Roblox unstable | Multi-instance mutex issue |

---

## What to do when training seems off

**First: don't panic.** Early RL is noisy. Iteration-to-iteration numbers swing wildly. Trends over hours matter, not single readings.

**Second: check the log tail** on your main PC:
```powershell
Get-Content C:\ClaudeWorkspace\BeeBot\logs\training.log -Wait -Tail 100
```

**Third: intervention ladder** (least → most disruptive):
1. **Wait 30 min** — most weirdness self-corrects
2. **Rescue bot manually** — walk out of a stuck spot, don't stop training
3. **Restart training** — fresh policy state, keeps checkpoints
4. **Reset from best checkpoint** — load `models/beebot_ppo_latest.zip`
5. **Send Claude the last 200 lines of log** — diagnose deeper issues

---

## Quick reference: files and paths

- Training script: `rl/train_ppo.py`
- Env definition: `rl/env.py`
- Reward function: `rl/reward.py`
- Checkpoint (auto-saved): `models/beebot_ppo_latest.zip`
- Training log: `logs/training.log` (if you used `Tee-Object`)
- TensorBoard: `logs/tensorboard/PPO_N/`
- View TB from main PC: `http://localhost:6006/` (after starting tensorboard on RDP)

---

## Common commands

```powershell
# Start training with logging
.\.venv-rdp\Scripts\python.exe -m rl.train_ppo *>&1 | Tee-Object -FilePath logs\training.log

# Live-tail from main PC
Get-Content C:\ClaudeWorkspace\BeeBot\logs\training.log -Wait -Tail 100

# TensorBoard (run on RDP, view from main PC)
.\.venv-rdp\Scripts\python.exe -m tensorboard.main --logdir logs/tensorboard --bind_all
# then browse http://localhost:6006/

# Check GPU utilization
nvidia-smi

# Check Roblox processes
Get-Process | Where-Object { $_.Name -match "Roblox" } | Format-Table Name, Id, StartTime

# Kill only Freddy's Roblox (leave main's alone)
Get-Process | Where-Object { $_.Name -match "Roblox" -and $_.StartTime -ne $null } | Stop-Process -Force
```
