# Training Output Cheat Sheet

Everything printed to the terminal during `python -m rl.train_ppo`, what it means, and what to do about it.

---

## Startup (once per training run)

```
[rl] setting up env — Roblox needs to be open and character in a field
[rl] press ESC any time to stop cleanly (bot will release keys and save)
[env] using dxcam for screen capture (GPU-accelerated)   ← or bettercam / mss
[env] popup handler loaded with 2 popup type(s): ['age_dialog', 'age_dialog_v2']
[env] BSS not loaded on startup — launching             ← optional; supervisor
[supervisor] BSS loaded after 0s                        ← optional; supervisor
[resume] loading existing PPO checkpoint: models\beebot_ppo_latest.zip
                                    OR
[fresh] creating new PPO with LSTM CNN warm-start
[warm-start] loaded 8 CNN backbone tensors from beebot_lstm_best.pt
Using cuda device                                    ← device PPO runs on
Wrapping the env in a DummyVecEnv.                   ← SB3 boilerplate
[resume] applied current hyperparameters: ent_coef=0.005, n_steps=4096
Logging to logs\tensorboard\PPO_N                    ← TB folder for this run
[cnn-freeze] CNN backbone FROZEN for first 200,000 steps (currently at 0)
```

**Watch for:**
- `[env] using dxcam` = GPU capture (fast, ~45ms). `bettercam` = same but on older GPUs. `mss` = CPU fallback (slow, ~200ms+, expect 1-2 fps).
- `popup handler loaded with N popup type(s)` — should list your snipped popups (`age_dialog`, `age_dialog_v2`, etc). `0 popup type(s)` means the handler is dormant; snip templates to enable.
- `[resume]` vs `[fresh]` — resume loads existing PPO weights. Fresh starts from imitation warm-start only (PPO policy/value heads reinitialized). Fresh happens after obs-space changes (e.g., HUD_DIM change) or when `beebot_ppo_latest.zip` is missing.
- `Using cuda device` OR `Using cpu device` — should match your hardware
- `[warm-start] loaded 8 CNN backbone tensors` — imitation warm-start worked. If missing or "no matching keys", first training hours will be much slower.
- `[cnn-freeze] CNN backbone FROZEN for first 200,000 steps` — CNN gradient updates are suppressed for the first 200k steps to protect imitation features from PPO gradient noise. Log will announce unfreeze at 200k.
- `PPO_N` — number increments per run. Old runs preserved.

**Ignore:**
- `UserWarning: You are trying to run PPO on the GPU...` — misleading, we use CNN via custom features extractor
- `SyntaxWarning: "\\."` — old docstring warning, harmless
- bettercam `AttributeError: 'BetterCam' object has no attribute 'is_capturing'` at shutdown — cosmetic garbage-collector complaint from a half-initialized bettercam object; doesn't affect training

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
| `X.Xfps` | Live steps/sec (last 30 sec) | **Target 4-6 fps** on desktop w/ dxcam. Lower = capture path is CPU or something else is slow |
| `pollen=X%` | HUD's pollen bar reading | Should vary as bot gathers/converts. `?` = OCR failed. **Values >100% are legitimate** (overfill mechanics), don't treat as bug |
| `honey=X` | HUD's honey reading | Should stay stable or grow. `?` = OCR failed. Very large single-tick jumps (10×/100×/1000× ratios) are OCR errors and get rejected by the reward function |
| `(+X this session)` | Honey earned since env init | Should track real bot productivity. **Known issue:** this display can lock onto an OCR-misread session-start value and show a garbage constant like `+5,238,643`. Real reward is unaffected — check `total_reward` and `ep_rew_mean` for actual signal |
| `total_reward=X` | Per-episode accumulated shaped reward (resets every 1024 steps) | Small-magnitude fluctuation (~-1 to +5) is normal. Spikes to hundreds indicate a reward outlier the guards missed |

---

## PPO iteration table (every 4096 steps = N_STEPS)

```
| time/              |         |
|    fps             | 4       |
|    iterations      | 5       |
|    time_elapsed    | 4296    |
|    total_timesteps | 20480   |
```

- `fps` — SB3's rolling average (rounds down; usually shows a bit lower than env's live fps)
- `iterations` — count of PPO gradient updates (N_STEPS=4096 per iteration). Meaningful learning starts around 20-40 iterations = ~80-160k timesteps.
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
| rollout/       |          |
|   ep_len_mean  | 1.02e+03 |
|   ep_rew_mean  | 0.112    |
```

- `ep_len_mean` — episode length. Our env has no natural episode end, so this equals `EPISODE_LENGTH_STEPS` (1024). Constant across the run.
- **`ep_rew_mean`** — **MASTER LEARNING METRIC.** Should trend up over hours of training.

**What healthy `ep_rew_mean` values look like (post-quest-removal, 2026-08-15):**
- First iteration or two: any value from -1 to +1 is normal (policy just started)
- After 20-40 iterations: should trend up into small positive range (~0 to +5)
- After 100+ iterations of good farming: could climb to +5 to +20 range
- **Sustained values above +50 are a RED FLAG** — historically that indicated reward-function pollution from the (now-removed) quest reward channel or an OCR bug slipping past the outlier guards. If you see this, check that `value_loss` isn't in the thousands and `explained_variance` isn't stuck at 0.

---

## Popup handler output

```
[popup] dismissed 'age_dialog' (detector conf 0.69, close-button conf 1.00) — clicked at (23, 43)
```
Handler detected a known popup, found its close button, and clicked to dismiss. Perfect state — the popup blocked farming for maybe 3 sec instead of 60+.

```
[popup] 'age_dialog' detected (conf 0.69) but couldn't locate close button (best conf 0.49) — consider re-snipping popup_age_dialog_close.png
```
Detector matched but close-button template failed threshold. Usually means the close-button template was snipped at a different resolution than current Roblox window. Fix: re-snip the close-button template at current resolution. Meanwhile the popup persists and dialogue rescue will eventually fire.

## Dialogue rescue output

```
[env t=8451] dialogue-rescue: burst 25 clicks at (863,637) (match conf 1.00) — E suppressed for 50 steps
```
One of the `dialogue_continue_*.png` templates matched at high confidence — bot burst-clicked to advance the dialogue and suppressed the E key for 50 steps so the policy can walk out of NPC range before re-triggering.

```
[env t=8513] dialogue-rescue: no template matched (best dialogue_continue_mother_bear.png conf 0.36 — need >0.65). Consecutive failures: 1
```
No template matched above threshold. If this repeats, bot may be stuck in an unfamiliar UI/dialogue. Consider snipping a new template for the specific bear or menu.

```
[env t=11006] BLIND rescue — clicked at [(863, 637), (865, 636)] (no template match but stall persistent). Also suppressed E.
```
After 5 consecutive failed template matches, rescue falls back to clicking previously-successful positions. Not reliable but sometimes breaks stalls in never-templated UI.

**Watch for:** dozens of consecutive rescue failures with the same low-confidence template — that's the pathological infinite-loop pattern that used to kill training runs. Post-popup-handler, the age dialog case is handled fast-path so this pattern is rare, but if it appears repeatedly you have a NEW popup type to template.

## Reward outlier protection output

```
[reward] accepting large POS delta after 15 consecutive reads: honey 3,935,310 -> 4,056,251 (likely session-hop / snapshot — re-baselining)
```
15+ consecutive readings crossed the outlier threshold in the same direction. Reward function accepts them as real state change (session-hop, big spend, real convert burst) and re-baselines. Cooldown of 60s prevents ping-pong.

```
[reward] POS delta hit threshold but REJECTED — within cooldown of last re-baseline (12.3s ago). Suspecting OCR bounce, not real state change.
```
Second re-baseline attempt happened within 60s of the last one — classic OCR-bounce pattern (persistent misread creates baseline A, real values coming back look like huge deltas creating baseline B, ad infinitum). Guard rejects and resets the counter.

**Watch for:** more than 2-3 re-baselines per hour. Occasional ones are normal (real convert bursts, shop spending). Frequent ones = OCR is unreliable at current window resolution → consider re-snipping `honey_display.png`.

## Manual pause output

```
[env t=1876] MANUAL PAUSE (holding F8) — bot released; drive manually
[env t=1923] resuming bot control
```
F8 hotkey — bot releases all keys, releases cursor clip. Rollout keeps running (obs/reward/value estimates still computed) so the pause doesn't corrupt training. Useful for manually walking the bot out of a stuck spot without stopping the run.

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
| `[env] using mss` on desktop | dxcam/bettercam failed to init | Update GPU driver; if hardware limit, accept mss + shrink Roblox window |
| fps < 3 for extended periods | Capture path is CPU (mss), OR HUD OCR not rate-limited | Check startup line for capture backend; check `[timing]` capture value |
| `(+X this session)` stuck at 0 for 30+ min | Bot not in field OR bot stuck in menu | Check for cascading dialogue-rescue failures; manually rescue via F8 |
| `(+X this session)` shows huge constant (millions from step 2+) | Cosmetic: env's session-start tracker latched onto OCR misread | Ignore the display; check `total_reward` and `ep_rew_mean` for real signal |
| `honey` jumps 10×/100×/1000× then snaps back | OCR decimal-shift misread | Handled automatically; if frequent, re-snip `honey_display.png` |
| `total_reward` diving into extended negatives | Stall penalty dominating | Bot stuck — rescue via F8; check for cascading dialogue-rescue failures |
| `value_loss` in thousands + `ep_rew_mean` > 50 | Reward outlier poisoning value function (quest reward channel was the historical culprit; now removed) | Diagnose — check for new OCR outlier not caught by guards |
| `ep_rew_mean` flat for 100+ iterations | Learning stalled | Bump entropy coefficient, verify observation isn't degenerate |
| `approx_kl` near 0 | Policy frozen | Check entropy coefficient not zero |
| `explained_variance` stays near 0 or negative for many iterations | Value function broken | Something wrong with reward signal — check for outliers |
| Dozens of consecutive dialogue-rescue failures at same coords | Bot stuck in unfamiliar UI (age dialog, purchase confirm, etc) | Snip a popup handler template for that dialog type |
| Constant supervisor recovery | Roblox crashing repeatedly | Check for multi-instance mutex issue, driver crash, memory pressure |

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
- HUD readers: `hud/reader.py`, `hud/pollen_ocr.py`, `hud/honey_ocr.py`, `hud/buff_classifier.py`, `hud/popup_handler.py`
- Imitation subpackage: `imitation/` (dataset, model, train, train_lstm, play, record_*)
- Shared IO utilities: `common/roblox_window.py`, `common/robo_input.py`
- Dev/user tools: `scripts/` (snip_template, test_gpu, fetch_wiki_buff_icons)
- Docs: `docs/` (COMMANDS.md, EXPERIMENT_LOG.md, TRAINING_CHEATSHEET.md, Personal_Documentation.txt)
- Checkpoint (auto-saved): `models/beebot_ppo_latest.zip`
- Training log: `logs/training/training_*.txt` (if you used `Tee-Object`)
- TensorBoard: `logs/tensorboard/PPO_N/`
- View TB from main PC: `http://<rdp-machine>:6006/` (after starting tensorboard on RDP with `--bind_all`)

---

## Common commands

```powershell
# Start training with logging (RDP session — swap .venv-rdp for .venv on main)
$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
.\.venv-rdp\Scripts\python.exe -m rl.train_ppo *>&1 | Tee-Object -FilePath "logs\training\training_$stamp.txt"

# Live-tail from main PC
Get-Content C:\ClaudeWorkspace\BeeBot\logs\training\training_*.txt -Wait -Tail 100

# TensorBoard (run on RDP, view from main PC)
.\.venv-rdp\Scripts\python.exe -m tensorboard.main --logdir logs/tensorboard --bind_all
# then browse http://localhost:6006/

# Snip templates (paths relative to BeeBot root)
.\.venv\Scripts\python.exe -m scripts.snip_template hud/probes/pollen_bar_frame
.\.venv\Scripts\python.exe -m scripts.snip_template hud/probes/popup_age_dialog

# Test HUD readers
.\.venv\Scripts\python.exe -m hud.pollen_ocr
.\.venv\Scripts\python.exe -m hud.honey_ocr
.\.venv\Scripts\python.exe -m hud.buff_classifier
.\.venv\Scripts\python.exe -m hud.popup_handler

# Check GPU utilization
nvidia-smi

# Check Roblox processes
Get-Process | Where-Object { $_.Name -match "Roblox" } | Format-Table Name, Id, StartTime

# Kill only Freddy's Roblox (leave main's alone) — run from Freddy session
Get-Process | Where-Object { $_.Name -match "Roblox" -and $_.StartTime -ne $null } | Stop-Process -Force
```
