# BeeBot Commands Reference

Everything you might need to run, tune, debug, or move between machines. Copy-paste ready.

---

## Which machine is which

| Machine | User account | Python venv | Purpose |
|---|---|---|---|
| **Main desktop** | `harla` | `.\.venv\Scripts\python.exe` | Primary dev machine |
| **Main desktop — RDP** | `Freddy` (second Windows user, RDP session) | `.\.venv-rdp\Scripts\python.exe` | Isolated training session on the same physical PC as main desktop, without interrupting your main account's use |
| **Laptop** | `harla` | `.\.venv\Scripts\python.exe` | Portable training machine |

All commands below use `.\.venv\...` — swap to `.\.venv-rdp\...` if you're in the Freddy RDP session.

All paths assume you're in `C:\ClaudeWorkspace\BeeBot\`. If unsure:
```powershell
cd C:\ClaudeWorkspace\BeeBot
```

---

## Daily training

### Start training (basic)
```powershell
.\.venv\Scripts\python.exe -m rl.train_ppo
```
Press `ESC` any time to stop cleanly (saves checkpoint before exiting).
Hold `F8` to pause the bot mid-training so you can reposition it manually. Release F8 to resume.

### Start training + save log — SIMPLEST (one file, appends across runs)
```powershell
.\.venv\Scripts\python.exe -m rl.train_ppo *>&1 | Tee-Object -FilePath "training_log.txt" -Append
```

### Start training + save timestamped log
Note: use dashes between date parts (`HH-mm-ss` not `HHmm`) so PowerShell casing doesn't confuse minutes with months.
```powershell
$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
.\.venv\Scripts\python.exe -m rl.train_ppo *>&1 | Tee-Object -FilePath "training_$stamp.txt"
```

### Start training + save timestamped log to OneDrive (visible from any device)
```powershell
$stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
New-Item -ItemType Directory -Path C:\Users\harla\OneDrive\BeeBotLogs -Force | Out-Null
.\.venv\Scripts\python.exe -m rl.train_ppo *>&1 | Tee-Object -FilePath "C:\Users\harla\OneDrive\BeeBotLogs\training_$stamp.txt"
```

### All in ONE line (no variable) — if you prefer
```powershell
.\.venv\Scripts\python.exe -m rl.train_ppo *>&1 | Tee-Object -FilePath "C:\Users\harla\OneDrive\BeeBotLogs\training_$((Get-Date).ToString('yyyy-MM-dd_HH-mm-ss')).txt"
```
Uses single quotes inside `.ToString(...)` to avoid nested double-quote issues.

### PowerShell Get-Date format gotcha
Case-sensitive tokens — easy to get wrong:
- `MM` = Month (e.g., `08`)
- `mm` = Minute (e.g., `45`)
- `HH` = Hour 24-format (e.g., `21`)
- `hh` = Hour 12-format (e.g., `09`)

Safest habit: separate every time component with `-` or `_` so a mis-cased letter produces a broken filename instead of the wrong date.

### View OneDrive log on another device
```powershell
# Any device with OneDrive synced
notepad C:\Users\harla\OneDrive\BeeBotLogs\training_2026-08-15_2145.txt
```

---

## TensorBoard (live metric graphs)

### Start TensorBoard on the machine that's TRAINING
```powershell
# LOCAL-only (only viewable on same machine)
.\.venv\Scripts\tensorboard --logdir logs\tensorboard

# LAN-visible (view from other devices on same WiFi)
.\.venv\Scripts\tensorboard --logdir logs\tensorboard --host 0.0.0.0 --port 6006
```

### Find your machine's LAN IP (for accessing from another device)
```powershell
ipconfig | findstr IPv4
```
Look for `192.168.x.x` — that's your LAN IP.

### Access TensorBoard from another device on same network
Open a browser on the other device to:
```
http://<training-machine-ip>:6006
```

**First time — allow through firewall:** Windows may prompt about Python needing network access. Click "Allow" for private networks.

---

## HUD reader tests (for tuning / debugging)

### Test pollen bar OCR
```powershell
.\.venv\Scripts\python.exe -m hud.pollen_ocr
```

### Test honey OCR
```powershell
.\.venv\Scripts\python.exe -m hud.honey_ocr
```

### Test quest tab reader (open the quest tab in-game first)
```powershell
.\.venv\Scripts\python.exe -m hud.quest_ocr
```

### Sample pixel colors for quest tab tuning
```powershell
.\.venv\Scripts\python.exe -m hud.quest_ocr --sample-colors
```
Open the quest tab in-game first. Prints the actual BGR colors at each sample point. Use to tune `hud/probes/quest_panel_bg_color.txt` if detection is unreliable.

### Test buff classifier + show detected buff strip region
```powershell
.\.venv\Scripts\python.exe -m hud.buff_classifier
```
Saves `hud/probes/debug_buff_strip.png` — inspect to see what region the classifier thinks the buff strip is in.

---

## Snipping templates

### General snip command
```powershell
.\.venv\Scripts\python.exe snip_template.py hud/probes/<template_name>
```
(or `bridges/probes/<template_name>` for dialogue rescue templates)

### Common templates to snip

Do the setup in-game FIRST (open the right menu, position character, etc), THEN run the snip command.

**Dialogue rescue for a specific bear** — snip while bear dialogue is open:
```powershell
.\.venv\Scripts\python.exe snip_template.py bridges/probes/dialogue_continue_black_bear
.\.venv\Scripts\python.exe snip_template.py bridges/probes/dialogue_continue_panda_bear
```

**Buff icon** — snip while that buff is active in-game (snip tightly around JUST the icon graphic, no stack-count text):
```powershell
.\.venv\Scripts\python.exe snip_template.py hud/probes/buff_haste
.\.venv\Scripts\python.exe snip_template.py hud/probes/buff_focus
```

**Quest tab open indicator** — snip inside the OPEN quest panel (stable element like a category header background):
```powershell
.\.venv\Scripts\python.exe snip_template.py hud/probes/quest_tab_indicator
```

**Pollen bar frame** — snip the pollen HUD element (includes label + bar):
```powershell
.\.venv\Scripts\python.exe snip_template.py hud/probes/pollen_bar_frame
```

**Honey display** — snip the honey number display:
```powershell
.\.venv\Scripts\python.exe snip_template.py hud/probes/honey_display
```

---

## Config file overrides (no code editing needed)

### Buff strip region override
Edit `hud/probes/buff_strip_bounds.txt` — one line with 4 floats:
```
x_start_frac y_start_frac x_end_frac y_end_frac
```
Example for the default:
```
0.0 0.05 0.18 0.12
```
Comments (lines starting with `#`) are ignored.

### Quest panel background color override
Save to `hud/probes/quest_panel_bg_color.txt` — one line with 3 ints (BGR order):
```
185 170 140
```

---

## Git — code sync between machines

### Get latest code changes
```powershell
git pull
```

### See what's changed since last sync
```powershell
git status
git log --oneline -20    # last 20 commits
```

### Push your local changes
```powershell
git add <files>
git commit -m "your message"
git push
```

---

## Model file sync (git-ignored files that don't auto-sync)

These files are NOT in git — you need to copy them manually between machines if you want to preserve training progress:

### Files to copy
```
models/beebot_ppo_latest.zip     ← current PPO weights (essential for resume)
models/beebot_lstm_best.pt        ← imitation warm-start (needed if fresh training)
hud/probes/*.png                  ← your snipped templates
hud/probes/*.txt                  ← your config overrides
```

### Copy via OneDrive (easiest)
```powershell
# On source machine — copy files to OneDrive
New-Item -ItemType Directory -Path C:\Users\harla\OneDrive\BeeBotSync\models -Force | Out-Null
Copy-Item -Path models\beebot_ppo_latest.zip -Destination C:\Users\harla\OneDrive\BeeBotSync\models\ -Force
Copy-Item -Path models\beebot_lstm_best.pt -Destination C:\Users\harla\OneDrive\BeeBotSync\models\ -Force
Copy-Item -Path hud\probes\*.png -Destination C:\Users\harla\OneDrive\BeeBotSync\probes\ -Force
Copy-Item -Path hud\probes\*.txt -Destination C:\Users\harla\OneDrive\BeeBotSync\probes\ -Force

# On destination machine — pull them back
Copy-Item -Path C:\Users\harla\OneDrive\BeeBotSync\models\*.* -Destination models\ -Force
Copy-Item -Path C:\Users\harla\OneDrive\BeeBotSync\probes\*.* -Destination hud\probes\ -Force
```

### Copy over LAN (if both machines on same network)
```powershell
# On source machine — enable folder sharing on C:\ClaudeWorkspace\BeeBot\models
# Then on destination:
Copy-Item -Path \\<source-machine-name>\BeeBotShare\models\*.zip -Destination models\ -Force
```

---

## Fresh setup on a new machine

### One-time setup
```powershell
cd C:\ClaudeWorkspace
git clone https://github.com/HarlanBarlan/BeeBot.git
cd BeeBot
.\setup.ps1                                              # creates .venv and installs deps
```

If PyTorch fails to install, `setup.ps1` should handle CUDA wheel index automatically. If not:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Verify GPU works
```powershell
.\.venv\Scripts\python.exe test_gpu.py
```

---

## Debug / recovery commands

### Check current HUD reader status (all detectors)
```powershell
.\.venv\Scripts\python.exe -c "from hud.reader import HudReader; HudReader().status()"
```

### Force-kill all Roblox processes (if launcher is stuck)
```powershell
Get-Process | Where-Object { $_.Name -match 'Roblox' } | Stop-Process -Force
```

### Rename current PPO checkpoint (before starting a fresh run)
```powershell
Move-Item models\beebot_ppo_latest.zip models\beebot_ppo_pretraining.zip
```

### Roll back PPO checkpoint (if current one is damaged)
```powershell
# List checkpoints saved every 5000 steps
Get-ChildItem models\beebot_ppo_*_steps.zip | Sort-Object LastWriteTime -Descending | Select-Object -First 5

# Restore an older one as "latest"
Move-Item models\beebot_ppo_latest.zip models\beebot_ppo_damaged.zip
Copy-Item models\beebot_ppo_XXXX_steps.zip models\beebot_ppo_latest.zip
```

### Delete VecNormalize stats (if reward scaling seems off)
```powershell
# We no longer use VecNormalize, but if a stale file exists:
Remove-Item models\vec_normalize.pkl -ErrorAction SilentlyContinue
```

### Wipe log directory (start fresh)
```powershell
Remove-Item logs\tensorboard\* -Recurse -Force
```

---

## Fetch buff icons from wiki (one-time / occasional)

```powershell
# Dry-run — see what WOULD be downloaded, download nothing
.\.venv\Scripts\python.exe scripts\fetch_wiki_buff_icons.py --dry-run

# Actual download
.\.venv\Scripts\python.exe scripts\fetch_wiki_buff_icons.py

# Force redownload (overwrite existing)
.\.venv\Scripts\python.exe scripts\fetch_wiki_buff_icons.py --force
```

---

## Prerequisites checklist before training

- [ ] Roblox is open and Fredrick is logged in
- [ ] Character is in a productive location (Sunflower Field is a safe default)
- [ ] No dialogue, menu, or shop UI is currently open
- [ ] Bag is empty or partially full (either is fine — bot handles both)
- [ ] Only Fredrick's Roblox running (supervisor kills all Roblox processes owned by current user on startup — don't have your main account playing at the same time)
- [ ] You're NOT going to use the computer for other things during training (bot takes over keyboard/mouse — F8 pauses temporarily)

---

## What each metric means (quick reference)

- `ep_rew_mean` — average total reward per episode (1024 steps ~= 3.5 min). **Want positive and trending up.**
- `explained_variance` — critic health, range [-∞, 1]. **Want positive, ideally >0.1.** Negative = critic worse than a constant predictor.
- `value_loss` — MSE of critic predictions. Small when rewards near zero OR critic is perfect (need context).
- `entropy_loss` — how spread-out the policy is. Stable = converging. Climbing more negative = increasing exploration.
- `std` — standard deviation of continuous actions. Stable-around-1.0 is fine for our setup.
- `clip_fraction` — % of PPO updates that hit the clip range. >0.4 = big updates being clipped, policy is thrashing.

---

## When to send logs to me for review

- After a long uninterrupted training session (2+ hours) — for trend analysis
- If ep_rew_mean regresses hard for many iterations — probable bug
- If a new error type appears in the log — probable bug
- Weekly check-in — general "are we on track?"

Send the last ~500 lines of stdout OR the whole `training_*.txt` file.
