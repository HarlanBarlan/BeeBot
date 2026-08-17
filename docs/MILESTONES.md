# BeeBot Milestones

Chronological log of significant events in Fredrick's training. Complements [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) — that one tracks **design decisions** (why we changed X); this one tracks **behavioral and performance outcomes** (what Fredrick actually achieved, and when).

For BSS mechanics context (what a boss is, where things are, how amulets work), see the reference memories — do not restate that stuff here.

---

## How to use this doc

**Check-in workflow.** When you check in on training (every few hours, once a day), spend 2 minutes looking at Fredrick's in-game state and drop a check-in entry at the bottom. That's it.

**Auto-detected milestones** (honey balance crossings, training-step counts) fire via [common/milestones.py](../common/milestones.py) and write to `logs/milestones.jsonl`. Copy the notable ones into the honey/step sections below with commentary.

---

## Check-in template

Copy this block each time. Add newest at the top of the check-in section (bottom of this doc).

```markdown
### Check-in YYYY-MM-DD HH:MM (~Xh training total, step ~N)
- **Bee count:** X (was Y)
- **Honey:** X (was Y)
- **Best gear equipped:** [tool] / [backpack] / [mask] / [amulets]
- **New bees since last check:** [notes]
- **Bosses defeated since last check:** [any]
- **Leaderboard mentions:** [any]
- **Notable behavior:** [1-2 sentences from a 5-min observation]
- **Concerns:** [anything worth investigating]
```

---

## Bee-count gates

7 numeric bee-count gates. Log the date + step-count when Fredrick clears each.

| Gate | Bees required | Cleared |
|---|---|---|
| Basic Bee Gate | 5 | pre-tracking (Fredrick started ~15 bees) |
| Brave Bee Gate | 10 | pre-tracking |
| Honey Bee Gate | 15 | pre-tracking (Fredrick's current unlock ceiling) |
| Ant Gate | 20 | *pending* |
| Lion Bee Gate | 25 | *pending* |
| Bear Gate | 30 | *pending* |
| Windy Bee Gate | 35 | *pending* |

---

## Boss defeats

| Boss | First defeat |
|---|---|
| Commando Chick | *pending* |
| King Beetle | *pending* |
| Mondo Chick | *pending* |
| Coconut Crab | *pending* |
| Tunnel Bear | *pending* |
| Stump Snail | *pending* |

Also worth logging when notable:
- **First Rogue Vicious Bee kill:** *pending*
- **First rare Aphid kill (20 BP):** *pending*

---

## Challenges

| Challenge | First completion | Best score / tier |
|---|---|---|
| Ant Challenge | *pending* | |
| Stick Bug Challenge | *pending* | |
| Robo Bear Challenge | *pending* | |

---

## Gear milestones

### Tools
- **First Pro Shop tool purchase:** *pending*
- **First Mountain Top Shop tool:** *pending*
- **First Red HQ / Blue HQ tool:** *pending*
- **First endgame trio tool** (Dark Scythe / Tide Popper / Gummyballer): *pending*

### Backpacks + belts
Log each upgrade with new total capacity.

- Pouch through Compressor: pre-tracking (Fredrick's earlier progression)
- **Port-O-Hive equipped:** ✅ (already achieved per user check 2026-08-16)
- **Red Port-O-Hive:** *pending*
- **Blue Port-O-Hive:** *pending*
- **Porcelain Port-O-Hive:** *pending*
- **Coconut Canister (+1M):** *pending*
- **First belt (any):** *pending / verify*
- **Mondo Belt Bag or better:** *pending*

### Amulets
- **First amulet ever (any type):** *pending*
- **First King Beetle Amulet:** *pending*
- **First Ant Amulet:** *pending*
- **First Shell Amulet:** *pending*
- **First Stick Bug Amulet:** *pending*
- **First Cog Amulet:** *pending*
- **First Moon Amulet:** *pending*
- **First Star Amulet (Bronze — 5 gifted types + 25M honey):** *pending*
- **Silver Star Amulet (10 gifted types):** *pending*
- **Gold Star Amulet (20):** *pending*
- **Diamond Star Amulet (30):** *pending*
- **Supreme Star Amulet (40):** *pending* (endgame anchor)

### Beequips
- **First beequip equipped:** *pending / verify*
- **Beequip case expansions (5 → 15 max via Dapper quests):** log each

### Masks / boots / gliders
- **First Glider (10-bee gate area):** ✅ (already achieved per user check 2026-08-16)
- **First mask beyond Basic:** *pending*
- **First boots upgrade:** *pending*
- **Endgame mask (Demon / Diamond / Gummy):** *pending*

---

## Bee milestones

### Bee count
| Count | Reached |
|---|---|
| 15 (start) | 2026-08-14 approx |
| 16 | *pending* |
| 20 | *pending* — unlocks Ant Gate |
| 25 | *pending* — unlocks Lion Bee Gate |
| 30 | *pending* |
| 35 | *pending* |
| 40 | *pending* |
| 45 | *pending* |
| 50 (max) | *pending* |

### Bee rarity firsts
- **First Rare bee:** ✅ (during imitation recording, pre-tracking)
- **First Epic bee:** ✅ (during imitation recording, pre-tracking)
- **First Legendary bee:** ✅ (during imitation recording, pre-tracking)
- **First Mythic bee:** *pending*

### Gifted bee milestones
- **First Gifted bee (any rarity):** *pending / verify*
- **5 unique gifted types** (Star Amulet Bronze eligible): *pending*
- **10 unique gifted types** (Silver): *pending*
- **20 unique gifted types** (Gold): *pending*
- **30 unique gifted types** (Diamond): *pending*
- **40 unique gifted types** (Supreme): *pending*

---

## Hive expansion

- **First slot expansion (26 total):** *pending* — requires Lion Bee Gate (25 bees) + 3M honey
- **30 slots:** *pending*
- **35 slots:** *pending*
- **40 slots:** *pending*
- **45 slots:** *pending*
- **50 slots (max):** *pending*

---

## Leaderboard mentions

All BSS leaderboards are global. Any mention for Fredrick is a genuine global-scale milestone.

- **First leaderboard mention ever (any board, any position):** *pending*
- **First top-1000 finish:** *pending*
- **First top-100 finish:** *pending*
- **First top-10 finish:** *pending*

---

## Honey thresholds (auto-detected)

Fired by [MilestoneTracker](../common/milestones.py); events in `logs/milestones.jsonl`.

| Threshold | Reached |
|---|---|
| 5,000,000 | pre-tracking |
| 10,000,000 | *pending* |
| 25,000,000 | *pending* |
| 50,000,000 | *pending* |
| 100,000,000 | *pending* |
| 250,000,000 | *pending* |
| 500,000,000 | *pending* |
| 1,000,000,000 | *pending* |
| 5,000,000,000 | *pending* |
| 10,000,000,000 | *pending* |

Training-step thresholds (100k / 500k / 1M / 5M / 10M / 50M / 100M) fire similarly.

---

## Training metric milestones

### First `ep_rew_mean` > 0 sustained
✅ 2026-08-16, iterations 3-5 of PPO_36. `ep_rew_mean` = 0.112 at iter 5, having climbed from -0.09. Sustained positive across three consecutive iterations. `value_loss` dropped 20,000× vs pre-quest-removal runs.

### First `explained_variance` > 0.3
✅ 2026-08-16, iteration 5 of PPO_36. Hit 0.386.

### First `explained_variance` > 0.5
*Pending.*

### First `explained_variance` > 0.7 (Phase 3 move-on criterion)
*Pending.*

### CNN unfreeze at 200,000 steps
*Pending.* Announced by `[cnn-freeze]` line in training log.

### First `value_loss` sustained under 0.01
✅ 2026-08-16, PPO_36 iterations 1-3.

### Training step landmarks
Auto-detected. All *pending*.

---

## Time / uptime milestones

- **First 24-hour continuous training uptime:** *pending*
- **First 100k steps without any dialogue-rescue firing:** *pending*
- **First Roblox crash recovered by supervisor:** *pending / verify*
- **First week of training accumulated:** *pending*
- **First month of training accumulated:** *pending*

---

## Regression events (paper case studies)

### 2026-08-15 — 3-hour age-dialog trap
6.2-hour training run (PPO_34). Bot caught in Roblox age confirmation dialog around step 8,451. Dialogue-rescue's generic template matched at conf 1.00 but clicks didn't dismiss. Cascading failure for 3+ hours.

Regression: `ep_rew_mean` +0.75 (iter 5) → -3.35 (iter 24).

Fix: popup handler subsystem ([hud/popup_handler.py](../hud/popup_handler.py)). Detector + close-button template pair per popup type. Runs before dialogue rescue.

Paper relevance: safety-net design must consider the ADVERSARIAL case (wrong match), not just the ABSENT case (nothing matched). Full writeup in EXPERIMENT_LOG.md.

### 2026-08-16 — Laptop's blind training hour
Attempted first laptop training. dxcam + bettercam both failed init (D3DERR). Fell back to mss at 1 fps. Desktop-resolution templates failed to match on laptop. HUD OCR returned nothing. Reward function returned -0.01 per step for 1,700 steps of pure negative reinforcement.

Fix: untrack resolution-sensitive templates in git, each machine snips its own. Laptop dropped from training rotation entirely (GPU too old for DXGI).

---

## Phase transitions (retrospective — happened before this doc existed)

### 2026-08-11 to 2026-08-12: Phase 2a — Imitation bootstrap
~2.5 hours of recorded gameplay. Trained CNN+LSTM. Result: `models/beebot_lstm_best.pt`.

### 2026-08-12 to 2026-08-13: Phase 2b — Live inference + safety plumbing
dxcam, pydirectinput, ClipCursor, supervisor, dialogue rescue, F8 pause. Result: bot runs 4+ hours untouched on non-crash issues.

### 2026-08-14 to 2026-08-15: Phase 2c — HUD readers
Pollen bar OCR, honey OCR, buff classifier, quest OCR (later removed).

### 2026-08-15 (afternoon): Phase 3 launched
PPO training active with multi-timescale reward + PBRS.

### 2026-08-15 (evening): Quest reward channels removed
See EXPERIMENT_LOG.md.

### 2026-08-15 (late evening): Quest OCR fully removed
Design reversal — deferred quest reading to Phase 5 VLM. HUD_DIM 14 → 10.

### 2026-08-16 (morning): First clean fresh training run (PPO_36)
22,500 steps. `ep_rew_mean = 0.112` at iter 5. First run where value function fit cleanly.

### 2026-08-16 (evening): Repo reorganization + extended training begins
Subpackages, machine-local templates, laptop dropped. Extended training on desktop + RDP.

---

## Check-ins

*Add newest at top. Use the template above.*
