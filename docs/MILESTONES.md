# BeeBot Milestones

Chronological log of significant events in Fredrick's training. Complements [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) — that one tracks **design decisions** (why we changed X); this one tracks **behavioral and performance outcomes** (what Fredrick actually achieved, and when).

Together they form the paper's learning-curve narrative. Anyone can plot `ep_rew_mean` — this doc captures the moments that plot can't show.

---

## How to use this doc

You won't be watching Fredrick constantly. Micro-behaviors (field switches, tool swings, quest tab opens) happen randomly under any policy and are meaningless as "firsts." What matters is **macro game achievements a human player would celebrate**: bee gates cleared, bosses defeated, gear upgrades, leaderboard mentions.

**When you check in on training** (every few hours, once a day, whenever), spend 2 minutes looking at Fredrick's in-game state and drop a check-in entry below. That's the whole workflow.

**Auto-detected milestones** (honey balance crossings, training-step counts) fire automatically via [common/milestones.py](../common/milestones.py) and write to `logs/milestones.jsonl`. Copy notable ones here with human commentary about what training iteration was running.

---

## Check-in template

Copy this block, fill in current state each time you check on Fredrick:

```markdown
### Check-in YYYY-MM-DD HH:MM (~Xh training total, step ~N)
- **Bee count:** X (was Y — delta since last check)
- **Honey:** X (was Y)
- **Best gear equipped:** [tool] / [backpack] / [mask] / [any amulets]
- **New bees since last check:** [count + rarities if notable]
- **Bosses defeated since last check:** [any]
- **Fields visible in observation:** [where bot spent most of the time]
- **Leaderboard mentions:** [any global leaderboard, any position]
- **Notable behavior:** [anything worth writing down — 1-2 sentences from a 5-min observation]
- **Concerns:** [regressions, new stall types, anything worth investigating]
```

---

## Bee-count gates (verified list)

BSS has 7 numeric bee-count gates (physical gates in-world). Each unlocks new fields/NPCs. Log the date + step-count when Fredrick clears each. Fredrick started at ~15 bees per most recent check-in, so the first 3 are already historic.

| Gate | Bees required | Approximate content unlocked | Fredrick cleared |
|---|---|---|---|
| Basic Bee Gate | 5 | early fields, first quest bears | pre-tracking |
| Brave Bee Gate | 10 | Pineapple Patch area, Pro Shop | pre-tracking |
| Honey Bee Gate | 15 | Cactus/Pumpkin/Pine Tree/Rose area | pre-tracking (Fredrick's current unlock ceiling) |
| Ant Gate | 20 | Ant Field + Ant Challenge access | *pending* |
| Lion Bee Gate | 25 | Mountain Top Field, Mondo Chick, Instant Converter (2nd), Ticket Shop, hive slot expansion | *pending* |
| Bear Gate | 30 | Strawberry / Bamboo / Blueberry, Panda Bear, Blue HQ + Red HQ NPCs, Bucko/Riley infinite quests | *pending* |
| Windy Bee Gate | 35 | Coconut Field, Pepper Patch, Spirit Bear, Wind Shrine, Petal Shop, endgame chain begins | *pending* |

Beyond 35 bees there are no more physical bee-count gates — just soft milestones (Field Booster requires 20 discovered types, endgame gear cost thresholds, etc.).

---

## Boss defeats (verified list)

| Boss | Where | Difficulty class | First defeat |
|---|---|---|---|
| **Commando Chick** | Commando Chick's Hideout (behind Brave Bee Gate) | Easiest boss — 30-min respawn, low HP, grenade phase | *pending* |
| **King Beetle** | Ant Field area | Early-mid — HP 2,500, 40 dmg/hit, 24h respawn | *pending* |
| **Mondo Chick** | Mountain Top Field | Public — HP 300k but auto-decays 333 HP/sec, trivial after 15 min. Any damage = loot share. Hourly UTC spawn. | *pending* |
| **Werewolf** | Spider Field cave, **night only** | Mid — HP 250, 35 dmg/hit, 1h respawn. Drops Diamond Egg (~0.5%) / Star Jelly | *pending* |
| **Coconut Crab** | Coconut Field | Mid-late — HP 250k, 36h respawn, 2h fight timer (HP resets if you leave). Guaranteed Coconuts + Micro-Converters | *pending* |
| **Tunnel Bear** | Tunnel behind Panda Bear's quests | Late — HP ~10k, ~1000 dmg per boulder (one-shots low-defense). ~48h respawn | *pending* |
| **Stump Snail** | Strawberry Field | Very late — HP 30M, 96h respawn, **HP persists across sessions** (chip-farmable). Drops Glue + Shell Amulet | *pending* |

Also worth logging separately when they happen:
- **First Rogue Vicious Bee kill** (spawns via gray spike in Clover/Spider/Cactus/Rose/Mountain Top/Pepper). First daily = +5 Stingers with 22h cooldown.
- **First Aphid kill** (10 BP normal, 20 BP rare).

---

## Challenges (separate from bosses)

| Challenge | Where | First completion / notable scores |
|---|---|---|
| **Ant Challenge** | Ant Field (20-bee gate). 5-min run, requires Ant Pass (regenerates 1 every 2h via dispenser). Bag pollen auto-converts to honey on start. Score tiers: 0-24 Bronze / 25-49 Silver / 50-99 Gold / 100-149 Diamond / 150+ Supreme. Amulet quality scales linearly to score 400. | *pending* |
| **Stick Bug Challenge** | Cactus Field. Talk to Stick Bug. 10-min fight. Free once per 36h, else 50 tickets. Score gates Bronze/Silver/Gold/Diamond (no Supreme). Score cap 200M for max quality. Only source of Monster Respawn -%. | *pending* |
| **Robo Bear Challenge** | Round-based. Every 5 rounds = new Cog Amulet tier (R5 Bronze → R25 Supreme). Cogs collected → linear quality scaling capped at 1,000. Only source of Nectar Rate/Duration + Super-Crit Power. | *pending* |
| **Wild Windy Bee** | Server-wide event. Camouflaged cloud with white trail spawns over a field. Touching starts fight. Drops Cloud Vials toward Windy Bee questline. | *pending* |

---

## Gear milestones

### Tool upgrades
BSS tools are NOT organized in a "wooden→plastic→silver→golden→diamond" tier system (that's a common misconception). Tools are organized by which SHOP sells them. Log each tool upgrade Fredrick makes.

- **Starting tool:** Scooper (2 pollen from 2 patches, 0.8s swing)
- **First Pro Shop tool** (~40k-1.5M honey range, Fredrick's current price range): *pending*
- **First Mountain Top Shop tool** (~20M-150M range): *pending*
- **First Red HQ tool / Blue HQ tool** (~3.5M-2.5T range): *pending*
- **Porcelain Dipper** (150M, Mountain Top): required for Tunnel Bear engagement. *pending*
- **Endgame trio:** Dark Scythe (Riley Q250 reward, red hive), Tide Popper (Bucko Q250 reward, blue hive), Gummyballer (Gummy Bear's Lair, 10T + crafting)
  - First unlock: *pending*

### Backpack upgrades
Fredrick starts with capacity ~405k. Verified progression (each is an item, not a tier):

- **Pouch** +200
- **Jar** +750
- **Backpack** +3,500
- **Canister** +10k
- **Mega-Jug** +25k
- **Compressor** +50k
- **Elite Barrel** +125k
- **Port-O-Hive** +250k
- **Red / Blue Port-O-Hive** +400k each
- **Porcelain Port-O-Hive** +600k
- **Coconut Canister** +1M

Belts stack on top: Belt Pocket +5k, Belt Bag +25k, Mondo Belt Bag +100k, Honeycomb Belt +150k, Petal Belt +250k.

Log each backpack + belt upgrade Fredrick purchases with the resulting total capacity.

### Amulets
BSS has 7 amulet types, all wearable simultaneously. Tier ladder: **Bronze → Silver → Gold → Diamond → Supreme** (Stick Bug caps at Diamond, no Supreme).

- **First amulet (any type):** *pending*
- **First Ant Amulet** (source: Ant Challenge score): *pending*
- **First King Beetle Amulet** (1/7 drop from KB): *pending*
- **First Shell Amulet** (Stump Snail): *pending*
- **First Stick Bug Amulet** (Stick Bug Challenge): *pending*
- **First Cog Amulet** (Robo Bear Challenge): *pending*
- **First Moon Amulet** (100 Moon Charms per generation): *pending*
- **First Star Amulet Bronze** (5 gifted types + 25M honey): *pending*
- **First Silver Star Amulet** (10 gifted types): *pending*
- **First Gold Star Amulet** (20 gifted types): *pending*
- **First Diamond Star Amulet** (30 gifted types): *pending*
- **First Supreme Star Amulet** (40 gifted types — endgame anchor): *pending*

### Beequips
- **First beequip equipped:** *pending*
- **First beequip past 3-star potential:** *pending*
- **First beequip past 5-star potential (if possible):** *pending*
- **Beequip case expansion:** starts 5 slots, expands to ~13-15 via Dapper Bear quests 2/4/6/8/10/12/14/16. Log each expansion.

### Masks / boots / gliders
- **First mask beyond default Basic:** *pending*
- **Endgame mask trio:** Demon Mask (red), Diamond Mask (blue), Gummy Mask (mixed). *pending*
- **First boots upgrade:** *pending*
- **First Glider (Mountain Top):** *pending*

---

## Bee milestones

### Bee count (auto-detectable via OCR eventually; manual for now)
| Count | Reached |
|---|---|
| 15 (start) | 2026-08-14 (approximate) — Fredrick's starting hive |
| 16 | *pending* |
| 20 | *pending* — unlocks Ant Gate |
| 25 | *pending* — unlocks Lion Bee Gate, first hive slot expansion available |
| 30 | *pending* — unlocks Bear Gate |
| 35 | *pending* — unlocks Windy Bee Gate |
| 40 | *pending* |
| 45 | *pending* |
| 50 | *pending* — max hive |

### Bee rarity firsts
Rarity ladder: Common → Rare → Epic → Legendary → Mythic → Event.

- **First Rare bee:** *pending*
- **First Epic bee:** *pending*
- **First Legendary bee:** *pending*
- **First Mythic bee:** *pending*
- **First Event bee** (e.g., Bear Bee, Puppy Bee, Photon Bee): *pending*

### Gifted bee milestones
Universal gifted hatch rate is 0.4% (1/250) on non-Basic hatches; 0.348% (~1/287) on Basic Egg.

- **First Gifted bee** (any rarity): *pending*
- **5 unique gifted types** (unlocks Bronze Star Amulet): *pending*
- **10 unique gifted types** (Silver): *pending*
- **20 unique gifted types** (Gold): *pending*
- **30 unique gifted types** (Diamond): *pending*
- **40 unique gifted types** (Supreme — endgame): *pending*

### Special bees
- **First Vicious Bee** (250 Stingers claim): *pending*
- **First Windy Bee** (donate Spirit Petal + Cloud Vials at Wind Shrine): *pending*
- **First Gummy Bee** (Gummy Bear questline): *pending*
- **First Bear Bee** (Robux voucher OR extremely rare Tunnel Bear drop): *pending*
- **First Mondo Chick Mythic Egg** (fast-kill drop): *pending*

---

## Hive expansion

Hive starts with **25 slots**. Max is **50** via individual purchases past Lion Bee Gate. First extra slot costs 3M honey; the 25th extra slot costs ~2.17T honey; total ~3.42T for full expansion.

- **First hive slot purchased** (26 total): *pending*
- **30 slots**: *pending*
- **35 slots**: *pending*
- **40 slots**: *pending*
- **45 slots**: *pending*
- **50 slots (max)**: *pending*

---

## Leaderboard mentions

**All BSS leaderboards are global (not per-server).** ANY leaderboard mention for Fredrick is a genuine global-scale milestone worth logging.

Confirmed leaderboards:
- All-Time Top Honeymakers
- Daily Top Honeymakers (resets 12:00 AM CST)
- Monthly Top Honeymakers (resets 1st of month)
- Daily Top Honey Gift Receivers
- All-Time Top Battlers (Battle Points)
- Per-bear Top Helpers: Top Brown Bear Helpers, Top Riley Bee Helpers, Top Bucko Bee Helpers, Top Stick Bug Fighters, etc.
- Ant Challenge monthly rank (rewards for top tiers)

- **First leaderboard mention ever (any board, any position):** *pending*
- **First top-1000 finish:** *pending*
- **First top-100 finish:** *pending*
- **First top-10 finish:** *pending*

---

## Honey thresholds (auto-detected)

Fired automatically by [MilestoneTracker](../common/milestones.py); see `logs/milestones.jsonl` for the machine-generated event stream. When the training log prints `[milestone] honey crossed X at step N`, copy the row here with commentary.

| Threshold | Reached | Notes |
|---|---|---|
| 5,000,000 | pre-tracking (2026-08-16, started ~4.4M) | Fredrick crossed during first fresh training run PPO_36 |
| 10,000,000 | *pending* | |
| 25,000,000 | *pending* | |
| 50,000,000 | *pending* | |
| 100,000,000 | *pending* | |
| 250,000,000 | *pending* | |
| 500,000,000 | *pending* | |
| 1,000,000,000 | *pending* | Mid-game economy |
| 5,000,000,000 | *pending* | |
| 10,000,000,000 | *pending* | Late-game milestone |

Auto-detected training step thresholds (100k / 500k / 1M / 5M / 10M / 50M / 100M) fire similarly.

---

## Training metric milestones

### First iteration with `ep_rew_mean` > 0 sustained
*2026-08-16, iterations 3-5 of PPO_36.* First fresh training run after quest OCR removal reached `ep_rew_mean = 0.112` at iteration 5, having climbed from -0.09 at iteration 3. Sustained positive across three consecutive iterations for the first time on record. `value_loss` dropped from ~6260 (pre-quest-removal) to ~0.008 in the same window — 20,000× cleaner value function fit.

### First iteration with `explained_variance` > 0.3
*2026-08-16, iteration 5 of PPO_36.* Hit `explained_variance = 0.386` — value function is genuinely fitting the return signal instead of predicting the mean.

### First iteration with `explained_variance` > 0.5
*Pending.*

### First iteration with `explained_variance` > 0.7
*Pending.* PLAN.md's Phase 3 move-on criterion is 0.7+ sustained.

### CNN unfreeze at 200,000 steps
*Pending.* Announced by `[cnn-freeze]` line in training log. From this point PPO gradients flow into the imitation-warm-started CNN backbone.

### First 100k / 500k / 1M / 5M / 10M timesteps
Auto-detected. *Pending.*

### First `value_loss` sustained under 0.01
*2026-08-16, PPO_36 iterations 1-3.* value_loss around 0.007-0.06. If it stays here across many iterations, value function is fully fit.

---

## Time / uptime milestones

- **First 24-hour continuous training uptime**: *pending*
- **First 100k steps without any dialogue-rescue firing**: *pending*
- **First Roblox crash recovered by supervisor**: *pending*
- **First week of training accumulated** (7 × 24h wall-clock, not continuous): *pending*
- **First month of training accumulated**: *pending*

---

## Regression events (paper case studies)

Serious training regressions are as paper-relevant as successes — they document what went wrong and how it was fixed. Log the metrics regression, root cause, and fix.

### 2026-08-15 — "The 3-hour age-dialog trap"
During a 6.2-hour training run (PPO_34), bot got caught in a Roblox age confirmation dialog around step 8,451. Dialogue-rescue's generic `dialogue_continue.png` template matched the age dialog at conf 1.00 — but clicking the matched coordinate did NOT dismiss the dialog. Cascading failure: template matched → click → dialog persists → matches again → click again → ... for over 3 hours of real time.

Metrics regression:
- `ep_rew_mean`: +0.75 (iteration 5) → -3.35 (iteration 24)
- Bot spent this time being punished for randomness while stuck, effectively unlearning good behavior.

**Root cause:** dialogue-rescue's generic template matched a UI element that wasn't a "Continue" button. Clicking the matched pixel had no dismissal effect.

**Fix:** popup handler subsystem added ([hud/popup_handler.py](../hud/popup_handler.py)). Snip a detector template + a close-button template per popup type. Popup handler runs BEFORE dialogue rescue and dismisses known popups in ~3 sec instead of waiting for stall detection.

**Paper relevance:** case study in why safety-net design must consider the ADVERSARIAL case (matched wrong thing) not just the ABSENT case (nothing matched). Fully documented in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

### 2026-08-16 — "The laptop's blind training hour"
User attempted first laptop training after repo transfer. dxcam and bettercam both failed init on laptop GPU (D3DERR_INVALIDCALL). Fell back to mss (CPU capture) at 1 fps. Templates snipped at desktop resolution failed to match on laptop. HUD OCR returned nothing (`pollen=?`, `honey=?`). Reward function returned -0.01 per step for blind observations. 1,700 steps of pure negative reinforcement before user stopped it.

**Fix:** untrack resolution-sensitive templates in git (`.gitignore`), each machine snips its own set. Laptop dropped from training rotation entirely (GPU too old for DXGI). Documented in EXPERIMENT_LOG.md and saved to memory as `project-beebot-laptop-incompatible`.

---

## Phase transitions (retrospective — happened before this doc existed)

### 2026-08-11 to 2026-08-12: Phase 2a — Imitation bootstrap
~2.5 hours of author gameplay recorded across Sunflower, Dandelion, Mushroom, Blue Flower. Trained CNN+LSTM (BeeBotLSTM) with per-frame BCE for keys/mouse, MSE for cursor. Backbone: 4-layer CNN (Conv2d 3→32→64→128→128) + AdaptiveAvgPool → 4096-dim embedding.
**Result:** `models/beebot_lstm_best.pt` — the warm-start substrate every PPO run uses.

### 2026-08-12 to 2026-08-13: Phase 2b — Live inference + safety plumbing
dxcam capture, pydirectinput input, ClipCursor confinement, supervisor auto-launch, dialogue rescue (multi-template + 25-click burst + E suppression), F8 manual pause.
**Result:** bot can run for 4+ hours without human intervention on non-crash issues.

### 2026-08-14 to 2026-08-15: Phase 2c — HUD readers
Pollen bar OCR, honey OCR (with decimal-shift and persistence-based outlier recovery), buff icon classifier (69 wiki-fetched templates), quest OCR (later removed — see below).
**Result:** bot's observation includes ground-truth game state instead of relying on CNN to visually parse HUD.

### 2026-08-15 (afternoon): Phase 3 launched — PPO training active
First PPO training runs. Multi-timescale reward + PBRS bag-fill shaping + quest reward channels (initially wired, later removed).

### 2026-08-15 (evening): Quest reward channels removed
Removed W_QUEST_PROGRESS and W_QUEST_COMPLETION from reward.py. Full rationale in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

### 2026-08-15 (late evening): Quest OCR fully removed
Deleted `hud/quest_ocr.py`, dropped HUD_DIM 14 → 10. Design reversal: even the observation channels were redundant with the raw image the CNN sees. Deferred quest reading to Phase 5 VLM entirely.

### 2026-08-16 (morning): First clean fresh training run
PPO_36. Fresh PPO from imitation CNN warm-start. 22,500 steps, 5 iterations. `ep_rew_mean = 0.112` at iteration 5, `explained_variance = 0.386`. First run where the value function actually fit the reward signal cleanly.

### 2026-08-16 (evening): Repo reorganization + extended training begins
Repo restructured into subpackages. Machine-local template convention established. Laptop dropped from training rotation (GPU incompatibility). Extended training begins on desktop + RDP Freddy session.

---

## Check-ins (add newest at top)

*Empty — add entries as you check in on training.*
