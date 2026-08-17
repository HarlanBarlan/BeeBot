# BeeBot Milestones

Chronological log of significant events in Fredrick's training. Complements [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) — that one tracks **design decisions** (why we changed X); this one tracks **behavioral and performance outcomes** (what Fredrick actually achieved, and when).

For BSS mechanics context (what a boss is, where things are, how amulets work), see the reference memories — do not restate that stuff here.

**Only bot-autonomous achievements count as milestones.** Anything the user did manually during imitation recording (buying gear, hatching bees) is baseline state, not a Fredrick milestone. Baseline listed below for reference; milestones start from there.

---

## Starting baseline (pre-Phase-3, from imitation recording era)

State Fredrick was in when PPO training began (~2026-08-15). None of this counts as a bot achievement — it's the setup the RL is starting from.

- **Bees:** ~15 (already through Honey Bee Gate)
- **Backpack:** Port-O-Hive (~250k capacity) + belt if equipped
- **Tools:** starting toolkit — Scooper baseline + any Pro Shop tools purchased manually
- **Masks/boots/gliders:** Glider (10-bee gate area) already equipped
- **Bees in hive:** includes at least one Rare, one Epic, one Legendary bee (hatched during imitation recording sessions)
- **Amulets:** none / verify
- **Beequips:** none / verify
- **Honey balance:** ~4.4M-5.2M range at PPO_36 fresh-start
- **Quest turn-ins ever:** 0

If Fredrick ever regresses below this (e.g., sells a Legendary bee), that's a data point worth logging.

---

## How to use this doc

**Check-in workflow.** When you check in on training (every few hours, once a day), spend 2 minutes looking at Fredrick's in-game state and drop a check-in entry at the bottom. That's it.

**Auto-detected milestones** (honey balance crossings, training-step counts) fire via [common/milestones.py](../common/milestones.py) and write to `logs/milestones.jsonl`. Copy the notable ones into the honey/step sections below with commentary.

---

## How to check in (2-min routine)

Do this order every time. Aim to finish in 2 minutes.

**Step 1 — Watch training terminal for 30 sec (tail the log).**
Look for:
- `[milestone] ...` — auto-fires when honey/step thresholds cross. Copy the exact line into the appropriate section of this doc.
- `[popup] dismissed ...` — normal, ignore unless it's a NEW popup type you haven't templated
- `[env t=... dialogue-rescue: no template matched ... Consecutive failures: N]` — if N climbs past ~10, bot is stuck. Screenshot the Roblox screen and note as a concern.
- Missing `[timing]` lines for extended stretches — capture is stalling; note as concern.
- `[reward] accepting large POS/NEG delta ... re-baselining` — usually fine. More than 2-3 per hour = re-snip `honey_display.png`.

**Step 2 — Look at Roblox window for 30 sec (no menus).**
- **Honey count** (top of screen) — write down the number for the check-in entry.
- **Pollen bar** — non-zero and moving?
- **Where is the bot?** Which field, or is it in a menu/UI/stuck spot?
- **Active buffs** (top-left strip) — anything interesting like Haste x10 stacks?
- **Any UI blocking gameplay?** If yes and the popup handler isn't dismissing it, snip a new popup template pair.

**Step 3 — Open the Hive menu for 30 sec.**
Just look, don't change anything (leaving it open blocks bot control — close when done).
- **Bee count** — total, and delta from baseline (~15).
- **Rarity of new slots** — any new Rare/Epic/Legendary/Mythic since last check?
- **Any Gifted bees** (star indicator)?
- Close the menu.

**Step 4 — Fill in check-in template (30 sec).**
Copy the template block below into the "Check-ins" section at the bottom of this doc. Fill in what changed since last time.

If nothing changed ("no progress since last check"), still log it — the ABSENCE of progression is data.

### When to spend more time (not routine)

Stop and investigate if you see:
- **Honey dropped** with no purchase reason (bot bought something surprising OR OCR misread)
- **Bee count dropped** (bot sold a bee?)
- **New popup type** the handler doesn't dismiss — snip template pair now
- **Bot stuck in same UI 5+ min** — F8 pause, walk out, unpause
- **Fresh gear equipped** you didn't see last check — write it down explicitly (this IS an autonomous milestone)
- **Any leaderboard mention** — huge, log to appropriate leaderboard row

### Weekly deeper check (10 min, once per week)

- Navigate to Ticket Tent → current ticket count
- Open quests menu → any progress or turn-ins?
- Check Star Hall (past Lion Bee Gate) → gifted-type count
- Wealth Clock used recently?
- Glance at TensorBoard once — training curve shape

### What to skip

- Watching micro-behavior (field switches, tool swings, quest tab opens — meaningless statistically)
- Checking every individual bee's bond level
- Reading full quest text (bot can't understand it pre-Phase-5)
- Trying to verify specific ability token pickups

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
Baseline: Port-O-Hive (user-equipped during imitation recording). Milestones are AUTONOMOUS upgrades by Fredrick during RL training.

- **First autonomous backpack upgrade (beyond baseline):** *pending*
- Red Port-O-Hive by bot: *pending*
- Blue Port-O-Hive by bot: *pending*
- Porcelain Port-O-Hive by bot: *pending*
- Coconut Canister (+1M) by bot: *pending*
- First belt purchased by bot: *pending*
- Mondo Belt Bag or better by bot: *pending*

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
Baseline: Glider (10-bee gate area) already equipped. Milestones = autonomous upgrades by Fredrick.

- **First autonomous glider upgrade:** *pending*
- First mask beyond Basic by bot: *pending*
- First boots upgrade by bot: *pending*
- Endgame mask (Demon / Diamond / Gummy) by bot: *pending*

---

## Bee milestones

### Bee count
Baseline: ~15 (user-set during imitation era). Milestones = counts Fredrick reaches AUTONOMOUSLY.

| Count | Reached |
|---|---|
| 15 (baseline) | pre-Phase-3, from imitation recording era |
| 16 (first autonomous +1) | *pending* — first hive-slot fill Fredrick earns on his own |
| 20 | *pending* — unlocks Ant Gate |
| 25 | *pending* — unlocks Lion Bee Gate |
| 30 | *pending* |
| 35 | *pending* |
| 40 | *pending* |
| 45 | *pending* |
| 50 (max) | *pending* |

### Bee rarity firsts
Baseline hive already includes at least one Rare, Epic, and Legendary bee (from imitation recording era, user-hatched). Milestones = bees Fredrick hatches AUTONOMOUSLY during RL training.

- **First autonomous Rare bee hatch by bot:** *pending*
- First autonomous Epic bee hatch by bot: *pending*
- First autonomous Legendary bee hatch by bot: *pending*
- **First Mythic bee (any source):** *pending* — extremely rare, worth tracking regardless of who does it

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

All BSS leaderboards are global (not per-server) and only display the top 100. Any leaderboard entry Fredrick earns IS a top-100 global finish by definition — meaningful milestones.

Track per-board because different boards test different bot capabilities. Log first-ever appearance on each; log top-10 finishes separately.

### First appearance on each board
- **All-Time Top Honeymakers:** *pending* (tests career honey accumulation)
- **Monthly Top Honeymakers:** *pending* (tests recent farming rate — resets 1st of month)
- **Daily Top Honeymakers:** *pending* (tests peak session output — resets 12:00 AM CST)
- **Daily Top Honey Gift Receivers:** *pending* (would require another player gifting Fredrick — unlikely for a bot but not impossible)
- **All-Time Top Battlers:** *pending* (Battle Points from mob kills)
- **Ant Challenge Monthly Rank:** *pending* (top Ant Challenge score, resets monthly)
- **Top Brown Bear Helpers:** *pending* (Brown Bear quests completed)
- **Top Riley Bee Helpers:** *pending* (Riley infinite quests completed)
- **Top Bucko Bee Helpers:** *pending* (Bucko infinite quests completed)
- **Top Stick Bug Fighters:** *pending* (Stick Bug Challenge scores)

### Top-10 finishes (rare, exceptional)
Log any time Fredrick reaches top-10 on any specific board. Each one is a distinct milestone.

- *pending*

---

## Honey thresholds (auto-detected)

Fired by [MilestoneTracker](../common/milestones.py); events in `logs/milestones.jsonl`.

| Threshold | Reached |
|---|---|
| 5,000,000 | Fredrick started at ~4.4M-5.2M baseline (partial credit — bot crossed 5M during PPO_36 fresh run 2026-08-16, but very close to starting balance) |
| 10,000,000 | *pending* (first meaningful autonomous milestone — Fredrick would need to earn ~5M in-training) |
| 25,000,000 | *pending* |
| 50,000,000 | *pending* |
| 100,000,000 | *pending* |
| 250,000,000 | *pending* |
| 500,000,000 | *pending* |
| 1,000,000,000 | *pending* |
| 5,000,000,000 | *pending* |
| 10,000,000,000 | *pending* |

Training-step thresholds (100k / 500k / 1M / 5M / 10M / 50M / 100M) fire similarly.

- **100,000 total training steps:** ✅ crossed 2026-08-17 during PPO_41 (session ended at step 102,400).

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

## Behavioral firsts

First-time occurrences of specific in-game behaviors worth calling out for the paper's learning-curve narrative. Different from honey/step thresholds — these are qualitative discovery events.

### 2026-08-17 — First bag fill to 100%
✅ PPO_41, iteration 6, ~step 21,900 of session (~step 28,500 total across resumed training). Pollen bar reached 100% during Sunflower/Dandelion farming after the starter Instant Converter became unusable (ticket depletion). Bot has never learned hive-convert during imitation (user used IC for conversions), so this is the first time full-bag stagnation is a live problem for the policy.

**Post-100% behavior (observed manually):** bot continues attempting to harvest in fields (its highest-probability action), occasionally drifts near hive pad and initiates conversion, but leaves the pad before conversion completes — jumps around and cancels the animation. Net honey gain during full-bag period ≈ 0.

**Reward-signal check (paper-relevant):** iteration-level `ep_rew_mean` trajectory shows the negative signal accumulating as expected — iter 4: +1.03, iter 5: +0.777, iter 6: +0.363, iter 7: -0.02. Full bag → PBRS bag-fill potential capped → per-tick reward drops toward zero → stagnation penalty accrues. This is the reward function working as designed; whether the policy can learn a new sequence from this signal alone is the actual open question.

**Why this matters:** first observed instance of the "post-full-bag exploration attractor" — a classic challenge for pure-RL bots when the demonstrating human never showed the required behavior (hive convert). Whether Fredrick learns to stand still at the pad through pure PPO + entropy exploration, and how long it takes, will be a core case study in the paper.

### First successful autonomous hive convert (start-to-honey-credit)
*pending* — user must directly observe the bot completing a hive convert to log this. 100%→0% pollen transitions in training logs are AMBIGUOUS: identical signatures come from both hive convert AND Instant Converter usage. This session (PPO_41, 2026-08-17) showed 6 such transitions and honey went 7.48M → 8.60M (+1.12M), but Fredrick had accumulated some tickets during the session, so any/all of those transitions may have been IC-based. Do not backfill from log evidence alone.

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

### Check-in 2026-08-17 (end of 6h PPO_41, step ~102,400)
- **Bee count:** 15 (unchanged; hive-menu not opened this check-in)
- **Honey:** ~8.60M (was 7.47M) → **+1.12M autonomous** this session
- **Best gear equipped:** Honey Dipper / Port-O-Hive / no mask / no amulets (unchanged)
- **New bees since last check:** none
- **Bosses defeated since last check:** none
- **Leaderboard mentions:** none
- **Notable behavior:** Session gain of +1.12M honey with 6 pollen-100→0 transitions in the logs. Origin AMBIGUOUS — Fredrick had accumulated tickets during the session, so the transitions could be Instant Converter uses rather than hive convert. User was not present to directly observe. Do not claim hive-convert learning until directly observed. `ep_rew_mean` climbed from -0.02 (iter 7) → +1.07 (iter 25) regardless — the reward pressure was real, whichever action closed the loop.
- **Concerns:** milestone tracker false-fired 10M/25M/50M at step 91,559 (true honey ~8.5M) — OCR misreads slipped past the ratio filter. Fixed via median-of-60 buffer + sequential threshold gate. Rate metric was also fooled by misread endpoints; fixed via median smoothing.

### Check-in 2026-08-17 (~7 iters into PPO_41, step ~28,500)
- **Bee count:** 15 (unchanged)
- **Honey:** ~7.64M (was 7.47M start of session — +170k earned in session)
- **Best gear equipped:** Honey Dipper / Port-O-Hive / no mask / no amulets (unchanged)
- **New bees since last check:** none
- **Bosses defeated since last check:** none
- **Leaderboard mentions:** none
- **Notable behavior:** Reached 100% pollen for the first time (~step 21,900). Tickets depleted so starter IC is no longer an option. Bot attempts hive convert when it drifts near the pad but leaves before conversion completes — walks off pad, cancels animation. Rest of the time continues to try to harvest in fields with a full bag.
- **Concerns:** hive-convert never demonstrated during imitation, so bot has no prior for "stand still on pad for N seconds." Reward signal IS pushing back (`ep_rew_mean` +1.03 → -0.02 over 4 iters), which is what we want. Question is whether entropy exploration + reward gradient is enough to discover the correct action sequence from scratch, or whether this becomes a persistent attractor.
