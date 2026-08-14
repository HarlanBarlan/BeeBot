# BeeBot — Full Roadmap

Goal (reaffirmed): a Bee Swarm Simulator bot that starts from minimal imitation and self-improves toward high-level play, with the smallest possible scripted layer needed to bridge deterministic UI interactions. Optimize for **fast progression** and **max honey/second** at every stage.

## Guiding principles

- **Learn as much as possible. Script only what learning fundamentally can't do.** Prior version of this doc said "script what doesn't need judgment." Correction: script only two categories:
  1. **Game constants** — egg prices, hive slot capacity, menu button pixel positions. Facts, not decisions.
  2. **Safety** — ESC quit, mouse-corner failsafe, key-release cleanup.
  Everything else — goal selection, boost sequences, item priority, field choice, boss fight tactics — is learned.
- **Minimize human demonstration.** Just enough to seed action priors. RL/self-play does the mastery.
- **Multi-module hybrid architecture.** Small specialized models for HUD reading, world detection, and behavior — coordinated by a LEARNED meta-policy that selects the current goal. HUD/world detectors stay specialized (small models, one job each). Goal selection is a learned policy on top of them, NOT a hardcoded rule set.
- **Measure everything.** Honey/hour is the primary north-star metric. Every module decision is judged by how it moves that number.
- **Don't let it be stubborn — design for curiosity.** The bot must try new things, not lock into the exact patterns you demonstrated. Concrete mechanisms:
  - Imitation model uses higher dropout (0.35) to prevent overfitting to your exact play
  - Inference uses stochastic sampling (temperature > 0) rather than always picking the argmax action
  - RL fine-tuning uses entropy bonus in PPO to encourage exploration
  - Consider intrinsic-motivation reward (curiosity bonus for visiting states rarely seen in training)
  - Occasionally force random-field selection so the bot has to figure out new areas
- **Camera control via mouse.** Beyond keyboard camera keys (`,` `.` I O), the bot must be able to look around via either shift-lock (Shift + WASD) or right-click-drag (helpers `right_mouse_down`/`up`, `drag_mouse`, `rotate_camera` in `robo_input.py`). Play recordings should include your natural camera-drag habits so the model learns them.

## What's scripted vs learned (revised — MUCH more learned)

**Scripted (never learned — game constants + safety):**
- Menu button locations (fixed UI pixels — buy egg, hatch, hive slot positions)
- Egg prices, RJ prices, tool prices (from `items/item_db.json`, updated when patched)
- Safety wrappers (ESC, FAILSAFE, key-release-on-exit)
- Screen capture pipeline / input send plumbing
- Base honey-equivalent values in `token_values.json` (anchor for reward shaping — TUNED, not learned)

**Bootstrap-scripted then LEARNED (RL replaces the script):**
- Menu navigation flows (buy egg, hatch, accept quest) — scripted v1 for immediate function, RL fine-tunes timing and adds context-aware skipping/reordering
- Nothing else is bootstrap-scripted anymore.

**Fully learned from Day 1:**
- **Goal selection** (previously scripted) — a learned meta-policy picks the current activity based on HUD + inventory + time state
- **Boost session sequence** (previously scripted) — RL discovers the order and timing
- **Field selection** (previously scripted) — RL picks the highest-value field considering boost multiplier, mob density, travel time
- **Item priority** during collection — RL prioritizes based on shaped reward, no hardcoded priority
- **Boss fight tactics** — RL learns from boss-specific training episodes
- **Route optimization** — RL discovers walking patterns
- **Camera control** — mouse-drag and shift-lock use learned from demo, refined by RL
- **In-boost token grabbing order** — RL discovers optimal grab sequence

**The "learned meta-policy" is the biggest change from v1.** Instead of a hardcoded `orchestrator.py` with 20 if/elif rules, we train a small policy network:
- Input: HUD state (honey, tickets, pollen%, buffs active, boss visible, night/day, quest tracker deltas), inventory summary, recent action history
- Output: current goal (one of ~10 activities)
- Reward: shaped honey earned over the next N minutes with that goal active
- Training: RL bootstrap-imitated from user's implicit goal choices (inferable from user's demonstrated activity flow), then self-play refinement

Cost: MUCH more compute, longer time-to-first-working-bot, more debugging when RL doesn't converge. Upside: dramatically higher innovation ceiling. Aligns with vision statement.

---

## Phase 2a — First imitation model (~1-2 weeks)

**Inputs:** ~4-5 hours of recorded gameplay across 10-15 bee stage. Coverage includes all activity types (farming, hive convert, bear quests, mob combat, menu use, sprout pop, planter place, one Wealth Clock use, one Memory Match).

**Deliverables:**
- `dataset.py` — PyTorch Dataset that reads `data/session_*/frames/*.jpg` + `labels.jsonl`
- `model.py` — small CNN → dense head → multi-label sigmoid over 75 keys + 2 mouse buttons + mouse (x,y) regression
- `train.py` — training loop with BCE loss on keys + MSE on cursor
- `play.py` — inference: screencap → model → keys/mouse each frame

**Milestone:** bot autonomously farms Sunflower Field for 20 minutes without human intervention, hive-returns and converts at least once.

**Honey/sec optimization at this stage:** minimal — the model just needs to farm competently. Focus on stability, not speed.

---

## Phase 2b — Add memory + scripted progression bridge (~2 weeks)

**Why:** feed-forward CNN can't chain multi-step actions like "walk to hive → click convert." Adding a short LSTM/attention window over the last ~30 frames unlocks short-horizon reactive behaviors. Scripted bridges handle deterministic UI flows the model won't learn from thin demonstration data.

**Deliverables:**
- Enhanced `model.py` — LSTM over 30-frame history, same output head
- `scripts/buy_egg.py` — walks to shop, clicks Basic Egg based on current honey tier
- `scripts/hatch_bee.py` — opens hive menu, clicks hatch, places in first empty slot
- `scripts/quest_cycle.py` — walks to each easy bear (Black/Brown/Panda when unlocked), accepts + turns in
- `scripts/wealth_clock.py` — daily interact
- `scripts/memory_match.py` — daily minigame (may need vision for card-flip pairs)
- `scripts/return_to_hive.py` — triggered when pollen bar visually full
- `orchestrator.py` — top-layer routine planner deciding "farm vs script cycle"

**Milestone:** bot runs autonomously for 24 hours without intervention, progresses in bee count.

**Honey/sec optimization:** scripted `return_to_hive` triggered on pollen-full detection → no idle time. Scripted `buy_egg` triggered whenever honey hits a threshold → bee count grows automatically. Quest turn-ins add bonus honey.

---

## Phase 2c — HUD reader modules (~1-2 weeks)

**Why:** the routine planner and RL reward function need to *read* game state — honey count, pollen %, buff timers, boss HP. A CNN over the whole screen won't do this reliably. Small purpose-built models will.

**Deliverables:**
- `hud/honey_ocr.py` — reads honey number (short-scale suffix parsing: k, M, B, T, q, Q, s, S, o, N, d)
- `hud/tickets_ocr.py` — reads ticket count
- `hud/pollen_bar.py` — reads pollen fill %
- `hud/buff_bar.py` — icon classifier for the ~30 common buff icons, with stack count + timer
- `hud/boss_bar.py` — detects boss engagement + HP remaining
- `hud/quest_tracker.py` — reads active quest text via OCR

**Milestone:** bot knows exact honey/ticket/pollen numbers at every frame; orchestrator can trigger scripts based on real thresholds (e.g., "honey ≥ 100k → buy Golden Egg script").

**Honey/sec optimization:** with accurate HUD reads, orchestrator can dynamically choose the highest-yield activity ("this field has +300% boost → farm here instead").

---

## Phase 3a — RL fine-tuning on farming (~3-4 weeks)

**Why:** this is where "better than a human" starts. The imitation model has learned "what farming looks like." RL improves execution past your skill via reward signal.

**Setup:**
- Reward = honey earned per minute (from HUD honey OCR)
- Environment = live game (no reset — long continuous episodes)
- Algorithm = PPO (stable, well-supported in stable-baselines3)
- Bootstrap from Phase 2b model weights
- Only fine-tune the last few layers initially, then unfreeze more

**Deliverables:**
- `rl/env.py` — Gym environment wrapping the game (observation = screen frame + HUD state, action = keys+mouse, reward = Δhoney per step)
- `rl/train_ppo.py` — training loop
- Trained checkpoint that outperforms your farming honey/hour

**Milestone:** bot's honey/hour exceeds your recorded honey/hour by 20%+ on the same field composition.

**Honey/sec optimization:** this IS the optimization. RL discovers walking patterns, camera angles, and swing timing better than you demonstrated.

---

## Phase 3b — Boost cycle execution (~2 weeks)

**Why:** endgame farming operates in 15-minute boost sessions with a formulaic sequence. Scripting the sequence gets you 2-10x honey/hour vs pure passive farming.

**Deliverables:**
- `boost_cycle.py` — trigger sequence:
  1. Detect materials available (Field Dice, Cloud Vial, Sprinkler, Super Smoothie via HUD/inventory)
  2. Execute in correct order: Field Dice → Cloud Vial → Sprinkler → Super Smoothie → grab Boost tokens FIRST → Rage → Focus → Melody → farm target field
  3. End when 15-min timer expires; return to hive with full boost still active
- Integration with orchestrator — auto-trigger when enough materials stocked

**Milestone:** when materials exist, bot executes full boost cycle automatically. Honey/hour during boost ≥ 5x passive rate.

---

## Phase 3c — Specialized skill modules (~4-6 weeks)

Each of these is a smaller learned module (imitation warmup, then task-specific RL), coordinated by the orchestrator when conditions match.

**Modules:**
- **King Beetle boss** — RL agent trained on repeated fights. Reward: boss defeated + time-to-defeat inverse.
- **Ant Challenge** — 5-min survival, reward = ants killed
- **Sprout collection** — recognizes sprout on ground, walks to it, collects drops
- **Night event** — recognizes night, handles Vicious Bee spawn OR flees Werewolf
- **Field selection policy** — reads on-screen field boost multiplier, picks best-return field
- **Planter management** — places planters on best matching field, returns to harvest after N hours

**Milestone:** bot handles all major recurring activity types without intervention.

---

## Phase 4 — Autonomous progression (~4 weeks continuous)

**Goal:** bot runs 24/7 (with occasional restarts), progresses through gates on its own.

**What's still manual:** strategic decisions — which bee to gift with Star Treats, red-vs-blue hive commitment, when to spend on Diamond Egg vs save. Log in weekly to make these calls. Bot does the execution.

**Milestone:** bot progresses from 15 bees to 25 bees autonomously in ≤2 weeks real time. Honey/hour scales with hive size.

---

## Optional Phase 5 — Deep specialization (open-ended)

- Boss-specific RL agents for Stump Snail, Tunnel Bear, Coconut Crab
- Bee Bear / Stick Bug challenge attempts (may or may not succeed — these are hard even for humans)
- Beequip loadout optimization (bandit search over compositions)
- Comp optimization via model-based RL on simulator learned from your play data

## Phase 6 — Full autonomy (long-term strategic + post-training learning)

**Trigger:** Phases 2-4 stable. Bot handles execution; now we push it to make strategic calls autonomously.

### 6a. Long-term decision modules

- **Multi-timescale reward** — reward function tracks honey earned over multiple windows: per-minute (fast RL feedback), per-hour (session-level), per-day (strategic). Weight blends over time so the bot cares about long-term outcomes, not just next-second gain.
- **Learned value function** (world model) — a small neural network trained on bot's own play data predicts `expected_honey_per_day(current_state, action)`. Used for planning: "if I spend 500k on Diamond Egg now, what's my day-out honey/hour vs saving?"
- **Bandit search on strategic choices** — treat "hive color commitment" as a multi-armed bandit. Bot tries a color for N days, measures cumulative honey, compares. Eventually converges on best-for-this-account color. Includes rollback: if bot commits blue then color composition changes (rare gifted bee obtained), it can un-commit.
- **Star Treat / Star Jelly usage policy** — RL policy that picks which bee to gift based on shaped reward: `expected_gifted_hive_contribution × obtain_probability_delta`. Bot may make "wrong" choices, RL updates values from measured outcome.
- **Save vs spend policy** — learned from cumulative return: bot compares "spend now on X" vs "save for future Y" using the value function. Occasionally overspends or oversaves as exploration.

### 6b. Post-training content learning

- **Curiosity/novelty detection** — feature extractor identifies "I've never seen this visual state before" (unfamiliar item icon, bee color, field texture). Novel state triggers `explore mode`: bot walks around it, interacts with it, records outcomes.
- **VLM item recognition** — Tier 3 of `item_recognizer.py` fully implemented. When an unknown item is on screen and the cursor hovers its tooltip, the bot captures the region, sends to a vision-language model (Claude/Gemini API or local LLaVA), receives structured item description, appends to `item_db.json` with `confidence: "vlm_inferred"`.
- **Auto-updating goal_profiles** — when a new item enters item_db, the reward calculator auto-slots it with default multipliers; over play sessions, RL refinement adjusts values based on measured outcome.
- **New-content activation cycle** — bot notices unknown NPC → walks to them → attempts interaction → observes dialog → uses VLM to parse dialog text → auto-generates a scripted flow for that NPC's quests if pattern is standard (accept → farm/kill target → return → turn in).

### 6c. Bucko / Riley questline modules

Bucko Bee (Blue HQ) and Riley Bee (Red HQ) unlock after obtaining a Translator (from Science Bear). Both have long quest chains (~250 quests each). Same format as other bear quests — collect X pollen from Y field, kill Z mobs.

- Extend `quest_cycle.py` to include Bucko and Riley as targets once Translator is detected in inventory
- Add Blue HQ and Red HQ locations to the routine planner
- Milestone rewards: Tide Popper at Bucko Q250, Dark Scythe at Riley Q250 — these get added to synergy_combos.json as unlocks
- Bucko/Riley questlines take dozens of real-time days; treated as background objectives — bot works on them whenever passing through appropriate fields

---

## Honey/sec optimization checklist (applied continuously)

Every phase should hit as many of these as possible:

- [ ] No idle time (immediate hive return on pollen full)
- [ ] Active field selection (highest field boost visible)
- [ ] Boost session executed whenever materials allow
- [ ] Wealth Clock daily
- [ ] Memory Match daily
- [ ] Ticket Tent purchases: Gold Eggs when 50+ tickets stockpiled
- [ ] Bear quest cycle: turn in ready quests before farming
- [ ] Planter place-and-forget on best-matching fields
- [ ] Scoop/backpack/mask upgrades as honey allows
- [ ] Never buy at bad honey/ticket ratios (learned heuristics or hardcoded)

---

## Rough timeline

- Weeks 1-2: Phase 2a (first imitation model)
- Weeks 3-4: Phase 2b (memory + scripted bridges)
- Weeks 5-6: Phase 2c (HUD readers)
- Weeks 7-10: Phase 3a (RL fine-tune on farming)
- Weeks 11-12: Phase 3b (boost cycles)
- Weeks 13-20: Phase 3c (specialized modules)
- Weeks 21-24: Phase 4 (autonomous progression)
- Beyond: Phase 5 (deep specialization)

Hobby pace at ~10 hrs/week. Compressed if you go harder, stretched if you go slower. Not linear — expect iteration, backtracking, and dead ends.

---

## What compromises the "pure trial-and-error" original vision

The scripted bridges in Phase 2b (buy_egg, hatch_bee, quest_cycle, wealth_clock, memory_match) are the concession. These are ~200 lines of deterministic UI clicks that could theoretically be learned via RL if given massive compute + curriculum learning. With hobby compute, learning them wastes weeks for zero improvement over hand-writing them.

Everything downstream — farming, bosses, sprouts, field selection, boost timing, mob dodging, execution optimization — is genuinely learned and can genuinely surpass your play. That's where the "innovative and better than a human" part lives.

Result: ~90% of the eventual bot's competence comes from learning. ~10% is scripted glue. That's the honest tradeoff.
