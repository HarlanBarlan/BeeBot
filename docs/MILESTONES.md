# BeeBot Milestones

Chronological log of significant events in Fredrick's training. Complements [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) — that one tracks **design decisions** (why we changed X); this one tracks **behavioral and performance outcomes** (what Fredrick actually did, and when).

Together they form the paper's learning-curve narrative. Anyone can plot `ep_rew_mean` — this doc captures the moments that plot can't show.

---

## How to use this doc

- **Auto-detected quantitative milestones** (honey thresholds, training-step landmarks, session starts) are written to `logs/milestones.jsonl` by [common/milestones.py](../common/milestones.py). Look there for the raw event stream. Copy notable ones into the sections below with human commentary about their significance.
- **Human-observed milestones** (novel behavior emergence, first quest turn-in, first field switch) must be logged by hand — bot has no semantic self-awareness of what "field switch" means.
- Add entries as they happen. When in doubt, log it — trimming later is easy, reconstructing forgotten moments is not.

**Entry format:**
```
### YYYY-MM-DD, step N (Xh training total): Title
What happened, specifically. What preceded it. What training iteration was running.
Log excerpt or screenshot reference if applicable.
```

---

## Tier 1 — Emergent behavior firsts (paper gold)

These prove nothing was scripted. Log the moment each first happens, even if you're not 100% sure it's a "learned" behavior vs a lucky one — the DATE is what matters for the timeline.

### First voluntary hive return
*Not yet observed.*

### First autonomous field switch
*Not yet observed.* Any pair — Sunflower → Dandelion, Blue Flower → Mushroom, etc.

### First successful bear dialogue completion
*Not yet observed.* Bot approaches bear, presses E, clicks through dialogue to close.

### First quest turn-in
*Not yet observed.* Per Fredrick's state, zero quest turn-ins completed as of 2026-08-16.

### First tool usage (Honey Dipper)
*Not yet observed.*

### First shop interaction that isn't accidental
*Not yet observed.*

### First mob avoidance
*Not yet observed.*

### First token pickup
*Not yet observed.*

### First quest tab open
*Not yet observed.* This is the meta-behavior called out in PLAN.md as a Phase 3 move-on criterion.

### First self-recovery from stall (no dialogue-rescue fired)
*Not yet observed.*

### First novel/unexpected strategy
*Not yet observed.* Reserve this section for anything that surprises you — the paper's most interesting cases live here.

---

## Tier 2 — Quantitative thresholds (auto-detected)

Fired automatically by MilestoneTracker; see `logs/milestones.jsonl` for the machine-generated event stream. Sections below track which have been reached, with human commentary where warranted.

### Honey thresholds
| Threshold | Reached | Notes |
|---|---|---|
| 5,000,000 | pre-tracking (2026-08-16, ~4.8M starting) | Fredrick started at ~4.4M and crossed 5M during first fresh training run |
| 10,000,000 | *pending* | |
| 25,000,000 | *pending* | |
| 50,000,000 | *pending* | |
| 100,000,000 | *pending* | |
| 250,000,000 | *pending* | |
| 500,000,000 | *pending* | |
| 1,000,000,000 | *pending* | Would put Fredrick in mid-game economy |
| 5,000,000,000 | *pending* | |
| 10,000,000,000 | *pending* | Late-game milestone |

### Training step thresholds
| Threshold | Reached | Notes |
|---|---|---|
| 100,000 | *pending* | Meaningful PPO iteration count starts here |
| 500,000 | *pending* | |
| 1,000,000 | *pending* | Common "the model has seen enough" checkpoint in the RL literature |
| 5,000,000 | *pending* | |
| 10,000,000 | *pending* | |
| 50,000,000 | *pending* | |
| 100,000,000 | *pending* | If we reach this, the paper writes itself |

### Bee count gates (manual — no OCR reader yet)
| Bee count | Reached | Notes |
|---|---|---|
| 15 (start) | 2026-08-14 (approximate) | Fredrick's starting count |
| 20 | *pending* | |
| 25 | *pending* | Unlocks Honey Bee Gate |
| 30 | *pending* | |
| 35 | *pending* | |
| 40 | *pending* | |
| 50 | *pending* | |

---

## Tier 3 — Training metric milestones

### First iteration with `ep_rew_mean` > 0 sustained
*2026-08-16, iteration 3-5 of PPO_36.* First fresh training run after quest OCR removal reached ep_rew_mean = 0.112 at iteration 5, having climbed from ep_rew_mean = -0.09 at iteration 3. Sustained positive across three consecutive iterations for the first time on record. `value_loss` dropped from ~6260 (pre-quest-removal) to ~0.008 in the same window — 20,000× cleaner value function fit.

### First iteration with `explained_variance` > 0.3
*2026-08-16, iteration 5 of PPO_36.* Hit `explained_variance = 0.386` — value function is genuinely fitting the return signal instead of predicting the mean.

### First iteration with `explained_variance` > 0.5
*Pending.*

### First iteration with `explained_variance` > 0.7
*Pending.* PLAN.md's Phase 3 move-on criterion is 0.7+ sustained.

### CNN unfreeze at 200,000 steps
*Pending.* Announced by `[cnn-freeze]` line in training log. From this point PPO gradients flow into the imitation-warm-started CNN backbone.

### First 1,000,000 timesteps trained
*Pending.*

### First `value_loss` sustained under 0.01
*2026-08-16, PPO_36 iterations 1-3.* value_loss around 0.007-0.06. If it stays here across many iterations, value function is fully fit.

---

## Tier 4 — System/reliability

### First 24-hour continuous training uptime
*Not yet.*

### First 100,000 steps without any dialogue-rescue firing
*Not yet.*

### First Roblox crash recovered by supervisor
*Not yet observed in visible logs.*

### Notable regression events (paper case studies)

**2026-08-15 — "The 3-hour age-dialog trap."**
During a 6.2-hour training run (PPO_34), bot got caught in a Roblox age confirmation dialog around step 8451. Dialogue-rescue's `dialogue_continue.png` (generic template) matched the age dialog at conf 1.00 — but clicking the matched coordinate did NOT dismiss the dialog. Cascading failure: template matched → click → dialog persists → matches again → click again → ... for over 3 hours of real time.

Metrics regressed catastrophically:
- `ep_rew_mean`: +0.75 (iteration 5) → -3.35 (iteration 24)
- Bot spent this training time being punished for randomness while stuck, effectively unlearning good behavior.

**Root cause:** dialogue-rescue's generic template was matching an age-dialog UI element that wasn't a "Continue" button. Clicking the matched pixel had no dismissal effect.

**Fix:** popup handler subsystem added ([hud/popup_handler.py](../hud/popup_handler.py)). Snip a detector template + a close-button template per popup type. Popup handler runs BEFORE dialogue rescue in the capture loop and dismisses known popups in ~3 sec instead of waiting for stall detection.

**Paper relevance:** case study in why safety-net design must consider the ADVERSARIAL case (matched wrong thing) not just the ABSENT case (nothing matched). Fully documented in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md).

**2026-08-16 — "The laptop's blind training hour."**
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
**Result:** bot's observation includes ground-truth game state instead of relying on CNN to visually parse HUD from downsized pixels.

### 2026-08-15 (afternoon): Phase 3 launched — PPO training active
First PPO training runs. Multi-timescale reward + PBRS bag-fill shaping + quest reward channels (initially wired, later removed).

### 2026-08-15 (evening): Quest reward channels removed
Removed W_QUEST_PROGRESS and W_QUEST_COMPLETION from reward.py. Full rationale in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) — quest rewards double-counted farming activity that Δhoney already rewarded.

### 2026-08-15 (late evening): Quest OCR fully removed
Deleted `hud/quest_ocr.py`, dropped HUD_DIM 14 → 10. Design reversal: even the observation channels for quest state were redundant with the raw image the CNN sees. Deferred quest reading to Phase 5 VLM entirely.

### 2026-08-16 (morning): First clean fresh training run
PPO_36. Fresh PPO from imitation CNN warm-start (HUD_DIM change invalidated prior PPO checkpoint). 22,500 steps, 5 iterations. `ep_rew_mean = 0.112` at iteration 5, `explained_variance = 0.386`. First run where the value function actually fit the reward signal cleanly.

### 2026-08-16 (evening): Repo reorganization + extended training begins
Repo restructured into subpackages (common/, imitation/, scripts/, docs/). Machine-local template convention established. Laptop dropped from training rotation (GPU incompatibility). Extended training begins on desktop + RDP Freddy session, both using dxcam on the same physical GPU.

---

## Notes on threshold spacing

Honey thresholds are log-spaced (roughly 2-2.5× jumps) so each crossing represents a genuinely new economic scale, not incremental noise. Once Fredrick is at 100M honey, hitting 105M isn't worth marking — 250M is.

Same for step counts: 100k, 500k, 1M, 5M... each crossing means "the training run is meaningfully longer than before."

If either category becomes too sparse or too dense in practice, edit `HONEY_THRESHOLDS` or `STEP_THRESHOLDS` in [common/milestones.py](../common/milestones.py). Deleting `logs/milestones_state.json` resets which thresholds have fired.
