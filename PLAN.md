# BeeBot — Roadmap

## The vision

Build a Bee Swarm Simulator AI that learns to play well **on its own terms** — discovering strategies, finding its own optimal patterns, adapting to whatever the game throws at it. Not a scripted macro dressed up as AI.

- **Minimize hard-coded behavior.** The bot should learn HOW to play. What we script is the bare minimum plumbing — nothing that suggests strategy.
- **Learn every playstyle organically.** Farming, questing, boss-fighting, boost sessions, hive management — RL discovers what works, in what order, and when. No hardcoded "always do X sequence" recipes.
- **Reward the outcomes, not the paths.** Give the model honey/hour and let it invent the path. Don't hardcode "Field Dice → Cloud → Sprinkler" as an ordered recipe just because humans do it.
- **Deliverable: publishable paper.** No published RL work exists on Roblox / macro-scale games / multi-hour reward cycles. This project fills that gap. Every design decision that trades cleanliness for pragmatism gets logged in `EXPERIMENT_LOG.md` for the eventual writeup.

## What is scripted (and what isn't)

**Scripted (permanent — genuine constants, not strategy):**
- Menu button pixel positions where they can't be visually discovered (buy egg, hatch bee)
- Item price tables (data, not decisions)
- Safety wrappers (ESC quit, mouse-corner failsafe, key-release cleanup)
- Screen capture + input plumbing
- HUD reader models (small purpose-built vision models — teach the bot to READ, not to decide)

**Learned (everything else, no exceptions):**
- Farming patterns per field
- When to talk to bears (and which ones)
- When to convert vs keep gathering
- Boost session composition and timing (bot discovers what item combos work)
- Field selection based on current game state
- Boss fight tactics (per boss, learned independently)
- Sprout collection routes
- Route optimization anywhere
- Camera work
- Item usage timing
- Mob dodging / engagement
- Quest engagement (which bears to talk to, when, for which quests)
- Strategic long-term decisions (hive color, save vs spend) — Phase 5

**Scripted safety nets that don't teach strategy:**
- Auto-click through dialogue on stall (prevents indefinite freeze, doesn't tell bot when to engage bears)
- F8 manual pause (human intervention without stopping training)
- Roblox auto-relaunch on crash
- Zombie process cleanup on launch
- OCR outlier filtering (prevents reward function corruption from bad reads)

---

## Phases

### Phase 2a — Imitation bootstrap ✅ DONE

Small CNN + LSTM trained on ~2.5h of recorded gameplay across all starter fields. Warm-starts PPO so it doesn't cold-start from noise.

- LSTM checkpoint saved at `models/beebot_lstm_best.pt`
- CNN backbone architecture reused as `_cnn_backbone()` in `model.py`
- Backbone weights transfer to PPO's feature extractor via `load_backbone_from_lstm_ckpt()`

---

### Phase 2b — Live inference + safety plumbing ✅ DONE

Real-time play loop with all safety nets shipped:

- `rl/env.py` — Gymnasium env wrapping Roblox
- Screen capture via dxcam (with mss fallback)
- Input via pydirectinput + SendInput for mouse (Roblox-compatible)
- ClipCursor for cursor confinement
- Auto-launch BSS + zombie process cleanup (`rl/supervisor.py`)
- Dialogue rescue: multi-template glob, 25-click bursts, 50-step E-suppression
- F8 manual pause hotkey
- Text-trigger popup dismissal system

**Bot can run untouched for 4+ hours without human intervention on non-crash issues** — met.

---

### Phase 2c — HUD readers ✅ MOSTLY DONE

Purpose-built vision models that give RL ground-truth state. Perception, not decisions.

**Built and wired into observation vector + reward function:**
- **Pollen bar %** — template match + OCR `current/max`
- **Honey count** — EasyOCR with comma stripping and robust outlier filtering (decimal-shift detection, persistence-based rebaseline with 60s cooldown, ping-pong protection)
- **Quest tracker** — multi-color panel-open detection (scroll-invariant), bbox-based description→progress line association, per-quest progress deltas fire reward channel when tab is open

**Built but deferred to Phase 4+ (infrastructure in place, returns empty for now):**
- **Buff icon classifier** — 69 wiki-fetched buff templates + multi-scale template matching. Currently returns 0 detections at 0.82 confidence threshold because wiki icons don't render pixel-identically to in-game. Aggregate observation channels (`active_buff_count_norm`, `total_buff_stacks_norm`) exist but stay at 0. Real fix (in-game template snipping) deferred until buff-specific strategy matters (Phase 4 boost cycles).

**Explicitly deferred to later phases:**
- **Boss HP bar detector** — bot has no chance against bosses at 15 bees. Build in Phase 4 when bot is engagement-capable.
- **Ticket count OCR** — low urgency until spending strategy becomes learnable (Phase 5).

Current HUD observation vector is 14 dims (see `HUD_INDEX` in `rl/env.py`). Room to grow.

---

### Phase 3 — RL fine-tuning 🟢 ACTIVE (ready for extended training)

PPO with MultiInputPolicy, warm-started from imitation LSTM CNN backbone. Everything below is CURRENTLY IN PLACE:

**Observation architecture:**
- `Dict({image: (3, 135, 240), hud: (14,)})`
- Custom `BSSMultiInputFeatures` extractor: CNN(image) 512-dim + MLP(hud) 32-dim → concat → linear(544→512) → SB3 policy/value heads
- CNN backbone frozen for first 200k training steps (protects imitation features from PPO gradient noise)

**Action space:**
- `Box` with per-dim bounds
- Keys/mouse: `[0, 1]` (thresholded at 0.5)
- Cursor deltas: `[-1, +1]` (symmetric around zero — critical fix that eliminated pinning-in-top-left drift)

**Reward function (`MultiTimescaleReward`):**
- `W_TICK=1.0` × Δhoney(step) × 1e-4
- `W_MINUTE=0.5` × Δhoney(60s) / 60 × 1e-4  (clamped ≥0)
- `W_HOUR=0.1` × Δhoney(3600s) / 3600 × 1e-4  (clamped ≥0)
- `W_STALL=-0.03` × step after 180s no progress
- **PBRS bag-fill:** `γΦ(s') - Φ(s)` where `Φ = pollen_fill × 1.0` (potential-based shaping, Ng-Harada-Russell invariant)
- **Quest progress:** `W_QUEST_PROGRESS=2.0` × (progress delta / target) when tab is open
- **Quest completion:** `W_QUEST_COMPLETION=5.0` one-time on transition to Complete!

**Hyperparameters:**
- Learning rate 3e-4, batch size 256, N_STEPS 4096, γ=0.99
- `ent_coef=0.005` (lowered from SB3 default 0.02 which was destroying imitation warm-start)
- No VecNormalize (tried, made things worse — rolling std unstable in our sparse-with-spikes env)

**Safety nets composed:**
- Dialogue rescue (multi-template, 25-click bursts, E-suppression)
- F8 pause
- Cursor clip
- OCR outlier filtering (persistence + cooldown)

**Move-on criteria:**
- Bot honey/hour exceeds imitation baseline by 50%+ over a 4-hour clean run
- Bot autonomously opens quest tab at some detectable cadence
- Bot self-recovers from stalls without user intervention for 90%+ of session time
- `explained_variance` sustained positive across 4+ iterations at ep_rew_mean > +1.0

**Next dev work only if metrics stall:** more HUD readers, RecurrentPPO (LSTM in PPO policy), dynamic PBRS scale, KL-to-BC penalty.

---

### Phase 4 — Deep specialization + autonomous progression 🔴 FUTURE

Bee count 15 → 25 → 30 → 35+ progression. Boss engagement. Once Phase 3 bot demonstrates real farming skill, unlock:

- Per-boss learned sub-policies (King Beetle, Coconut Crab, Tunnel Bear, Stump Snail)
- Ant Challenge (5-min episodes — natural PPO training substrate)
- Sprout collection routes
- Planter management (3-planter scheduling)
- Night events (fireflies, moon sprouts, werewolf)
- Boost session emergence (Rage → Focus → Melody rotation — learned, not scripted)

Buff observation becomes strategically valuable here — in-game buff template snipping happens now.

**Move-on criteria:**
- Bot beats King Beetle without human intervention
- Bot reaches 25 bees on its own
- Bot handles Ant Challenge with positive amulet quality trajectory

---

### Phase 5 — Long-horizon strategy + VLM 🔴 FUTURE

The hardest problems:
- **Semantic quest understanding via VLM** — bot can currently learn "quest progress = reward" but NOT "this quest wants Blue Pollen from Clover Field". VLM (vision-language model) reads quest text, translates into structured goals for RL.
- **Long-window strategic decisions** — hive color commitment, Star Jelly gift targets, wax choice decisions. Extended-horizon reward + small decision head.
- **Bucko/Riley questlines Q100+** — repeatable quest chains with escalating rewards.
- **Post-training content adaptation** — VLM reads new item descriptions after game updates, no retraining needed.

---

## Current state (2026-08-15 evening)

**Fredrick's in-game state** (see `project_beebot_fredrick_state` memory):
- ~15 bees, Honey Bee Gate highest unlock
- Honey range ~3.5M-4.3M
- Honey Dipper tool, no amulets
- Bag capacity 405,625 pollen
- Zero quest turn-ins completed (bot has never successfully engaged a bear)
- Farms Sunflower / Dandelion / Mushroom / Blue Flower

**Code state:**
- All Phase 2b and 2c infrastructure done except boss HP + tickets (deferred)
- Phase 3 training pipeline is bug-free and ready for extended runs
- `EXPERIMENT_LOG.md` has full chronological record for paper writeup

**Ready for:** long training runs. No more dev work required to make progress — bot needs TIME with the current setup to learn.

---

## Timeline predictions (revised)

Original estimates were mostly right on the phases but off on Phase 2c (didn't account for how many iterations OCR robustness would need). New realistic table:

| Phase | Status | Realistic remaining calendar |
|---|---|---|
| 2a | ✅ done | — |
| 2b | ✅ done | — |
| 2c | ✅ mostly done (buffs+boss+tickets deferred) | — |
| **3** | 🟢 **active, ready for extended training** | **1-3 months of training-uptime-dominated calendar** |
| 4 | 🔴 future | 4-6 months (dev + training) |
| 5 | 🔴 future | 3-6 months (VLM integration is significant) |

**Total realistic horizon:** ~10-15 months from now to "bot plays most of the game autonomously."

**Calendar time in Phase 3 depends on training uptime:**
- 24/7: ~1 month to Phase 3 move-on criteria
- 8 hrs/night: ~3 months
- Weekends only: ~10 months

---

## Non-negotiables

1. **Never hardcode strategic sequences.** If it feels like it belongs in a macro tool, it doesn't belong here. See `feedback_beebot_pure_rl_vision`.
2. **Always give RL the choice.** Safety nets prevent indefinite freezes; they don't remove decisions.
3. **Measure everything against honey/hour.** Reward shaping is fine, but the ultimate benchmark is what the bot actually achieves.
4. **Don't rewrite requirements once training data is collected.** New action keys, new obs channels require careful handling (fresh policy).
5. **Verify BSS facts before asserting.** Wiki summaries via agents have been unreliable. Ask user or fetch specific wiki page directly. See `feedback_verify_bss_facts`.
6. **Document every substantive change in `EXPERIMENT_LOG.md`.** Paper depends on it.

---

## What each phase looks like when "working"

**Phase 2b working:** ✅ Bot runs for hours without needing attention. Dialogue traps get auto-rescued. Roblox crashes get auto-recovered.

**Phase 2c working:** ✅ Bot's HUD observation includes pollen/honey/quest state. Reward function reacts to quest progress signals.

**Phase 3 working:** Over multiple training sessions, `ep_rew_mean` trends up week over week. Bot's farming becomes visibly more efficient. Bot starts opening the quest tab occasionally. Convert cycles become reliable. `explained_variance` climbs into +0.1 to +0.5 range sustained.

**Phase 4 working:** Bot beats King Beetle first-try, farms through gates, doesn't need intervention for days at a time.

**Phase 5 working:** Bot makes strategic calls (hive color, big purchases) that don't disagree with your check-ins. Handles game updates without breaking.
