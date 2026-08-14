# BeeBot — Roadmap

## The vision

Build a Bee Swarm Simulator AI that learns to play well **on its own terms** — discovering strategies, finding its own optimal patterns, adapting to whatever the game throws at it. Not a scripted macro dressed up as AI.

- **Minimize hard-coded behavior.** The bot should learn HOW to play. What we script is the bare minimum plumbing — nothing that suggests strategy.
- **Learn every playstyle organically.** Farming, questing, boss-fighting, boost sessions, hive management — RL discovers what works, in what order, and when. No hardcoded "always do X sequence" recipes.
- **Reward the outcomes, not the paths.** Give the model honey/hour and let it invent the path. Don't hardcode "Field Dice → Cloud → Sprinkler" as an ordered recipe just because humans do it.

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
- Strategic long-term decisions (hive color, save vs spend) — Phase 5

**Scripted safety nets that don't teach strategy:**
- Auto-click through dialogue on stall (prevents indefinite freeze, doesn't tell bot when to engage bears)
- Roblox auto-relaunch on crash
- Zombie process cleanup on launch

## Phases (revised — minimize-scripting focus)

### Phase 2a — Imitation bootstrap ✅ DONE

Small CNN + LSTM trained on ~2.5h of your recorded gameplay. Gives RL a warm-start policy so it doesn't start from pure noise.

**Move-on criteria:** LSTM checkpoint saved, model produces coherent farming behavior (walks in field, swings scoop) even if imperfect.

---

### Phase 2b — Live inference + minimal scripted plumbing 🟡 IN PROGRESS

Real-time play loop. Model reads screen → predicts actions → executes via `robo_input`. Cursor clip zones, whitelist filter, stall-rescue clicks — all safety nets, NOT strategy.

**In scope:**
- `play.py` and `rl/env.py` runtime
- LSTM hidden state management
- Cursor exclusion zones (title bar, top-bar dropdown)
- Auto-click-through-dialogue on stall (template-based, only fires when dialogue visible)
- Roblox auto-relaunch + zombie process cleanup

**Explicitly NOT in scope (removed from earlier plan):**
- ~~Scripted boost cycles~~ — boosting is strategy, must be learned
- ~~Scripted quest cycles~~ — bear engagement is strategy, must be learned  
- ~~Scripted "buy egg on honey threshold"~~ — spending decisions are strategy, learned in Phase 5
- ~~Recorded path playback~~ — navigation is learned via RL

**Move-on criteria:**
- Bot can run untouched for 4+ hours without human intervention on non-crash issues
- Dialogue-rescue click works when triggered
- Roblox auto-recovers from crashes at least once during a test session

---

### Phase 2c — HUD readers 🟡 PARTIAL

Purpose-built vision models that let RL SEE game state. NOT decision-makers — just eyes.

**Built:**
- Pollen bar % (template match + OCR fill measurement)
- Honey count (EasyOCR)

**To build:**
- Ticket count OCR (needs bot to open menu — build with quest tracker)
- Buff icon strip classifier (detects active Field Boost, Haste, Focus, etc. — informs RL "what buffs are active" as observation)
- Boss HP bar detector (know when a boss fight is happening)
- Quest tracker OCR (parse text on left side — enables RL to see quest progress and get reward on quest advancement)

**Move-on criteria:**
- All 5 HUD reader modules working with acceptable accuracy (>85%)
- RL observation includes buff state + quest progress
- Reward function can distinguish quest progress from pure honey gain

---

### Phase 3 — RL fine-tuning (the main event) 🟡 STARTED

PPO with CNN feature extractor, warm-started from imitation LSTM. Multi-timescale honey reward with shaped bonuses for quest progress + buff engagement. This is where the bot ACTUALLY learns to play well.

**Reward function surface (as HUD readers land):**
- Δ honey (per tick, minute, hour) — primary signal
- Quest progress (once quest tracker OCR works) — encourages bear interaction
- Buff-active bonuses — rewards engaging with boost items when opportunity present
- Stall penalty (mild) — discourages doing nothing productive

**No scripted "do this then that" sequences.** RL discovers boost cycles, quest cycles, farm routes.

**Move-on criteria:**
- Bot honey/hour exceeds recorded imitation baseline by 50%+
- Bot autonomously engages bears and completes at least some simple quests
- Bot autonomously uses items (dice, potions) when it has them
- No repeated stalls > 5 min per hour of training

---

### Phase 4 — Deep specialization + autonomous progression

Multiple heads of specialized RL — a distinct sub-policy for each activity that has enough distinct patterns to warrant its own learning:
- Boss fights (per boss: King Beetle, Stump Snail, Tunnel Bear, Coconut Crab)
- Ant Challenge
- Sprout collection
- Planter management
- Night events

The top-level policy from Phase 3 learns WHEN to switch into a specialized sub-policy. Both levels learned, not scripted.

Also: bot autonomously progresses through bee-count gates (15 → 25 → 35+).

**Move-on criteria:**
- Bot beats King Beetle without human intervention
- Bot progresses through at least the 15-bee gate on its own
- Bot handles all major activity types (farm, quest, boss, sprout, planter) autonomously

---

### Phase 5 — Long-term strategy + post-training content adaptation

The remaining hardest problems:
- **Long-term strategic decisions** — hive color commitment, save-vs-spend, which bee to gift with Star Treat. Learned via extended-window rewards (per-day, per-week) and a small strategic-decision policy.
- **VLM integration** — vision-language model reads novel item descriptions, updates item database automatically. Lets bot adapt to game updates without retraining.
- **Bucko/Riley questlines** — extend RL to handle long quest chains once quest tracker OCR is solid.

**Move-on criteria:**
- Bot commits to a hive color autonomously and stays consistent
- Bot handles new items (added post-training) via VLM lookup
- Bot progresses through Bucko or Riley to Q100+

---

## Timeline predictions

**Per-phase estimates.** Split into DEV TIME (my code + your testing) and TRAINING TIME (bot playing autonomously to accumulate learning). Real calendar time = whichever is longer.

| Phase | Dev time | Training time | Realistic calendar | Move-on criteria |
|---|---|---|---|---|
| 2a | ~ done | ~ done | ~ done | LSTM produces coherent behavior |
| 2b | 1-2 weeks | — | 1-2 weeks | 4+ hour unattended runs stable |
| 2c | 2-3 weeks | — | 2-3 weeks | 5 HUD readers accurate |
| 3 | 1 week (reward tuning) | 200-500 hrs | 1-3 months | 50%+ over imitation baseline |
| 4 | 2-4 weeks per module × 5 modules | 100-200 hrs per module | 4-6 months | Beats KB, progresses to 15 bees |
| 5 | 2-3 months | 500+ hrs | 3-6 months | Autonomous hive color, VLM works, Q100+ |

**Total realistic horizon:** **10-18 months** from now to "bot plays most of the game autonomously."

**Calendar time depends heavily on training uptime.** Rough:
- 24/7 training: fastest, but hardware/electricity commitment
- 8 hrs/night (overnight only): ~3x slower calendar time, most livable
- Weekends only: ~10x slower, but still gets there

## Non-negotiables

1. **Never hardcode strategic sequences.** If it feels like it belongs in a macro tool, it doesn't belong here.
2. **Always give RL the choice.** Safety nets prevent indefinite freezes; they don't remove decisions.
3. **Measure everything against the honey/hour north star.** Reward shaping is fine (quest progress, buff engagement) but the ultimate benchmark is what the bot actually achieves.
4. **Don't rewrite requirements once training data is collected.** New action keys, new obs channels — those require re-recording. Additions to reward or safety are fine anytime.

## What each phase looks like when "working"

**Phase 2b working:** Bot runs for hours without needing your attention. Occasional dialogue traps get auto-rescued. Roblox crashes get auto-recovered. You come back after 4 hrs and honey is meaningfully higher than when you left.

**Phase 2c working:** Bot's env prints show buff icons detected, quest progress detected, boss HP visible when engaged. Reward function reacts to these signals.

**Phase 3 working:** Over multiple training sessions, ep_rew_mean trends up week over week. Bot's farming becomes visibly more efficient. Bot starts talking to bears without you nudging it. Boss fights (initially disaster) become survivable.

**Phase 4 working:** Bot beats King Beetle first-try, farms through gates, doesn't need you to intervene for days at a time.

**Phase 5 working:** Bot makes strategic calls (hive color, big purchases) that you don't disagree with when you check in weekly. Handles game updates without breaking.
