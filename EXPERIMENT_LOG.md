# BeeBot Experiment Log

**Purpose:** chronological record of experiments, design decisions, and results — feeds directly into the paper's methodology + results sections. Log EVERY substantive change with rationale, empirical impact, and failed approaches.

**Format per entry:**
- Date + short title
- **Motivation:** what problem we were solving
- **Approach:** what we tried, with enough detail to reproduce
- **Result:** empirical outcome (metrics, behavior change, or "hypothesis unverified")
- **Learning:** what generalizes; what we'd do differently

**Paper thesis (as of 2026-08-15):** how far can almost-pure RL (PPO with potential-based shaping and CNN+scalar hybrid observations, warm-started from behavior cloning) scale to a commercial multi-hour-reward-cycle game (Bee Swarm Simulator, Roblox)? No published RL work exists on Roblox games, macro-scale games, or games with multi-hour reward cycles — this fills that gap.

---

## Phase 2a — Imitation bootstrap (complete, pre-conversation)

**Motivation:** cold-start PPO in a sparse-reward continuous-play game is intractable at hobby scale. Warm-start the policy from human demonstration.

**Approach:**
- Recorded ~2.5 hours of author gameplay across multiple starter fields (Sunflower, Dandelion, Mushroom, Blue Flower) using `record_gameplay.py`
- Captured: 3×135×240 downsized RGB frames + labels {keys (11 tracked), mouse buttons, cursor position normalized [0,1]}
- Trained small CNN + LSTM (`BeeBotLSTM` in model.py) with per-frame BCE for keys/mouse, MSE for cursor
- Backbone: 4-layer CNN (Conv2d 3→32→64→128→128) + AdaptiveAvgPool → 4096-dim embedding

**Result:** CNN backbone learned useful visual features (flower/hive/mob discrimination visible in embeddings); LSTM produced coherent farming behavior.

**Learning:** LSTM temporal context significantly reduced per-frame action jitter vs feed-forward CNN alone. Warm-startable CNN backbone was the key architectural choice — everything downstream reuses it.

---

## Phase 2b — Live inference + safety plumbing (complete)

**Motivation:** run the imitation policy live against Roblox to (a) validate the imitation model works in a game environment, (b) build the infrastructure PPO would need later.

### Screen capture
- Started with `mss` (CPU-side capture)
- Switched to `dxcam` (GPU-accelerated via DirectX Desktop Duplication API) — ~3× faster
- Kept mss as fallback for RDP configurations where dxcam fails

### Input
- Started with pyautogui — **did not work** in Roblox. pyautogui uses standard virtual-key events; Roblox filters those.
- Switched to `pydirectinput` for keyboard — worked
- **Mouse teleport bug:** `SetCursorPos` moved cursor visually but Roblox's internal hover state didn't update — clicks landed at old position
- Fix: `SendInput` with `MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK`, coordinates normalized to 0..65535 across virtual desktop (multi-monitor safe)

### Cursor confinement
- Windows `ClipCursor` API — confines cursor to Roblox window rectangle
- Silently released on focus changes; re-applied every step

### Auto-launch + crash recovery
- `supervisor.py` — detects if Roblox is running with BSS loaded, launches via `https://www.roblox.com/games/start?placeId=X` (browser-based; the raw `roblox://` scheme opens tray without joining)
- Zombie-cleanup on startup: kills lingering Roblox processes owned by current user (fixes "mutex already held" errors)

### Dialogue rescue (safety net)
Multiple iterations — see 2026-08-14 entries below for the story.

### Manual pause (F8)
- Hold F8 → releases all keys, releases cursor clip, skips action-apply
- Rollout keeps running (obs/reward/value estimates still computed)
- Doesn't require restarting training to reposition the bot

**Learning:** every input path in Roblox needs its own gotcha discovery — the game's anti-cheat filters plenty of "normal" Windows APIs. `pydirectinput` + `SendInput`/`ABSOLUTE` is the working stack.

---

## Phase 2c — HUD readers (partial → in progress 2026-08-15)

**Motivation:** RL needs to READ game state (pollen, honey, buffs, quest progress, boss HP) to compute reward AND to condition its policy on state variables the pixels alone don't reliably encode.

### Built so far
- **Pollen bar %:** template match to locate the bar, OCR the numeric current/max
- **Honey count:** EasyOCR on the top-center honey display, with comma stripping
- **Quest tracker OCR (2026-08-15):** template match to detect tab-open state, EasyOCR the panel region, regex-parse each visible quest into {name, current, target, complete}. Rate-limited to every 50 steps (~10 sec) since it's expensive and quests move slowly.

### Planned (Phase 2c continuation — pending user materials)
- Buff icon classifier (Haste, Focus, Rage, Melody, Mark, etc. with stack counts) — waiting on user screenshot with active buffs
- Boss HP bar detector — deferred (bot at 15 bees, no boss engagement yet)
- Ticket count OCR — deferred (low urgency at current stage)

**Design principle:** HUD readers are PERCEPTION, not decision-makers. They produce observation channels the policy reads, and provide reward signals for the reward function. They do NOT decide what to do — that's the RL policy's job. Per [almost-pure-RL vision](https://github.com/HarlanBarlan/BeeBot/blob/main/PLAN.md).

---

## Phase 3 — RL fine-tuning (in progress, main work of 2026-08-14 → 15)

### 2026-08-13/14: Initial PPO setup + reward function

**Motivation:** get PPO training on top of the imitation warm-start.

**Approach:**
- Stable-Baselines3 PPO with custom CNN feature extractor (`BSSCnnFeatures`) matching imitation model's backbone
- Warm-start CNN backbone from imitation checkpoint (loads matching `backbone.*` weights, leaves head + LSTM fresh)
- Custom `MultiTimescaleReward`: honey_delta at per-tick + rolling 1-min + rolling 1-hour windows + stall penalty
- OCR outlier protection: reject honey deltas > 100k/tick
- Action space: `Box(0, 1)` with cursor interpretation `(action - 0.5) × 2 × MAX_DELTA`
- Artificial episode boundary every 1024 steps so SB3's Monitor can compute `ep_rew_mean`

**Result:**
- `explained_variance` bounced -10 to +0.5, `ep_rew_mean` oscillated 0.3 to -1.5 with no clear trend
- `value_loss` tiny (0.0001-0.03), `entropy_loss` climbing (bad direction)
- Bot learned SOME farming but was fundamentally not converging

**Learning:** value function couldn't fit sparse reward. Diagnosis pointed to two coordinated issues: pixels-only observation + reward density too low. Fixes came in later entries.

---

### 2026-08-14: Deep research pass on sparse-reward PPO

**Motivation:** before making architectural changes, understand what production RL agents actually do for sparse reward in continuous-play games.

**Approach:** subagent research pass covering:
- OpenAI Five (Dota 2, arXiv 1912.06680): 28-channel reward with team-spirit blending
- AlphaStar (StarCraft II): 3 stacked reward layers — ternary win/loss + Blizzard score + pseudo-rewards
- MineRL Diamond: winning solutions used imitation warm-start + milestone rewards
- Andrychowicz et al. 2020 "What Matters in On-Policy RL": accurate state features are top-5 win
- Ng/Harada/Russell 1999: potential-based shaping preserves optimal policy
- Devlin/Kudenko 2012: dynamic potential functions extend to non-stationary Φ

**Result:** clear recipe — dense proxy carries the value function while sparse true reward keeps optimal policy. HUD scalars in observation is the highest-leverage single change. PBRS densifies the reward. Together they fix critic starvation.

**Learning:** no published RL work exists on Roblox / macro-scale games / multi-hour cycles — reinforced the paper thesis. Sub-agents occasionally hallucinated arXiv IDs; core citations (1999, 2003, 2012, 2020, PIRLNav) are all real and load-bearing.

---

### 2026-08-14: HUD scalars in observation (Sequential #1) — commit `6fa51bb`

**Motivation:** critic couldn't fit sparse reward because it had to visually parse pollen%/honey from downsized pixels while ALSO fitting the reward. Give it ground-truth scalars.

**Approach:**
- Observation changed from `Box(3, 135, 240)` to `Dict{image: Box(3,135,240), hud: Box(-2,+2, shape=(8,))}`
- 8-dim HUD vector: `pollen_fill`, log-normalized `honey`, 60s deltas of both, time-since-last-gain, time-since-pollen-change, per-source validity flags
- New feature extractor `BSSMultiInputFeatures`: CNN(image) 512-dim + MLP(hud) 32-dim → concat → linear(544→512)
- Switched from `MlpPolicy` to `MultiInputPolicy`
- Preserved warm-start: CNN backbone weights transfer, policy head fresh

**Result:**
- Explained_variance touched +0.513 in the first run (previously never positive)
- ep_rew_mean climbed to +3-5 range
- Value function finally had fittable data

**Learning:** confirmed the research pass's top recommendation. Adding scalar HUD to observation is the biggest single lift for sparse-reward continuous-play RL.

---

### 2026-08-14: Potential-based bag-fill reward (Sequential #2) — commit `acae50e`

**Motivation:** honey reward fires only during rare conversion events. Densify by adding a smooth signal for pollen bag fill.

**Approach:**
- Φ(s) = pollen_fill × PBRS_SCALE (SCALE=1.0)
- F(s, a, s') = γ·Φ(s') - Φ(s) added to base reward per step
- γ=0.99 matches PPO's gamma (required for Ng invariance theorem)
- Chose static PBRS_SCALE for MVP; dynamic HPP estimation deferred (Devlin-safe implementation would need HPP freeze per rollout)

**Result:** every step of bag filling now generates ~+0.05 shaping reward, comparable to per-tick honey reward magnitude. Full fill-convert cycle telescopes to near-zero net PBRS contribution — cumulative reward unchanged, only per-step shape differs.

**Learning:** simple potential-based shaping works. Dynamic Φ is theoretically better but adds complexity we didn't need to prove out the concept.

---

### 2026-08-14: Bundle of three quick wins — commit `cfe7525`

**Motivation:** research pass identified three quick, compose-cleanly improvements.

**Approach:**
1. **`ent_coef` 0.02 → 0.005** — original value was actively destroying the imitation warm-start (entropy_loss was climbing over the run)
2. **`N_STEPS` 1024 → 4096** — at 5 fps, 1024 steps ≈ 3.4 min = one convert event per rollout; 4096 = ~14 min = 3-4 events
3. **CnnFreezeCallback** — freeze CNN backbone for first 200k timesteps so early PPO gradient noise doesn't destroy imitation-learned visual features (PIRLNav 2023 recommended workflow)

**Result:** ep_rew_mean rose from ~0 to +3+, explained_variance climbed steadily.

**Learning:** the `ent_coef` default (0.02) is too high for warm-started PPO. Standard BC-warm-start range is 0.001-0.005.

---

### 2026-08-14: Dialogue rescue evolution

Multi-step story:

**v0:** Single click on template match at threshold 0.65. Bot got stuck in dialogues for hours.

**v1 — E suppression (`0ab49ab`):** After successful click, suppress E for 50 steps. Hypothesis: bot re-triggers dialogue by pressing E while still in NPC range. Didn't fully fix — user later diagnosed some dialogues have multiple lines requiring multiple clicks.

**v2 — Rapid burst (`c0639e0`):** 10 clicks per firing, 50ms apart. Better but still failed on long quest chains.

**v3 — Bigger burst (`3539bb4`):** 25 clicks per firing, 80ms apart. Handles most dialogue chains.

**v4 — Multiple templates (`7ffe8fc`):** Glob `dialogue_continue*.png` — user can snip separate templates per bear (Black Bear, Mother Bear, etc.). Log best-observed confidence when no template matches, tells user what to snip next.

**Learning:** dialogue rescue is a per-bear affordance problem. Real fix will come when Phase 2c quest tracker gives us richer state to detect "we're in a dialogue" without template matching at all. Meanwhile, per-bear templates are the workaround.

---

### 2026-08-14: Robust OCR outlier handling — commits `a208cb0`, `fa6ca03`, `866d4a8`

Multi-step story:

**v0:** any honey delta > 100k rejected forever. Broke on real state changes (spending honey → permanent reward-function failure).

**v1 (`a208cb0`) — persistence-based rebaseline:** after 5 consecutive same-direction outliers, accept as real state change and re-baseline. Simulation showed working. Also clamped `delta_minute` and `delta_hour` to non-negative so spending events don't punish.

**Discovered bug — spurious huge reward (`fa6ca03`):** after re-baseline, `honey_history` still contained pre-baseline values. Rolling 60s window computed `current(31.2M) - past(3.1M) = +28M` for 60 seconds, contributing ~+23 reward per step. **Produced `total_reward=3649` in one 30-min test — corrupted the value function.** Fix: clear honey_history on re-baseline.

**Also (`fa6ca03`):** widened decimal-shift detection from just 10x to also cover 100x, 1000x, 10000x ratios (comma-truncation OCR errors). Added Monitor wrapping (had been silently missing after VecNormalize wrap, breaking ep_rew_mean tracking).

**Discovered bug — ping-pong (`866d4a8`):** systematic OCR errors (persistent 34M misreads of 3.4M) triggered false re-baseline; then real value returning caused opposite re-baseline; then bad reading again caused another. Cycle. Fix: bumped persistence threshold to 15 consecutive reads AND added 60-sec cooldown between re-baseline events.

**Learning:** OCR is not just noisy — it produces systematic errors (period-vs-comma parse failures) that persist for many frames. Simple filtering isn't enough; need temporal reasoning about baseline consistency. Also: value function corruption is CATASTROPHIC — one 3649-reward episode's gradient damage takes many iterations to recover from.

---

### 2026-08-14: VecNormalize attempted and reverted — commits `3539bb4`, `39f9b36`

**Motivation:** research pass ranked reward normalization among top-5 design choices.

**Approach:** wrapped env in VecNormalize(norm_obs=False, norm_reward=True).

**Result — REGRESSION:**
- Hour-4 clean run without VecNormalize: ep_rew_mean +3.05 → +1.82 → +2.01 → +1.67, EV climbing to +0.156
- With VecNormalize: ep_rew_mean -1.98 → -1.57, EV -4.24

**Learning:** rolling std of returns is unstable in our env because reward magnitudes vary wildly (mostly small, occasional large conversion spike). Value function chases moving target. VecNormalize's benefit assumes reward-magnitude stability we don't have. **Reverted.** Kept Monitor wrapping fix as separate improvement.

**Publishable insight:** VecNormalize is standard practice but assumes reward-magnitude stationarity. In our multi-hour-cycle env with occasional huge spike events, this assumption fails.

---

### 2026-08-15: Cursor drift root-cause fix — commit `f774c5e`

**Motivation:** user reported bot cursor chronically stuck in top-left ~1/8 of screen, causing unwanted clicks on Roblox More menu, Voice Chat button, BSS quest button, etc. Also prevented RL from ever learning cursor positions needed for other behaviors.

**Failed workarounds first:**
- **Bigger exclusion zones** (top 50→150, left 30→100): pushed unusable area OUT of reachable zone. User (correctly) pointed out this is hardcoding — bot needs to eventually click those UI elements.
- **Periodic cursor recenter safety net**: workaround, not root cause. User: "figure out the real reason and fix it, don't work around it."

**Root cause diagnosis:** PPO's continuous action head is Gaussian with default init mean ~0 and std ~1. Our action space was `Box(low=0, high=1)` for all dims including cursor. Gaussian(0, 1) clipped to [0, 1] has effective mean ~0.16 (most samples ≤0 clip to 0). Our code interpreted 0.5 as neutral cursor delta, so effective average delta per step was `(0.16 - 0.5) × 2 × MAX ≈ -0.68 × MAX`. Every step, cursor drifted ~55 pixels toward top-left. RL couldn't self-correct because cursor position doesn't earn direct reward.

**Fix:** cursor dims changed to `Box(-1, +1)` symmetric around zero. Gaussian(0, 1) clipped to [-1, +1] has effective mean ≈ 0 (verified via simulation). Cursor delta interpretation: `action × MAX_DELTA` (direct, no offset). Keys/mouse stay `[0, 1]` (binary threshold, slight bias-to-off is fine).

**Result:** cursor no longer drifts. Bot's dialogue rescue clicks fired at (1139, 866) and (1142, 867) — actual bear dialogue positions in the middle-right of screen — vs old top-left stuck positions.

**Learning — MOST IMPORTANT DESIGN LESSON SO FAR:** almost-pure-RL means the ACTION SPACE must be RL-friendly, not just the reward function. Symmetric action-space bounds matter when the policy is Gaussian. This is an invisible bias baked into action-space design that RL can never fix. **Prime paper example** of an implementation detail that dominates outcome.

**Publishable insight:** the "asymmetric bounds trap" — using `Box(0, 1)` with 0.5-centered interpretation for continuous-delta actions is a common but subtly wrong choice for on-policy Gaussian-output algorithms.

---

### 2026-08-15: Bulk-fetched buff icon templates from wiki

**Motivation:** manually snipping ~40 buff templates per-buff-encountered is tedious. Wiki has all buff icons on the Buffs & Debuffs page. Automate the download.

**Approach — MediaWiki API:**
- Fandom's regular wiki HTML endpoint returns HTTP 403 to programmatic fetches even with browser User-Agent (Cloudflare/anti-bot)
- Fandom's MediaWiki `api.php` endpoint accepts programmatic requests and responds fine
- Script `scripts/fetch_wiki_buff_icons.py`:
  1. Uses `action=parse&prop=images` to list all images on the Buffs & Debuffs page (84 images total)
  2. Uses `action=query&prop=imageinfo` in batches of 50 to resolve each image to its CDN URL + dimensions
  3. Filters: skip too-big (>1024px), too-small (<16px), non-square-ish (aspect ratio >2.5), and screenshots/wiki-chrome (regex on filename)
  4. Downloads remaining to `hud/probes/buff_<slug>.png`

**Result:** 80 buff templates downloaded, including all the core game buffs (Haste, Focus, Rage, Melody, Boost, Science Enhancement, etc.).

**Gotcha caught during dev:** MediaWiki normalizes titles (underscores → spaces) between request and response. My initial code keyed the result dict by the returned title, so lookups from the original underscore-form filenames all missed. Fix: use the `normalized` field in the API response to map returned titles back to the original request form.

**Scale mismatch:** wiki icons are 120-225px, in-game buff icons are ~40-60px. Added multi-scale template matching to `BuffClassifier` (tries scales 0.2-1.0). Slower but necessary for wiki-sourced templates. If user later snips their own templates at in-game size, the 1.0 scale hits immediately.

**Publishable insight for paper:** wiki-sourced templates give a substantial cold-start for buff detection without manual snipping. Roblox games with active Fandom wikis have this data source available; other RL-in-commercial-games work should exploit similar sources when they exist.

---

### 2026-08-15: Buff icon classifier (MVP)

**Motivation:** buffs are a real strategic signal — bots at midgame need to correlate buff-active windows with better farming outcomes. Adding buffs to the observation lets the RL policy learn buff-aware behavior (harvest more aggressively during Field Boost, etc.).

**Approach — same progressive-template pattern as dialogue rescue:**
- `hud/buff_classifier.py`: BuffClassifier
- Loads all templates matching `hud/probes/buff_*.png` (glob)
- For each frame, template-matches each buff type in the top-left buff strip region
- Nearby OCR extracts stack count ("xN" overlay)
- Works with 0 templates (returns empty list) OR many — user snips as encountered

**Buff strip location:** top-left of screen, above the row of side-panel icons (egg/quest/bee/badge/settings/robux). User confirmed via screenshot 2026-08-15. Layout uses fixed-ratio defaults with optional `buff_strip_region.png` anchor template for precise dynamic location.

**MVP scope:**
- Detect PRESENCE + stack count per known buff type
- Skip time-remaining / duration measurement (deferred to Phase 4+ when strategic decisions need it)
- Skip specific-buff observation channels — MVP uses aggregate `active_buff_count_norm` + `total_buff_stacks_norm`. RL policy must learn buff correlation via honey rate rather than knowing "haste vs focus" specifically. Aligned with almost-pure-RL vision.

**Observation additions:** `active_buff_count_norm`, `total_buff_stacks_norm` (dims 12-13). HUD_DIM went 12→14.

**Rate limit:** 25 steps (~5 sec) — faster than quest (buffs expire within seconds) but slower than pollen/honey.

**No dedicated reward channel:** bot learns buff value INDIRECTLY via honey rate improvement during buff windows. If bot ends up ignoring token pickups after long training, we can add a small "new buff appeared" reward — but starting minimal per pure-RL vision.

**User must snip templates before this works:** `hud/probes/buff_haste.png`, `buff_focus.png`, `buff_rage.png`, etc. as they encounter each buff type. Working set can grow over time.

**BREAKING:** HUD_DIM changed 12 → 14, obs space shape changed again. Fresh training needed (auto-fallback on checkpoint load).

---

### 2026-08-15: Quest tracker OCR + progress reward channel

**Motivation:** unlock a new reward signal for quest progress — bot's biggest untapped learning direction. Also add meta-behavior "check quest tab periodically" as an emergent RL target.

**Design (Option A per user — fully learned):**
- Quest tab only visible when bot clicks the map icon (top-left)
- OCR the panel when tab is open, extract each quest's {name-like text, current, target, complete}
- Reward fires ONLY when tab is open AND we detect progress delta on a tracked quest
- Bot must LEARN to open the tab periodically to earn quest rewards (natural incentive)
- Strategy of WHICH quests to prioritize is left for Phase 4+ (bot picks fields based on quest state)

**Approach:**
- `hud/quest_ocr.py`: template match on `quest_tab_indicator.png` → detect open state → OCR panel region (relative to template) → parse quests with `POLLEN_PAIR_RE`-style regex
- `hud/reader.py`: separate `read_quest()` method so env can rate-limit independently from pollen/honey
- `rl/env.py`: quest OCR runs every 50 steps (~10 sec) — expensive, quests move slowly. Added 4 quest scalars to HUD observation vector (`quest_tab_open`, `active_quest_count_norm`, `completed_quest_count_norm`, `time_since_quest_tab_seen`)
- `rl/reward.py`: new `W_QUEST_PROGRESS=2.0` (per full-target progress) and `W_QUEST_COMPLETION=5.0` (one-time bonus). Quest snapshot keyed by `quest_key` (text with progress numbers stripped) for stable identity across OCR reads.

**Bug caught in testing:** initial implementation keyed the snapshot by `raw_line` which INCLUDED the current progress numbers — so every OCR read looked like a "new quest" and progress delta never fired. Fix: added `_strip_progress()` helper that removes the "N/M" and "Complete!" from the OCR text, uses the remaining text as identity key.

**Verified via simulation:**
- Quest 100/1000 → 200/1000 with tab open: reward +0.20 (matches `W_QUEST_PROGRESS * 100/1000 = 0.2`)
- Tab closed between reads: no quest reward
- Quest 200/1000 → 500/1000 after tab reopened: reward +0.60 (delta accumulated correctly)
- Quest completion transition: +5.00 (matches `W_QUEST_COMPLETION`)

**User must snip template before this works:** `hud/probes/quest_tab_indicator.png` — a stable visual element only visible when the quest tab is open (e.g., a category header or the panel background).

**Learning — key design pattern:** persistent-menu HUD reading requires a separate "is menu open" gate and stable per-item identity keys across reads (raw text changes with progress). This pattern will repeat for other menus (backpack, hive, shop UIs).

---

## Phase 4 — future

Deep specialization + autonomous progression through bee gates. Waiting on Phase 2c HUD readers to unlock quest / buff / boss reward channels.

## Phase 5 — future

Long-term strategic decisions + VLM integration.

---

## Failed approaches inventory (paper "what didn't work" section)

- **Naive outlier rejection** (reject-and-hope): permanently broke on real state changes
- **VecNormalize reward normalization**: rolling std unstable in our sparse-with-spikes env
- **Cursor exclusion zones for problem UI**: hardcodes what bot can and can't touch, violates "learn organically" principle
- **Periodic cursor recenter safety net**: masks symptom of action-space bias, doesn't fix root cause
- **Single-template dialogue rescue at threshold 0.65**: fails per-bear because dialogue box visuals differ
- **1-click-per-rescue dialogue advance**: BSS dialogues need multiple clicks per chain

---

## Open questions / to-investigate

- Would RecurrentPPO (LSTM-in-PPO from sb3_contrib) help with temporal action smoothing and hive identification?
- What's the actual pollen→honey conversion rate for a dynamic HPP estimate in PBRS?
- Can we detect "in a dialogue" via non-template method (movement blocked, HUD partial-occlusion) to eliminate template-per-bear brittleness?
- How does bot behavior change across bee gates (15 → 25 → 30 → 35) once Phase 2c reward channels enable strategic learning?
