"""
Reward function for BSS RL agent.

Multi-timescale reward:
  - Δhoney per tick (fast, dense feedback)
  - Δhoney per minute (smooths noise)
  - Δhoney per hour (values sustained farming rates)
  - Curiosity bonus (visits to novel visual states)
  - Stall penalty (0-change state = bad)
  - Item pickup bonuses (from token_values × goal_profiles)

The blended reward is what PPO's advantage function backpropagates.
Weights determine impulsive vs strategic behavior. Tune during training
by watching bot behavior and adjusting.
"""

import time
from collections import deque


class MultiTimescaleReward:
    """Tracks honey over multiple windows and combines into per-step reward."""

    # Weight blend — higher long-term coefficients = more strategic behavior
    W_TICK = 1.0
    W_MINUTE = 0.5
    W_HOUR = 0.1
    # Stall penalty: moderate. Strong enough to matter over minutes of being
    # stuck, weak enough that a mildly-productive session (which earns ~+0.01
    # per tick at scale=1e-4) can still net positive if a stall happened
    # earlier. Previous -0.2 was 20x good farming rate and dominated everything.
    W_STALL = -0.03
    W_DEATH = -100.0  # detect via HUD "you died" or screen fade

    # OCR outlier protection: any single-tick honey delta larger than this is
    # treated as a bad reading and IGNORED (both the delta and the "new" honey
    # value). Prevents phantom OCR reads from poisoning the value function.
    # Realistic max for hobby-tier hive: ~200k honey per session, not per tick.
    MAX_HONEY_DELTA_PER_TICK = 100_000

    # If BOTH pollen and honey haven't budged for this many seconds, bot is
    # stuck (in a wall/menu/dead-end). Longer timeout because farming has
    # bursts (fill bag → walk to hive) where honey doesn't change but pollen
    # does, and boss/mob activity may have no immediate honey but visible
    # change. Only fire penalty on TRUE stuck states.
    STALL_TIMEOUT_SEC = 180

    # Potential-based reward shaping (Ng, Harada, Russell 1999).
    # Φ(s) = pollen_fill × PBRS_SCALE
    # F(s, a, s') = γ·Φ(s') - Φ(s)  is added to base reward per step
    #
    # Guarantees: optimal policy is unchanged (bot still ultimately optimizes
    # for honey/hour), but the reward signal is DENSIFIED — every step of
    # bag filling generates a small positive gradient toward flowers, and
    # every step where bag drains without honey ticking up generates a small
    # negative gradient. Fixes the "value function has no data to fit" problem
    # that dominated pre-PBRS training.
    #
    # PBRS_GAMMA MUST equal PPO's gamma (default 0.99 in SB3) for the
    # invariance theorem to hold. If PPO's gamma is ever changed, mirror it
    # here.
    #
    # PBRS uses ABSOLUTE pollen (pollen_current) normalized against a fixed
    # reference max, NOT the pollen_fill ratio. Reason: ratio depends on
    # both numerator AND denominator, so if bag capacity changes (accidental
    # downgrade in a shop), the ratio jumps without the true pollen state
    # changing — invariance theorem breaks and bot gets spuriously rewarded
    # for downgrading its bag. (Seen in real training 2026-08-15: bot
    # downgraded backpack in a shop and got a reward pulse. Fixed here.)
    #
    # Ref max chosen well above the current known Fredrick capacity (405k)
    # so upgrades don't clip. Full-bag Φ scales with actual pollen amount,
    # not display fraction.
    PBRS_SCALE = 1.0
    PBRS_GAMMA = 0.99
    PBRS_POLLEN_REF_MAX = 2_000_000   # Φ = min(1.0, pollen_current / this)

    # Persistence-based baseline recovery for honey drops.
    # Old logic: any delta > MAX_HONEY_DELTA_PER_TICK gets rejected forever,
    # so a legitimate honey drop (buying gear, session-hop) permanently broke
    # the reward function's baseline. (Death does NOT drop honey — only bag
    # pollen — so death isn't relevant to this path; user-corrected 2026-08-14.)
    # New logic: still reject the first N large deltas (protects vs OCR
    # noise), but after OUTLIER_ACCEPT_AFTER_N consecutive same-direction
    # rejects, force-accept and re-baseline.
    #
    # Bumped from 5 to 15 on 2026-08-14 after seeing OCR produce SYSTEMATIC
    # persistent misreads (5-10 consecutive frames all misreading 3.2M as
    # 32M) that caused ping-pong re-baselines and reward corruption.
    # 15 consecutive means ~30 sec at 5 fps of the same wrong reading —
    # that's a strong signal it's real. Legitimate spending events last
    # longer than 30 sec of honey-being-lower, so still handled.
    OUTLIER_ACCEPT_AFTER_N = 15

    # Cooldown between re-baseline events. If we JUST re-baselined and
    # another opposite-direction re-baseline wants to fire, that's the
    # classic OCR-bounce pattern (persistent misread creates one baseline,
    # then real values look like huge deltas creating another). Enforce a
    # 60-second gap so the ping-pong can't happen.
    REBASELINE_COOLDOWN_SEC = 60.0

    # Decimal-shift / comma-drop / partial-parse OCR errors are the most
    # common failure modes for honey OCR:
    #   - 2.87M mis-read as 28.7M (comma looks like period) → 10x ratio
    #   - 3,127,866 mis-read as 3,128 (OCR truncated) → 1000x ratio
    #   - 3.15M mis-read as 315M (period lost entirely) → 100x ratio
    # Any ratio matching one of these decimal-shift patterns is almost
    # certainly an OCR error, not a real event. Reject WITHOUT incrementing
    # the persistence counter (so a streak of these can't force a false
    # re-baseline).
    # Pattern: [1e-4, 1e-3, 1e-2] for shrink-shifts and [100, 1000, 10000]
    # for expand-shifts, with ~20% tolerance around each.
    DECIMAL_SHIFT_RATIOS = [
        (0.00008, 0.00012),   # 10000x truncation
        (0.0008, 0.0012),     # 1000x truncation
        (0.008, 0.012),       # 100x truncation
        (0.08, 0.12),          # 10x truncation
        (8.0, 12.0),           # 10x expansion
        (80.0, 120.0),         # 100x expansion
        (800.0, 1200.0),       # 1000x expansion
    ]

    def __init__(self):
        self.honey_history = deque(maxlen=36000)  # (ts, honey) pairs; ~1 hour at 10 FPS
        self.last_honey = None
        self.last_pollen_fill = None
        self.last_progress_ts = None
        # Previous step's potential. None = "no prior potential this episode"
        # so first step after reset gets F=0 (no phantom shaping spike).
        self.last_potential = None
        self.total_reward_this_episode = 0.0
        # Counters for the persistence-based outlier recovery — one for each
        # direction so we don't confuse "N negative in a row" with mixed.
        self._consecutive_neg_outliers = 0
        self._consecutive_pos_outliers = 0
        # Wall-clock timestamp of the last re-baseline event, used to enforce
        # REBASELINE_COOLDOWN_SEC and prevent OCR-bounce ping-pong.
        self._last_rebaseline_ts = None

    def reset(self):
        self.honey_history.clear()
        self.last_honey = None
        self.last_pollen_fill = None
        self.last_progress_ts = None
        # Don't carry potential across artificial episode boundaries — first
        # step of each new episode gets F=0 to avoid a spurious shaping spike
        # from whatever the last-episode bag state happened to be.
        self.last_potential = None
        self.total_reward_this_episode = 0.0
        self._consecutive_neg_outliers = 0
        self._consecutive_pos_outliers = 0
        # Keep rebaseline cooldown across artificial episode boundaries so
        # ping-pong protection isn't reset every 1024 steps.
        # (self._last_rebaseline_ts intentionally NOT reset here.)

    def _honey_delta_over(self, seconds):
        """Estimated Δhoney over the last `seconds`."""
        if not self.honey_history:
            return 0.0
        now_ts, now_honey = self.honey_history[-1]
        target_ts = now_ts - seconds
        # Find the oldest honey reading within the window
        past_honey = now_honey
        for ts, h in self.honey_history:
            if ts >= target_ts:
                past_honey = h
                break
        return now_honey - past_honey

    def compute(self, hud_state, obs_frame=None):
        """Called each RL step. Returns a scalar reward for this timestep."""
        now = time.time()
        honey = hud_state.get("honey")
        if honey is None:
            # No reading this tick — return small negative to punish 'blind' states
            return -0.01

        # OCR outlier protection with persistence-based baseline recovery.
        #
        # Old logic: any delta > MAX_HONEY_DELTA_PER_TICK gets rejected
        # forever. That works for OCR noise but permanently breaks the
        # baseline on REAL state changes (spending honey, dying, etc.).
        #
        # New logic:
        # 1. Decimal-shift errors (8-12x or 0.08-0.12x jumps) are ALWAYS
        #    rejected — that's the most common honey-OCR failure mode and
        #    real events never look like that.
        # 2. Other large deltas get rejected initially but a same-direction
        #    counter increments. After OUTLIER_ACCEPT_AFTER_N consecutive
        #    rejects in the same direction, we accept the reading as real
        #    and force-update the baseline. Handles legitimate spending /
        #    death events with ~1 sec of confusion, then resumes normal
        #    tracking.
        if self.last_honey is not None and self.last_honey > 0:
            raw_delta = honey - self.last_honey
            ratio = honey / self.last_honey
            is_decimal_shift = any(
                low <= ratio <= high for low, high in self.DECIMAL_SHIFT_RATIOS
            )
            if is_decimal_shift:
                # Almost certainly OCR decimal-shift error — never real.
                # Reject WITHOUT incrementing persistence counters so a
                # streak of decimal-shift errors doesn't force a baseline reset.
                return 0.0
            if abs(raw_delta) > self.MAX_HONEY_DELTA_PER_TICK:
                # Large delta but not a decimal-shift. Might be real
                # (spending / big convert burst / session-hop snapshot) or
                # might be OCR noise. Increment same-direction counter;
                # if we hit N in a row, POSSIBLY accept it as a real state
                # change (unless within cooldown of last re-baseline).
                if raw_delta < 0:
                    self._consecutive_neg_outliers += 1
                    self._consecutive_pos_outliers = 0
                    direction, counter = "NEG", self._consecutive_neg_outliers
                    hint = "likely spending"
                else:
                    self._consecutive_pos_outliers += 1
                    self._consecutive_neg_outliers = 0
                    direction, counter = "POS", self._consecutive_pos_outliers
                    hint = "likely session-hop / snapshot"

                if counter < self.OUTLIER_ACCEPT_AFTER_N:
                    return 0.0

                # Counter hit the threshold — check the ping-pong cooldown
                # before actually re-baselining. If we JUST re-baselined,
                # this is almost certainly the OCR-bounce pattern (5-15
                # persistent misreads created one baseline, real values
                # coming back look like huge opposite-direction deltas).
                if (self._last_rebaseline_ts is not None
                        and now - self._last_rebaseline_ts < self.REBASELINE_COOLDOWN_SEC):
                    # Reject and RESET the counter so it needs another full
                    # N same-direction reads to try again. This anchors us
                    # to the earlier baseline and starves the ping-pong.
                    print(f"[reward] {direction} delta hit threshold but "
                          f"REJECTED — within cooldown of last re-baseline "
                          f"({now - self._last_rebaseline_ts:.1f}s ago). "
                          f"Suspecting OCR bounce, not real state change.")
                    if raw_delta < 0:
                        self._consecutive_neg_outliers = 0
                    else:
                        self._consecutive_pos_outliers = 0
                    return 0.0

                # Cooldown passed — accept as real state change
                print(f"[reward] accepting large {direction} delta after "
                      f"{self.OUTLIER_ACCEPT_AFTER_N} consecutive reads: "
                      f"honey {self.last_honey:,.0f} -> {honey:,.0f} "
                      f"({hint} — re-baselining)")
                if raw_delta < 0:
                    self._consecutive_neg_outliers = 0
                else:
                    self._consecutive_pos_outliers = 0
                self._last_rebaseline_ts = now
                # After accepting a big delta, treat it as a full re-baseline:
                # - Update last_honey to the new value
                # - CLEAR honey_history so rolling minute/hour deltas don't
                #   compute against pre-baseline values (that bug caused
                #   total_reward=3649 in the 2026-08-14 hour-4-ish run —
                #   history had (old_honey_3M, new_honey_31M) and every
                #   subsequent step contributed +23 reward from the
                #   spurious "gained 28M in the last minute" signal)
                # - Return 0 reward for this step (we're not rewarding or
                #   punishing the detected state change itself)
                self.honey_history.clear()
                self.last_honey = honey
                self.honey_history.append((now, honey))
                return 0.0
            else:
                # Normal-sized delta — reset outlier counters (streak broken)
                self._consecutive_neg_outliers = 0
                self._consecutive_pos_outliers = 0

        # Log the honey reading with timestamp
        self.honey_history.append((now, honey))

        # Δhoney over three windows. Clamp all to non-negative — honey CAN
        # go down (spending, session-hop) and we don't want to punish the
        # bot for its own strategic spending decisions. Lost honey shows up
        # as absence-of-reward via opportunity cost, not direct penalty.
        # (This is important for the "almost pure RL" vision — see
        # [[feedback-beebot-pure-rl-vision]] — bot must be free to discover
        # spending strategies without the reward function fighting it.)
        delta_tick = 0.0 if self.last_honey is None else max(0.0, honey - self.last_honey)
        delta_minute = max(0.0, self._honey_delta_over(60))
        delta_hour = max(0.0, self._honey_delta_over(3600))

        # Stall detection — track PROGRESS which is EITHER honey up OR pollen
        # bar fill going up. This way farming (pollen up, no honey change) OR
        # converting (honey up, pollen goes back to 0) both reset the timer.
        # Mob kills / quest turn-ins usually cause honey changes eventually.
        pollen_fill = hud_state.get("pollen_fill")
        pollen_progress = (
            pollen_fill is not None
            and self.last_pollen_fill is not None
            and pollen_fill > self.last_pollen_fill + 0.01
        )
        honey_progress = delta_tick > 0
        made_progress = honey_progress or pollen_progress or self.last_progress_ts is None

        if made_progress:
            self.last_progress_ts = now

        stall_pen = 0.0
        if self.last_progress_ts is not None:
            stalled_for = now - self.last_progress_ts
            if stalled_for > self.STALL_TIMEOUT_SEC:
                stall_pen = self.W_STALL

        self.last_honey = honey
        if pollen_fill is not None:
            self.last_pollen_fill = pollen_fill

        # Potential-based shaping (see PBRS_SCALE docstring above).
        # Uses ABSOLUTE pollen (pollen_current) normalized to a fixed ref
        # max, NOT pollen_fill ratio — avoids spurious reward when bag
        # capacity changes (shop downgrade).
        # Falls back to pollen_fill * capacity_guess if pollen_current
        # isn't available (older HUD reader / degraded OCR).
        pbrs_shaping = 0.0
        pollen_current = hud_state.get("pollen_current")
        if pollen_current is None and pollen_fill is not None:
            # Fallback: infer from fill × known-max. Uses REF_MAX as the
            # assumed capacity — imperfect but doesn't have the downgrade
            # bug because REF_MAX is constant across bag changes.
            clamped_fill = max(0.0, min(1.0, pollen_fill))
            pollen_current = clamped_fill * self.PBRS_POLLEN_REF_MAX
        if pollen_current is not None:
            pollen_current = max(0.0, float(pollen_current))
            normalized = min(1.0, pollen_current / self.PBRS_POLLEN_REF_MAX)
            current_potential = normalized * self.PBRS_SCALE
            if self.last_potential is not None:
                pbrs_shaping = self.PBRS_GAMMA * current_potential - self.last_potential
            self.last_potential = current_potential

        # Reward is blended; scale down absolute honey deltas so per-step
        # reward stays in a range PPO handles well (roughly [-1, +10])
        scale = 1e-4  # 10k honey ≈ +1 reward
        reward = (
            self.W_TICK * delta_tick * scale
            + self.W_MINUTE * delta_minute * scale / 60
            + self.W_HOUR * delta_hour * scale / 3600
            + stall_pen
            + pbrs_shaping
        )

        self.total_reward_this_episode += reward
        return reward


class ShapedItemReward:
    """Adds item-pickup bonuses to the base reward.
    (Phase 3a+ — plugs into MultiTimescaleReward once we have item detection
    from token classification in Phase 3c.)"""

    def __init__(self):
        pass

    def add_pickup(self, item_slug, honey_equivalent):
        """Called when the world-object detector flags a pickup we grabbed.
        Adds a shaped bonus proportional to the item's honey-equivalent value."""
        # TODO Phase 3c: integrate with token_values.json + goal_profiles.json
        # For now, stubbed.
        pass
