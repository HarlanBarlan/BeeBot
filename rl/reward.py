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
    # PBRS_SCALE choice: max Φ = 1.0 (when bag is full). Per-step F magnitudes
    # during typical farming end up around ±0.01–0.05, comparable to the
    # W_TICK × Δhoney × scale terms. Higher SCALE = stronger shaping (bot
    # more aggressive about filling); lower = weaker (falls back to sparse
    # honey signal). Start conservative.
    PBRS_SCALE = 1.0
    PBRS_GAMMA = 0.99

    # Persistence-based baseline recovery for honey drops.
    # Old logic: any delta > MAX_HONEY_DELTA_PER_TICK gets rejected forever,
    # so a legitimate honey drop (buying gear, dying, session-hop) permanently
    # broke the reward function's baseline. New logic: still reject the first
    # few large deltas (protects vs single-frame OCR spikes), but after
    # OUTLIER_ACCEPT_AFTER_N consecutive same-direction rejects, force-accept
    # and re-baseline. Handles real state changes (~1 sec of confusion) while
    # still filtering true OCR noise (which is single-frame or bidirectional).
    OUTLIER_ACCEPT_AFTER_N = 5

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
        # direction so we don't confuse "5 negative in a row" with "3 neg + 2 pos"
        self._consecutive_neg_outliers = 0
        self._consecutive_pos_outliers = 0

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
                # (spending / death / big convert burst) or might be OCR
                # noise. Increment same-direction counter; if we hit N in
                # a row, accept it as a real state change.
                if raw_delta < 0:
                    self._consecutive_neg_outliers += 1
                    self._consecutive_pos_outliers = 0
                    if self._consecutive_neg_outliers >= self.OUTLIER_ACCEPT_AFTER_N:
                        print(f"[reward] accepting large NEG delta after "
                              f"{self.OUTLIER_ACCEPT_AFTER_N} consecutive reads: "
                              f"honey {self.last_honey:,.0f} -> {honey:,.0f} "
                              f"(likely spending or death — re-baselining)")
                        self._consecutive_neg_outliers = 0
                        # Fall through — update baseline, but zero the
                        # reward contribution (bot shouldn't be rewarded
                        # or punished for detected spending events).
                    else:
                        return 0.0
                else:
                    self._consecutive_pos_outliers += 1
                    self._consecutive_neg_outliers = 0
                    if self._consecutive_pos_outliers >= self.OUTLIER_ACCEPT_AFTER_N:
                        print(f"[reward] accepting large POS delta after "
                              f"{self.OUTLIER_ACCEPT_AFTER_N} consecutive reads: "
                              f"honey {self.last_honey:,.0f} -> {honey:,.0f} "
                              f"(likely session-hop / snapshot — re-baselining)")
                        self._consecutive_pos_outliers = 0
                    else:
                        return 0.0
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

        # Potential-based shaping (see PBRS_SCALE / PBRS_GAMMA docstring above).
        # Only compute when we have a fresh pollen reading — if OCR failed
        # this tick, keep last_potential unchanged so the next valid reading
        # doesn't see a phantom drop-to-zero.
        pbrs_shaping = 0.0
        if pollen_fill is not None:
            # Clamp pollen_fill to [0, 1] — OCR occasionally reads garbage
            # like 147% (seen at t=11400 in the 2026-08-14 log). Without
            # clamping this creates a massive spurious PBRS spike.
            clamped_fill = max(0.0, min(1.0, pollen_fill))
            current_potential = clamped_fill * self.PBRS_SCALE
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
