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

    def __init__(self):
        self.honey_history = deque(maxlen=36000)  # (ts, honey) pairs; ~1 hour at 10 FPS
        self.last_honey = None
        self.last_pollen_fill = None
        self.last_progress_ts = None
        self.total_reward_this_episode = 0.0

    def reset(self):
        self.honey_history.clear()
        self.last_honey = None
        self.last_pollen_fill = None
        self.last_progress_ts = None
        self.total_reward_this_episode = 0.0

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

        # OCR outlier protection: reject readings with impossibly-large jumps.
        # (Legitimate honey gains are capped at hive-conversion rate — a
        # SINGLE tick jumping by hundreds of thousands is always OCR error.)
        if self.last_honey is not None:
            raw_delta = honey - self.last_honey
            if abs(raw_delta) > self.MAX_HONEY_DELTA_PER_TICK:
                # Keep old honey value, don't log this reading — return small
                # neutral reward so we don't move on bad data
                return 0.0

        # Log the honey reading with timestamp
        self.honey_history.append((now, honey))

        # Δhoney over three windows
        delta_tick = 0.0 if self.last_honey is None else max(0.0, honey - self.last_honey)
        delta_minute = self._honey_delta_over(60)
        delta_hour = self._honey_delta_over(3600)

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

        # Reward is blended; scale down absolute honey deltas so per-step
        # reward stays in a range PPO handles well (roughly [-1, +10])
        scale = 1e-4  # 10k honey ≈ +1 reward
        reward = (
            self.W_TICK * delta_tick * scale
            + self.W_MINUTE * delta_minute * scale / 60
            + self.W_HOUR * delta_hour * scale / 3600
            + stall_pen
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
