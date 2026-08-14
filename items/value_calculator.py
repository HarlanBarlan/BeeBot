"""
Runtime reward shaping for the RL system.

Combines three inputs to price any collectible in the moment:
  1. Base honey-equivalent from token_values.json (the anchor)
  2. Multiplier from the currently-active goal (goal_profiles.json)
  3. Satiation: each additional copy is worth less; at cap, worth 0

Used by the Phase 3a RL reward function:
    calc = ValueCalculator()
    calc.set_goal("boost_session_active")
    reward = delta_honey + sum(
        calc.value_of(item_type, inventory_count=inv.get(item_type, 0),
                      cap=CAPS.get(item_type))
        for item_type in items_picked_up_this_tick
    )

The scripted top-level planner is responsible for picking the current
goal based on HUD state, inventory, time of day, quest state, etc.
Values here are BASELINE ESTIMATES — tune during Phase 3a based on
observed bot behavior.
"""

import json
import math
from pathlib import Path

DEFAULT_TOKENS_PATH = Path(__file__).parent / "token_values.json"
DEFAULT_GOALS_PATH = Path(__file__).parent / "goal_profiles.json"


class ValueCalculator:
    def __init__(self, tokens_path=DEFAULT_TOKENS_PATH,
                 goals_path=DEFAULT_GOALS_PATH,
                 default_goal="default_farming"):
        with open(tokens_path) as f:
            self.tokens = json.load(f)
        with open(goals_path) as f:
            self.goals = json.load(f)["goals"]

        # Flatten world_pickups + ability_tokens + bee_treats + eggs into one map
        # so lookups don't need to know the category.
        self.base_values = {}
        for section in ("world_pickups", "ability_tokens", "bee_treats", "eggs"):
            for slug, entry in self.tokens.get(section, {}).items():
                if slug.startswith("_"):
                    continue
                self.base_values[slug] = entry.get("value", 0)

        self.current_goal = default_goal
        if default_goal not in self.goals:
            raise ValueError(f"Unknown default goal: {default_goal}")

    # --- goal management ----------------------------------------------------
    def set_goal(self, goal_name):
        if goal_name not in self.goals:
            raise ValueError(f"Unknown goal: {goal_name}. "
                             f"Known: {list(self.goals.keys())}")
        self.current_goal = goal_name

    def known_goals(self):
        return list(self.goals.keys())

    # --- value computation --------------------------------------------------
    def _satiation_factor(self, count, cap):
        """Diminishing returns per additional copy.
        - At cap (if provided): 0
        - First copy (count=0): 1.0
        - Each subsequent: 1/sqrt(count+1)
        Rough curve: 1, 0.71, 0.58, 0.5, 0.45, ..."""
        if cap is not None and count >= cap:
            return 0.0
        return 1.0 / math.sqrt(count + 1)

    def value_of(self, item_slug, inventory_count=0, cap=None, goal_override=None):
        """Return the current shaped honey-equivalent value of picking up
        one of this item, given inventory count and (optional) cap.

        goal_override lets a caller ask 'what would this be worth if we
        switched to goal X?' — handy for goal-selection heuristics.
        """
        base = self.base_values.get(item_slug, 0.0)
        if base <= 0:
            return 0.0
        goal = goal_override or self.current_goal
        goal_mults = self.goals[goal]["value_modifiers"]
        goal_multiplier = goal_mults.get(item_slug, 1.0)
        satiation = self._satiation_factor(inventory_count, cap)
        return base * goal_multiplier * satiation

    def best_goal_for(self, item_slug):
        """Which goal makes this item most valuable? Helpful for debugging
        or for reward-driven goal switching."""
        base = self.base_values.get(item_slug, 0.0)
        if base <= 0:
            return self.current_goal, 0.0
        best = None
        best_val = -1
        for name, spec in self.goals.items():
            m = spec["value_modifiers"].get(item_slug, 1.0)
            v = base * m
            if v > best_val:
                best = name
                best_val = v
        return best, best_val


if __name__ == "__main__":
    calc = ValueCalculator()
    print(f"Loaded {len(calc.base_values)} item base values, "
          f"{len(calc.known_goals())} goals.")
    print()
    tests = [
        ("star_jelly", 0),
        ("honey_token", 0),
        ("neonberry", 0),
        ("neonberry", 3),  # already have 3 — satiation kicks in
        ("cloud_vial_pickup", 9),  # near cap (10)
        ("rage", 0),
    ]
    for goal in ["default_farming", "expanding_hive", "boost_session_active",
                 "crafting_super_smoothie", "boss_prep_and_fight"]:
        calc.set_goal(goal)
        print(f"[goal={goal}]")
        for slug, inv in tests:
            cap = 10 if slug == "cloud_vial_pickup" else None
            v = calc.value_of(slug, inventory_count=inv, cap=cap)
            print(f"  {slug:20s} (own {inv}): {v:>10,.0f}")
        print()
