"""
Stub bridges — same pattern as WealthClockBridge / ReturnToHiveBridge,
just not yet implemented. When we build these, they follow the same
recipe: recorded paths for navigation + template matching for UI +
pydirectinput.press/click for interactions.
"""

from .base import BridgeScript, BridgeStatus


class BuyEggBridge(BridgeScript):
    """Walk to Basic Egg Shop, buy the highest-tier egg current honey affords.
    Recorded paths needed: to_egg_shop.json + back
    Template needed: egg_shop_ui.png (to confirm we're at the shop)
    Also needs to click through egg-choice buttons based on visible prices."""
    name = "buy_egg"
    priority = 7
    cooldown_seconds = 300              # every 5 min at most — anti-spam

    def can_run(self, hud_state):
        # TODO Phase 2c: read honey number, check against next-egg price
        return False                     # DISABLED until implemented

    def execute(self):
        return BridgeStatus.FAILED


class HatchBeeBridge(BridgeScript):
    """Open hive menu, find empty slot, click Hatch Egg.
    Template needed: hatch_button.png + empty_hive_slot.png"""
    name = "hatch_bee"
    priority = 6
    cooldown_seconds = 60

    def can_run(self, hud_state):
        # TODO: detect empty hive slot + egg in inventory via templates
        return False

    def execute(self):
        return BridgeStatus.FAILED


class QuestCycleBridge(BridgeScript):
    """Rotate through easy bear NPCs (Black, Brown, Panda). For each:
    walk to bear, check for '!' or ready-to-turn-in indicator, accept/turn
    in as appropriate, walk to next.
    Complex — probably needs one bridge per bear or a state machine.
    Recorded paths: to_black_bear.json, to_brown_bear.json, to_panda_bear.json"""
    name = "quest_cycle"
    priority = 5
    cooldown_seconds = 600              # 10 min

    def can_run(self, hud_state):
        return False

    def execute(self):
        return BridgeStatus.FAILED


class MemoryMatchBridge(BridgeScript):
    """Play the daily Memory Match minigame at Ticket Tent.
    Requires 25+ bees to access. Recorded path + template for the game UI +
    active vision logic to match card pairs.
    Higher complexity — active reasoning during the game, not just a
    scripted click sequence."""
    name = "memory_match"
    priority = 8
    cooldown_seconds = 86400            # once per day

    def can_run(self, hud_state):
        # TODO: bee count >= 25 (progression gate check)
        return False

    def execute(self):
        return BridgeStatus.FAILED


class HiveConvertBridge(BridgeScript):
    """Actively wait for hive conversion to complete before resuming farming.
    Distinct from return_to_hive — this fires AFTER the walk, ensures
    conversion actually happens (watches pollen bar drop, or timer).
    May be redundant with return_to_hive's wait — decide during integration."""
    name = "hive_convert"
    priority = 8
    cooldown_seconds = 30

    def can_run(self, hud_state):
        return False

    def execute(self):
        return BridgeStatus.FAILED
