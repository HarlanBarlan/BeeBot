"""
Item knowledge base for BeeBot.

Loads item_db.json and offers structured queries:
  - by name / slug
  - by category (consumable/treat/jelly/tool/world_interaction)
  - by trigger condition (used by the routine planner)
  - by usage method (used by the interaction executor)

Also holds the "manual_only" flag — items the bot must NEVER auto-use
because they're strategic decisions (Star Treats, Star Jelly, Bitterberry,
Wind Shrine offerings).

Future: the recognizer module (item_recognizer.py) can APPEND newly-
identified items back to this DB (via VLM lookup + wiki fallback).
"""

import json
from pathlib import Path

DB_PATH = Path(__file__).parent / "item_db.json"
MANUAL_ONLY_TRIGGER = "strategic_experimentation"
# Historically this flagged items the bot must NOT auto-use (irreversible
# consequences that only a human should decide). Per updated project scope
# (2026-08-12): user is running the bot as a FULL autonomy experiment and
# accepts irreversible errors as learning signal. Flag is retained for
# priority-weighting purposes (these items still deserve caution — the bot
# should be MORE deliberate about them, not banned from them).


class ItemDB:
    def __init__(self, path=DB_PATH):
        self.path = Path(path)
        with open(self.path) as f:
            raw = json.load(f)
        self.schema_version = raw.get("_schema_version", 1)
        self.items = raw["items"]

    def get(self, slug):
        """Look up an item by its slug key. Returns dict or None."""
        return self.items.get(slug)

    def find_by_name(self, name):
        """Case-insensitive fuzzy lookup by display name."""
        name_lower = name.strip().lower()
        for slug, item in self.items.items():
            if item["name"].lower() == name_lower:
                return slug, item
        return None, None

    def by_category(self, category):
        """All items in a given category (e.g. 'consumable', 'treat')."""
        return {slug: item for slug, item in self.items.items()
                if item.get("category") == category}

    def by_trigger(self, trigger):
        """All items whose trigger_conditions include the given trigger."""
        return {slug: item for slug, item in self.items.items()
                if trigger in item.get("trigger_conditions", [])}

    def is_strategic(self, slug):
        """True if this item's use has irreversible consequences and deserves
        MORE deliberation than routine items. NOT a hard filter — the bot
        may still use these; it should just be more conservative about them
        (higher confidence threshold, more evidence required)."""
        item = self.get(slug)
        if not item:
            return False
        return MANUAL_ONLY_TRIGGER in item.get("trigger_conditions", [])

    # Legacy alias for older callers — same semantics as is_strategic now
    is_manual_only = is_strategic

    def top_priorities_for(self, trigger, limit=5):
        """Items matching the trigger, sorted by descending priority.
        Strategic-experimentation items are INCLUDED (bot has full autonomy)
        but their `strategic_experimentation` marker tells the planner to
        require more evidence before firing them."""
        candidates = [(slug, item) for slug, item in self.by_trigger(trigger).items()]
        candidates.sort(key=lambda kv: kv[1].get("priority", 0), reverse=True)
        return candidates[:limit]

    def append_from_recognition(self, slug, item_dict):
        """Add a newly-recognized item (from OCR+VLM fallback) to the DB
        and persist. slug should be a stable kebab_case key."""
        if slug in self.items:
            return False
        self.items[slug] = item_dict
        self._save()
        return True

    def _save(self):
        with open(self.path, "w") as f:
            payload = {
                "_schema_version": self.schema_version,
                "_schema_doc": "See item_db.py for schema definition.",
                "items": self.items,
            }
            json.dump(payload, f, indent=2)


if __name__ == "__main__":
    db = ItemDB()
    print(f"Loaded {len(db.items)} items.")
    print("\nBoost-session items by priority:")
    for slug, item in db.top_priorities_for("boost_session_open"):
        print(f"  [{item['priority']}] {item['name']:22s} -> {item['effect']}")
    print("\nManual-only (strategic) items:")
    for slug, item in db.items.items():
        if db.is_manual_only(slug):
            print(f"  {item['name']}")
