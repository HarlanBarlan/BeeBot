"""
Item recognizer — three-tier lookup pipeline.

STUB — real implementation lives in Phase 2c. Structure and interfaces
are defined here so the routine planner and HUD-reader modules can
already integrate against them.

Tier 1 — Template matching against sprite database.
  Fast. Zero false positives if template quality is good. Fails on:
  - Items whose sprite has never been captured
  - Items with visual variants (color-shifted, animated)

Tier 2 — OCR on tooltip text.
  Triggered when the cursor hovers an item. Reads item name from the
  tooltip, looks up in ItemDB by name. Handles the "sprite unknown but
  I can read the label" case.

Tier 3 — VLM fallback.
  Triggered when both templates and OCR fail. Sends the tooltip region
  (name + description) to a vision-language model, gets a structured
  description of the item's effect, appends result to item_db.json via
  ItemDB.append_from_recognition().

The bot ALWAYS checks the DB after any tier — recognized items must
resolve to a DB entry (either pre-existing or freshly added) so the
routine planner has structured info to act on.
"""

from .item_db import ItemDB


class ItemRecognizer:
    def __init__(self, item_db=None):
        self.db = item_db or ItemDB()

    # --- Tier 1: template matching ------------------------------------------
    def match_template(self, image_crop):
        """Given a cropped region of an item icon, return matching slug or None.
        Uses cv2.matchTemplate against sprites/ folder.
        NOT YET IMPLEMENTED — Phase 2c."""
        raise NotImplementedError("Template matching pending — Phase 2c")

    # --- Tier 2: OCR --------------------------------------------------------
    def read_tooltip(self, tooltip_crop):
        """Given a cropped tooltip image (visible when cursor hovers item),
        run OCR to extract the item name and short description text.
        Returns (name_string, description_string) or (None, None) on failure.
        NOT YET IMPLEMENTED — Phase 2c (Tesseract or small transformer)."""
        raise NotImplementedError("OCR pending — Phase 2c")

    # --- Tier 3: VLM --------------------------------------------------------
    def vlm_identify(self, tooltip_crop):
        """Send the tooltip region to a vision-language model and get back
        a structured item description matching item_db.json schema.
        NOT YET IMPLEMENTED — Phase 2c or 3.

        Options:
          - Local: LLaVA / MiniCPM-V (compute-heavy, private)
          - API: Claude/Gemini vision (fast, costs money per call)

        Returns a dict conforming to item_db schema, or None on failure.
        Caller passes to ItemDB.append_from_recognition() to persist.
        """
        raise NotImplementedError("VLM pending — Phase 2c or 3")

    # --- Full pipeline ------------------------------------------------------
    def identify(self, image_crop, tooltip_crop=None):
        """Full three-tier pipeline. Returns item dict or None.
        `image_crop` = just the item sprite (small, ~40×40).
        `tooltip_crop` = optional larger crop including tooltip name+desc
          if the cursor was hovering.

        Order: template → OCR → VLM. Stop at first success.
        Newly-VLM-identified items get appended to the DB automatically.
        NOT YET IMPLEMENTED — stubs above indicate the shape.
        """
        raise NotImplementedError("Full pipeline pending — Phase 2c")
