# Intentionally minimal — do NOT re-export anything here.
# Re-exports trigger cascading imports that break `python -m hud.pollen_bar`
# (the __init__ tries to import pollen_bar while it's being loaded as __main__).
# Callers should import directly: `from hud.reader import HudReader` or
# `from hud.pollen_bar import PollenBarReader`.
