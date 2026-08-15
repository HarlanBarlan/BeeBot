#!/usr/bin/env python
"""
Fetch buff icons from the Bee Swarm Simulator wiki (Buffs & Debuffs page)
and save them as hud/probes/buff_<name>.png templates.

Uses Fandom's MediaWiki API to list images on the page + get their CDN URLs,
then downloads each icon. The regular wiki HTML endpoint 403s but the API
endpoint responds fine with a browser UA.

Usage:
  python scripts/fetch_wiki_buff_icons.py [--dry-run] [--force]

  --dry-run: list images that WOULD be downloaded, without downloading
  --force:   redownload icons even if the target file already exists

After running, review templates in hud/probes/ and delete any that don't
match in-game appearance. Some wiki icons are outdated or have different
rendering than the live game — those won't template-match well.
"""

import argparse
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("This script needs 'requests'. Install with: pip install requests")
    sys.exit(1)


WIKI_API = "https://bee-swarm-simulator.fandom.com/api.php"
WIKI_PAGE = "Buffs & Debuffs"
OUTPUT_DIR = Path(__file__).parent.parent / "hud" / "probes"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/121.0.0.0 Safari/537.36"
)

# Skip images that are obviously not buff icons — screenshots, wiki chrome, etc.
SKIP_PATTERNS = [
    r"^RobloxScreenShot",           # Roblox in-game screenshots
    r"Site-logo",
    r"Wordmark",
    r"Wiki-vector",
    r"^Ac[0-9]+screenshot",         # more screenshots
    r"^Screen_?[Ss]hot",
]

# Some Fandom images are variants of the same buff (e.g., "Focus_Buff",
# "Focus_Buff_2"). Prefer the plain-name version and skip the variants.


def slugify(name):
    """Filename-safe lowercase-underscore slug from a name."""
    name = name.strip().lower()
    name = re.sub(r"\.(png|jpg|jpeg|webp|gif)$", "", name)
    name = re.sub(r"_?buff$|_?debuff$|_?icon$", "", name)  # normalize suffixes
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def list_page_images():
    """Return list of image filenames on the target wiki page."""
    resp = requests.get(WIKI_API, params={
        "action": "parse",
        "page": WIKI_PAGE,
        "format": "json",
        "prop": "images",
    }, headers={"User-Agent": UA}, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"API error: {data['error']}")
    return data["parse"]["images"]


def get_image_urls(filenames):
    """Batch-lookup CDN URLs for a list of image filenames.
    Returns dict {filename: info} — key is the ORIGINAL filename as passed
    in (underscore form), even though MediaWiki normalizes titles to use
    spaces in the response."""
    result = {}
    # API supports up to 50 titles per request
    for i in range(0, len(filenames), 50):
        batch = filenames[i:i + 50]
        titles = "|".join(f"File:{f}" for f in batch)
        resp = requests.get(WIKI_API, params={
            "action": "query",
            "titles": titles,
            "prop": "imageinfo",
            "iiprop": "url|size",
            "format": "json",
        }, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        query = data.get("query", {})
        # MediaWiki normalizes titles (usually "_" -> " "). Build a reverse
        # map so we can key the result by the ORIGINAL requested filename.
        normalized_to_original = {}
        for norm in query.get("normalized", []):
            # {"from": "File:Haste_Buff.png", "to": "File:Haste Buff.png"}
            frm = re.sub(r"^File:", "", norm["from"])
            to = re.sub(r"^File:", "", norm["to"])
            normalized_to_original[to] = frm
        for page in query.get("pages", {}).values():
            title = page.get("title", "")
            fname_returned = re.sub(r"^File:", "", title)
            # Map back to original requested form if the title was normalized
            fname_original = normalized_to_original.get(fname_returned, fname_returned)
            infos = page.get("imageinfo", [])
            if infos:
                result[fname_original] = {
                    "url": infos[0]["url"],
                    "width": infos[0].get("width"),
                    "height": infos[0].get("height"),
                }
        time.sleep(0.3)   # be nice to the API
    return result


def should_skip(fname, info):
    """Filter out non-buff images."""
    for pat in SKIP_PATTERNS:
        if re.search(pat, fname, re.IGNORECASE):
            return True, "matches skip pattern"
    w = info.get("width", 0)
    h = info.get("height", 0)
    # Skip extremely large images — probably banners or comparison charts
    if w > 1024 or h > 1024:
        return True, f"too big ({w}x{h})"
    # Skip tiny (probably corner decorations)
    if w > 0 and h > 0 and (w < 16 or h < 16):
        return True, f"too small ({w}x{h})"
    # Skip non-square-ish images — buff icons are always ~square
    if w > 0 and h > 0:
        ratio = max(w, h) / min(w, h)
        if ratio > 2.5:
            return True, f"non-square ({w}x{h})"
    return False, None


def download_icon(fname, url, output_dir, force=False):
    """Download and save as buff_<slug>.png. Returns (path, status)."""
    slug = slugify(fname)
    if not slug:
        return None, "empty slug"
    target = output_dir / f"buff_{slug}.png"
    if target.exists() and not force:
        return target, "exists (skip)"
    try:
        resp = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        return None, f"download failed: {e}"
    target.write_bytes(resp.content)
    return target, "downloaded"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Parse the page and print icons, but don't download")
    p.add_argument("--force", action="store_true",
                   help="Redownload icons even if the target file exists")
    args = p.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print(f"Listing images on '{WIKI_PAGE}' via MediaWiki API...")
    try:
        filenames = list_page_images()
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)
    print(f"Found {len(filenames)} image(s) on the page.")

    print(f"Fetching CDN URLs...")
    urls = get_image_urls(filenames)
    print(f"Resolved {len(urls)} URL(s).")

    # Filter
    kept = {}
    skipped = 0
    for fname in filenames:
        info = urls.get(fname)
        if not info:
            skipped += 1
            continue
        skip, reason = should_skip(fname, info)
        if skip:
            print(f"  SKIP  {fname:45s} ({reason})")
            skipped += 1
            continue
        kept[fname] = info

    print(f"\n{len(kept)} candidate buff icons after filtering "
          f"({skipped} skipped):")
    for fname in list(kept)[:10]:
        info = kept[fname]
        print(f"  {fname:35s} {info['width']}x{info['height']}  "
              f"-> buff_{slugify(fname)}.png")
    if len(kept) > 10:
        print(f"  ... and {len(kept) - 10} more")

    if args.dry_run:
        print("\n(dry-run — no downloads)")
        return

    print(f"\nDownloading to {OUTPUT_DIR}/...")
    counts = {"downloaded": 0, "exists (skip)": 0, "error": 0}
    for fname, info in kept.items():
        target, status = download_icon(fname, info["url"], OUTPUT_DIR,
                                         force=args.force)
        if target:
            print(f"  {target.name:45s} {status}")
            counts[status] = counts.get(status, 0) + 1
        else:
            print(f"  {slugify(fname):45s} ERROR: {status}")
            counts["error"] += 1
        time.sleep(0.15)

    print(f"\nDone. Downloaded {counts.get('downloaded', 0)}, "
          f"skipped {counts.get('exists (skip)', 0)}, "
          f"errors {counts.get('error', 0)}.")
    print(f"\nReview {OUTPUT_DIR}/ and delete templates that don't match "
          f"in-game rendering. Icons on the wiki are static; some in-game "
          f"buffs animate or have effects that make template match fail.")


if __name__ == "__main__":
    main()
