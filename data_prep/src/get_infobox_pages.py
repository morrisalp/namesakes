"""
Build data/infobox_pages.tsv from Wikipedia SQL dumps and a pageview file.

For each namespace-0 page that transcludeS Template:Infobox_person, looks up
its January 1 2026 view count and writes title→views to a TSV, filtered to
pages with at least MIN_VIEWS views.

Download the dumps first — see data_prep/README.md for instructions.
Expected locations:
  data_prep/dumps/enwiki-20260101-linktarget.sql.gz
  data_prep/dumps/enwiki-20260101-templatelinks.sql.gz
  data_prep/dumps/enwiki-20260101-page.sql.gz
  data_prep/dumps/pageviews-20260101-user.bz2
"""

import argparse
import bz2
import gzip
import io
import re
import sys
from pathlib import Path

from tqdm.auto import tqdm

MIN_VIEWS = 10

DUMP_DIR      = Path(__file__).parent.parent / "dumps"
LINKTARGET    = DUMP_DIR / "enwiki-20260101-linktarget.sql.gz"
TEMPLATELINKS = DUMP_DIR / "enwiki-20260101-templatelinks.sql.gz"
PAGE          = DUMP_DIR / "enwiki-20260101-page.sql.gz"
PAGEVIEWS     = DUMP_DIR / "pageviews-20260101-user.bz2"
OUTPUT        = Path(__file__).parent.parent.parent / "data" / "infobox_pages.tsv"

RE_LINKTARGET = re.compile(r"\((\d+),10,'Infobox_person'\)")
RE_PAGE       = re.compile(r"\((\d+),0,'((?:[^'\\]|\\.)*)'")


def iter_gz_lines(path: Path, desc: str):
    size = path.stat().st_size
    with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
        with open(path, "rb") as raw:
            class _Tracked(io.RawIOBase):
                def readable(self): return True
                def readinto(self, b):
                    n = raw.readinto(b)
                    if n: pbar.update(n)
                    return n
            with gzip.open(io.BufferedReader(_Tracked()), "rt", encoding="utf-8", errors="replace") as f:
                yield from f


def iter_bz2_lines(path: Path, desc: str):
    size = path.stat().st_size
    with tqdm(total=size, unit="B", unit_scale=True, unit_divisor=1024, desc=desc) as pbar:
        with open(path, "rb") as raw:
            class _Tracked(io.RawIOBase):
                def readable(self): return True
                def readinto(self, b):
                    n = raw.readinto(b)
                    if n: pbar.update(n)
                    return n
            with bz2.open(io.BufferedReader(_Tracked()), "rt", encoding="utf-8", errors="replace") as f:
                yield from f


def find_infobox_target_id() -> int:
    for line in iter_gz_lines(LINKTARGET, "linktarget"):
        if "Infobox_person" not in line:
            continue
        for m in RE_LINKTARGET.finditer(line):
            target_id = int(m.group(1))
            print(f"  lt_id for 'Infobox_person' = {target_id}")
            return target_id
    raise RuntimeError("'Infobox_person' not found in linktarget dump")


def collect_transclusion_ids(target_id: int) -> set[int]:
    suffix = f",{target_id}"
    ids: set[int] = set()
    for line in iter_gz_lines(TEMPLATELINKS, "templatelinks"):
        if suffix not in line:
            continue
        pos = 0
        while True:
            idx = line.find(suffix, pos)
            if idx == -1:
                break
            open_paren = line.rfind("(", 0, idx)
            if open_paren != -1:
                m = re.match(r"(\d+),", line[open_paren + 1:])
                if m:
                    ids.add(int(m.group(1)))
            pos = idx + len(suffix) + 1
    print(f"  {len(ids):,} pages transclude Template:Infobox_person")
    return ids


def resolve_titles(ids: set[int]) -> set[str]:
    titles: set[str] = set()
    for line in iter_gz_lines(PAGE, "page"):
        if ",0,'" not in line:
            continue
        for m in RE_PAGE.finditer(line):
            if int(m.group(1)) in ids:
                title = m.group(2).replace("\\\\", "\\").replace("\\'", "'")
                titles.add(title)
    print(f"  {len(titles):,} titles resolved")
    return titles


def collect_views(titles: set[str]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for line in iter_bz2_lines(PAGEVIEWS, "pageviews"):
        # format: domain title page_id access_type views hourly
        parts = line.split(" ")
        if len(parts) < 5 or parts[0] != "en.wikipedia":
            continue
        if parts[1] not in titles:
            continue
        try:
            totals[parts[1]] = totals.get(parts[1], 0) + int(parts[4])
        except ValueError:
            continue
    print(f"  {len(totals):,} infobox titles with views")
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-views", type=int, default=MIN_VIEWS,
                   help="Exclude pages with fewer views than this (default: %(default)s)")
    args = p.parse_args()

    missing = [f for f in (LINKTARGET, TEMPLATELINKS, PAGE, PAGEVIEWS) if not f.exists()]
    if missing:
        for f in missing:
            print(f"Missing: {f}", file=sys.stderr)
        print("See DATA_COLLECTION.md for download instructions.", file=sys.stderr)
        sys.exit(1)

    print("Step 1: finding lt_id for 'Infobox_person'...")
    target_id = find_infobox_target_id()

    print("\nStep 2: collecting page IDs from templatelinks...")
    ids = collect_transclusion_ids(target_id)

    print("\nStep 3: resolving titles from page dump...")
    titles = resolve_titles(ids)

    print("\nStep 4: looking up view counts...")
    views = collect_views(titles)

    filtered = {t: v for t, v in views.items() if v >= args.min_views}
    print(f"  {len(filtered):,} pages with >= {args.min_views} views")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as f:
        for title, v in sorted(filtered.items(), key=lambda x: x[1], reverse=True):
            f.write(f"{title}\t{v}\n")

    print(f"\nDone. Wrote {len(filtered):,} entries to {OUTPUT}")


if __name__ == "__main__":
    main()
