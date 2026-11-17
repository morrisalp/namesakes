"""
Find a perturbed name for each collected celebrity.

For each person in data_prep/faces/, splits their name into parts and for each
part finds name parts in data/infobox_pages.tsv that:
  - start with the same letter
  - have minimum edit distance >= 1 (not an exact match)
  - have length >= max(3, len(query_part) - 1)  (filters initials / short codes)

When multiple candidates exist for a part, one is chosen randomly (fixed seed).
The perturbed name is constructed by joining the chosen part per position.

Usage:
    uv run data_prep/src/get_perturbed_names.py
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

from tqdm.auto import tqdm

FACES_DIR   = Path(__file__).parent.parent / "faces"
INFOBOX_TSV = Path(__file__).parent.parent.parent / "data" / "infobox_pages.tsv"
OUTPUT_CSV  = Path(__file__).parent.parent.parent / "data" / "perturbed_names.csv"
SEED = 42


def levenshtein(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = prev if a[i-1] == b[j-1] else 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def build_index(tsv_path: Path) -> dict[str, set[str]]:
    """Return letter_to_parts: first letter (upper) -> set of name parts."""
    letter_to_parts: dict[str, set[str]] = defaultdict(set)
    for line in tsv_path.read_text(encoding="utf-8").splitlines():
        cols = line.split("\t")
        if not cols or not cols[0]:
            continue
        title = cols[0].replace("_", " ")
        for part in title.split():
            if "," not in part:
                letter_to_parts[part[0]].add(part)
    return letter_to_parts


def close_parts(query_part: str, letter_to_parts: dict[str, set[str]]) -> set[str]:
    """
    Return all same-first-letter parts at the minimum edit distance >= 1.
    Candidates shorter than max(3, len(query_part) - 1) are excluded to
    filter out initials and short codes (e.g. 'A', 'AB', 'A4').
    Comparison is case-insensitive.
    """
    bucket = letter_to_parts.get(query_part[0], set())
    q = query_part.lower()
    min_len = max(3, len(query_part) - 1)

    min_dist: int | None = None
    result: set[str] = set()

    for candidate in bucket:
        if len(candidate) < min_len:
            continue
        if candidate.lower() == q:
            continue
        d = levenshtein(q, candidate.lower())
        if min_dist is None or d < min_dist:
            min_dist = d
            result = {candidate}
        elif d == min_dist:
            result.add(candidate)

    return result


def main():
    rng = random.Random(SEED)

    print(f"Loading index from {INFOBOX_TSV} ...")
    letter_to_parts = build_index(INFOBOX_TSV)
    print(f"  {sum(len(v) for v in letter_to_parts.values()):,} unique name parts indexed\n")

    face_slugs = sorted(p.name for p in FACES_DIR.iterdir() if p.is_dir())
    names = [slug.replace("_", " ") for slug in face_slugs]

    rows = []
    for name in tqdm(names, unit="person"):
        perturbed_parts = []
        for part in name.split():
            candidates = close_parts(part, letter_to_parts)
            if candidates:
                chosen = rng.choice(sorted(candidates))  # sort before choice for reproducibility
            else:
                chosen = part  # no match found, keep original
            perturbed_parts.append(chosen)
        rows.append((name, " ".join(perturbed_parts)))

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "perturbed"])
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
