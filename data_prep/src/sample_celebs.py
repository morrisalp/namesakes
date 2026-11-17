"""
Build data/wp_celeb_fame.csv from data/infobox_pages.tsv.

Reads every infobox page and its view count, filters by --min-views, then
subsamples each log-views bin down to target_n // n_bins so the total is
approximately --target-n entries with a uniform log-views distribution.

Saves a 3×2 diagnostic plot to data_prep/plots/fame_distribution.png showing
raw views and log-views (fine bins + the n_bins used for subsampling).

Output: data/wp_celeb_fame.csv  (columns: name, views, log_views)
"""

import argparse
import csv
import math
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt

MIN_VIEWS = 10
N_BINS    = 10
TARGET_N  = 2000
SEED      = 42

INFOBOX_TSV = Path(__file__).parent.parent.parent / "data" / "infobox_pages.tsv"
OUTPUT      = Path(__file__).parent.parent.parent / "data" / "wp_celeb_fame.csv"
PLOT_PATH   = Path(__file__).parent.parent / "plots" / "fame_distribution.png"


def load_infobox_pages() -> list[tuple[str, int]]:
    if not INFOBOX_TSV.exists():
        print(f"Missing: {INFOBOX_TSV}", file=sys.stderr)
        print("Run src/get_infobox_pages.py first.", file=sys.stderr)
        sys.exit(1)
    entries: list[tuple[str, int]] = []
    for line in INFOBOX_TSV.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        title, views_str = parts
        try:
            entries.append((title, int(views_str)))
        except ValueError:
            continue
    return entries


def uniform_subsample(
    entries: list[tuple[str, int]], n_bins: int, target_n: int, seed: int
) -> list[tuple[str, int]]:
    """Downsample each log-views bin so total ≈ target_n with uniform distribution.

    Binary-searches for the largest per-bin cap T where sum(min(b, T)) <= target_n,
    so sparse high-views bins don't leave the total short of the target.
    """
    log_v = [math.log(v) for _, v in entries]
    lo, hi = min(log_v), max(log_v)

    def bin_idx(lv: float) -> int:
        idx = int((lv - lo) / (hi - lo) * n_bins)
        return max(0, min(n_bins - 1, idx))

    bins: list[list[tuple[str, int]]] = [[] for _ in range(n_bins)]
    for entry, lv in zip(entries, log_v):
        bins[bin_idx(lv)].append(entry)

    sizes = [len(b) for b in bins]
    # Binary search for largest T with sum(min(b, T)) <= target_n
    lo_t, hi_t = 0, max(sizes)
    while lo_t < hi_t:
        mid = (lo_t + hi_t + 1) // 2
        if sum(min(s, mid) for s in sizes) <= target_n:
            lo_t = mid
        else:
            hi_t = mid - 1
    target_per_bin = lo_t + 1
    actual_total = sum(min(s, target_per_bin) for s in sizes)
    print(f"  Bin sizes: {sizes}")
    print(f"  Target per bin: {target_per_bin}  (total = {actual_total:,})")

    rng = random.Random(seed)
    result: list[tuple[str, int]] = []
    for b in bins:
        rng.shuffle(b)
        result.extend(b[:target_per_bin])
    return result


def save_plot(
    before: list[tuple[str, int]], after: list[tuple[str, int]], n_bins: int
) -> None:
    bv  = [v           for _, v in before]
    av  = [v           for _, v in after]
    blv = [math.log(v) for v in bv]
    alv = [math.log(v) for v in av]

    lo, hi = min(blv), max(blv)
    bin_edges = [lo + (hi - lo) * i / n_bins for i in range(n_bins + 1)]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Fame distribution — before vs after uniform subsampling")

    # Row 0: raw views (log-scale y)
    axes[0, 0].hist(bv, bins=100, log=True)
    axes[0, 0].set_title(f"Views — before  (n={len(bv):,})")
    axes[0, 0].set_xlabel("Views")
    axes[0, 0].set_ylabel("Count (log scale)")

    axes[0, 1].hist(av, bins=100, log=True)
    axes[0, 1].set_title(f"Views — after  (n={len(av):,})")
    axes[0, 1].set_xlabel("Views")
    axes[0, 1].set_ylabel("Count (log scale)")

    # Row 1: log(views) — fine bins
    axes[1, 0].hist(blv, bins=60)
    axes[1, 0].set_title(f"log(views) — before  (n={len(blv):,})")
    axes[1, 0].set_xlabel("log(views)")
    axes[1, 0].set_ylabel("Count")

    axes[1, 1].hist(alv, bins=60)
    axes[1, 1].set_title(f"log(views) — after  (n={len(alv):,})")
    axes[1, 1].set_xlabel("log(views)")
    axes[1, 1].set_ylabel("Count")

    # Row 2: log(views) — n_bins (actual subsampling bins)
    axes[2, 0].hist(blv, bins=bin_edges)
    axes[2, 0].set_title(f"log(views) — before, {n_bins} bins  (n={len(blv):,})")
    axes[2, 0].set_xlabel("log(views)")
    axes[2, 0].set_ylabel("Count")

    axes[2, 1].hist(alv, bins=bin_edges)
    axes[2, 1].set_title(f"log(views) — after, {n_bins} bins  (n={len(alv):,})")
    axes[2, 1].set_xlabel("log(views)")
    axes[2, 1].set_ylabel("Count")

    plt.tight_layout()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved plot to {PLOT_PATH}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-views", type=int, default=MIN_VIEWS,
                   help="Exclude pages with fewer views than this (default: %(default)s)")
    p.add_argument("--n-bins", type=int, default=N_BINS,
                   help="Number of log-views bins for uniform subsampling (default: %(default)s)")
    p.add_argument("--target-n", type=int, default=TARGET_N,
                   help="Approximate total sample size (default: %(default)s)")
    p.add_argument("--seed", type=int, default=SEED,
                   help="Random seed for subsampling (default: %(default)s)")
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading infobox pages...")
    all_entries = load_infobox_pages()
    print(f"  {len(all_entries):,} pages total")

    filtered = [(t, v) for t, v in all_entries if v >= args.min_views]
    print(f"  {len(filtered):,} pages with >= {args.min_views} views")

    print(f"Subsampling to uniform log-views ({args.n_bins} bins, target={args.target_n:,}, seed={args.seed})...")
    subsampled = uniform_subsample(filtered, args.n_bins, args.target_n, args.seed)
    print(f"  {len(subsampled):,} pages after subsampling")

    save_plot(filtered, subsampled, args.n_bins)

    subsampled.sort(key=lambda x: x[1], reverse=True)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "views", "log_views"])
        writer.writeheader()
        for title, views in subsampled:
            writer.writerow({
                "name": title.replace("_", " "),
                "views": views,
                "log_views": math.log(views),
            })

    print(f"Wrote {len(subsampled):,} rows to {OUTPUT}")


if __name__ == "__main__":
    main()
