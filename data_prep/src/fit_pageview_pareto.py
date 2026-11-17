"""Fit a Pareto (power law) distribution to Wikipedia pageview counts.

Reads data/infobox_pages.tsv (output of Step 1 in data_prep/README.md),
fits a discrete power law and log-normal to the tail using the Clauset et al.
(2009) method, prints a comparison, and saves a CCDF diagnostic plot to
data_prep/plots/pageview_pareto.png.

Usage:
    uv run data_prep/src/fit_pageview_pareto.py
    uv run data_prep/src/fit_pageview_pareto.py --min-views 10
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import powerlaw

MIN_VIEWS = 10

INFOBOX_TSV = Path(__file__).parent.parent.parent / "data" / "infobox_pages.tsv"
PLOT_PATH   = Path(__file__).parent.parent / "plots" / "pageview_pareto.png"
JSON_PATH   = Path(__file__).parent.parent / "plots" / "pageview_pareto.json"


def load_views(min_views: int) -> np.ndarray:
    if not INFOBOX_TSV.exists():
        print(f"Missing: {INFOBOX_TSV}", file=sys.stderr)
        print("Run src/get_infobox_pages.py first.", file=sys.stderr)
        sys.exit(1)
    views = []
    for line in INFOBOX_TSV.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        try:
            v = int(parts[1])
        except ValueError:
            continue
        if v >= min_views:
            views.append(v)
    return np.array(views, dtype=float)


def save_plot(fit: powerlaw.Fit) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    fit.plot_ccdf(ax=ax, color="steelblue", linewidth=1.5, label="Empirical CCDF")
    fit.power_law.plot_ccdf(ax=ax, color="crimson",    linestyle="--", label=f"Power law  α={fit.alpha:.2f}")
    fit.lognormal.plot_ccdf(ax=ax, color="darkorange", linestyle=":",  label="Log-normal")

    ax.set_xlabel("Pageviews")
    ax.set_ylabel("P(X ≥ x)")
    ax.set_title(
        f"Wikipedia infobox pageview distribution\n"
        f"(tail fit: x_min={int(fit.xmin):,}, n_tail={int(fit.n):,})"
    )
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    print(f"Saved plot → {PLOT_PATH}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--min-views", type=int, default=MIN_VIEWS,
                   help="Pre-filter: exclude pages with fewer views (default: %(default)s)")
    args = p.parse_args()

    print("Loading views...")
    views = load_views(args.min_views)
    print(f"  {len(views):,} pages with >= {args.min_views} views")
    print(f"  min={int(views.min()):,}  max={int(views.max()):,}  median={int(np.median(views)):,}")

    print("\nFitting power law (Clauset et al. MLE)...")
    fit = powerlaw.Fit(views, discrete=True)

    print(f"\n  Power law  α = {fit.alpha:.4f}  (σ = {fit.sigma:.4f})")
    print(f"  x_min      = {int(fit.xmin):,}")
    print(f"  Tail size  = {int(fit.n):,} / {len(views):,} pages")

    print("\nComparing power law vs log-normal (log-likelihood ratio):")
    R_ln, p_ln = fit.distribution_compare("power_law", "lognormal")
    R_ex, p_ex = fit.distribution_compare("power_law", "exponential")
    R_st, p_st = fit.distribution_compare("power_law", "stretched_exponential")

    for name, R, p_val in [
        ("log-normal",           R_ln, p_ln),
        ("exponential",          R_ex, p_ex),
        ("stretched exponential",R_st, p_st),
    ]:
        direction = "power law" if R > 0 else name
        print(f"  vs {name:<24s}  LLR={R:+.2f}  p={p_val:.3f}  → favors {direction}")

    print("\nLog-normal fit parameters (over full data x >= x_min):")
    print(f"  μ  = {fit.lognormal.mu:.4f}")
    print(f"  σ  = {fit.lognormal.sigma:.4f}")

    save_plot(fit)

    stats = {
        "n_total": len(views),
        "min_views_filter": args.min_views,
        "data": {
            "min": int(views.min()),
            "max": int(views.max()),
            "median": int(np.median(views)),
            "mean": float(views.mean()),
        },
        "power_law": {
            "alpha": fit.alpha,
            "alpha_sigma": fit.sigma,
            "xmin": int(fit.xmin),
            "n_tail": int(fit.n),
        },
        "lognormal": {
            "mu": fit.lognormal.mu,
            "sigma": fit.lognormal.sigma,
        },
        "comparisons": {
            "power_law_vs_lognormal":            {"LLR": R_ln, "p": p_ln},
            "power_law_vs_exponential":          {"LLR": R_ex, "p": p_ex},
            "power_law_vs_stretched_exponential":{"LLR": R_st, "p": p_st},
        },
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(stats, indent=2))
    print(f"Saved stats → {JSON_PATH}")


if __name__ == "__main__":
    main()
