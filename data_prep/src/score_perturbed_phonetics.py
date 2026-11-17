"""Score perturbed name pairs for phonetic similarity.

Reads data/perturbed_names.csv and writes data/perturbed_phonetics.csv
with additional columns for phonetic encodings and similarity scores for each
name part pair.

Phonetic encodings (per part): soundex, metaphone, nysiis, match_rating_codex
Similarity scores (per pair of parts):
  - phonetic_match  : bool from match_rating_comparison (None = incomparable length)
  - jaro_winkler    : float [0,1] on raw strings (character-level, prefix-weighted)
  - soundex_match   : bool (codes identical)
  - metaphone_match : bool (codes identical)
  - nysiis_match    : bool (codes identical)

All per-part scores are joined with '|' when there are multiple name parts.
A summary `all_parts_phonetic_match` column flags pairs where ANY part's
phonetic codes match (likely too similar for a good probe).

Usage:
    uv run data_prep/src/score_perturbed_phonetics.py
    uv run data_prep/src/score_perturbed_phonetics.py --input data/perturbed_names.csv
"""

import argparse
import csv
from pathlib import Path

import jellyfish

INPUT_CSV  = Path(__file__).parent.parent.parent / "data" / "perturbed_names.csv"
OUTPUT_CSV = Path(__file__).parent.parent.parent / "data" / "perturbed_phonetics.csv"


def _alpha(s: str) -> str:
    """Strip non-alphabetical characters (hyphens, apostrophes, etc.)."""
    return "".join(c for c in s if c.isalpha())


def score_parts(orig_part: str, pert_part: str) -> dict:
    o, d = _alpha(orig_part.lower()), _alpha(pert_part.lower())
    if not o or not d:
        # Degenerate part (e.g. a bare hyphen); return nulls
        return {k: None for k in (
            "soundex_orig", "soundex_pert", "metaphone_orig", "metaphone_pert",
            "nysiis_orig", "nysiis_pert", "mrc_orig", "mrc_pert",
            "soundex_match", "metaphone_match", "nysiis_match",
            "match_rating_comparison", "jaro_winkler",
        )}

    sx_o, sx_d   = jellyfish.soundex(o),            jellyfish.soundex(d)
    mp_o, mp_d   = jellyfish.metaphone(o),           jellyfish.metaphone(d)
    ny_o, ny_d   = jellyfish.nysiis(o),              jellyfish.nysiis(d)
    mrc_o, mrc_d = jellyfish.match_rating_codex(o),  jellyfish.match_rating_codex(d)
    mr_cmp       = jellyfish.match_rating_comparison(o, d)  # True/False/None
    jw           = jellyfish.jaro_winkler_similarity(o, d)

    return {
        "soundex_orig":        sx_o,
        "soundex_pert":        sx_d,
        "metaphone_orig":      mp_o,
        "metaphone_pert":      mp_d,
        "nysiis_orig":         ny_o,
        "nysiis_pert":         ny_d,
        "mrc_orig":            mrc_o,
        "mrc_pert":            mrc_d,
        "soundex_match":       sx_o == sx_d,
        "metaphone_match":     mp_o == mp_d,
        "nysiis_match":        ny_o == ny_d,
        "match_rating_comparison": mr_cmp,  # None means "incomparable"
        "jaro_winkler":        round(jw, 4),
    }


def join(vals) -> str:
    return "|".join(str(v) for v in vals)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",  default=str(INPUT_CSV),  help="Input CSV (default: %(default)s)")
    p.add_argument("--output", default=str(OUTPUT_CSV), help="Output CSV (default: %(default)s)")
    args = p.parse_args()

    rows_in = list(csv.DictReader(Path(args.input).open(encoding="utf-8")))
    print(f"Loaded {len(rows_in):,} pairs from {args.input}")

    fieldnames = [
        "name", "perturbed",
        # per-part encodings (joined with | for multi-part names)
        "soundex_orig", "soundex_pert",
        "metaphone_orig", "metaphone_pert",
        "nysiis_orig", "nysiis_pert",
        "mrc_orig", "mrc_pert",
        # per-part similarity scores
        "soundex_match", "metaphone_match", "nysiis_match",
        "match_rating_comparison", "jaro_winkler",
        # all-parts summary flags (one per metric)
        "all_parts_soundex_match",
        "all_parts_metaphone_match",
        "all_parts_nysiis_match",
    ]

    rows_out = []
    for row in rows_in:
        orig_parts = row["name"].split()
        pert_parts = row["perturbed"].split()

        # Pair up parts positionally; if lengths differ, score only the paired prefix
        n = min(len(orig_parts), len(pert_parts))
        scores = [score_parts(orig_parts[i], pert_parts[i]) for i in range(n)]

        # match_rating_comparison is excluded from the summary flag: it returns True
        # for ~99.9% of pairs (it is designed for loose name matching, not a filter).
        # all_phonetic_match = True only when every paired part matches on all three metrics.
        valid = [s for s in scores if s["soundex_match"] is not None]
        def all_parts(metric):
            return bool(valid) and all(s[metric] for s in valid)
        any_match = {
            "all_parts_soundex_match":   all_parts("soundex_match"),
            "all_parts_metaphone_match": all_parts("metaphone_match"),
            "all_parts_nysiis_match":    all_parts("nysiis_match"),
        }

        rows_out.append({
            "name":        row["name"],
            "perturbed": row["perturbed"],
            "soundex_orig":   join(s["soundex_orig"]   for s in scores),
            "soundex_pert":   join(s["soundex_pert"]   for s in scores),
            "metaphone_orig": join(s["metaphone_orig"]  for s in scores),
            "metaphone_pert": join(s["metaphone_pert"]  for s in scores),
            "nysiis_orig":    join(s["nysiis_orig"]     for s in scores),
            "nysiis_pert":    join(s["nysiis_pert"]     for s in scores),
            "mrc_orig":       join(s["mrc_orig"]        for s in scores),
            "mrc_pert":       join(s["mrc_pert"]        for s in scores),
            "soundex_match":  join(s["soundex_match"]   for s in scores),
            "metaphone_match":join(s["metaphone_match"] for s in scores),
            "nysiis_match":   join(s["nysiis_match"]    for s in scores),
            "match_rating_comparison": join(s["match_rating_comparison"] for s in scores),
            "jaro_winkler":   join(s["jaro_winkler"]   for s in scores),
            **any_match,
        })

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    import json, random as _random
    rng = _random.Random(42)
    N_EXAMPLES = 4

    n = len(rows_out)

    def any_true(col):
        return sum(any(v == "True" for v in r[col].split("|")) for r in rows_out)

    all_parts_metrics = [
        "all_parts_soundex_match",
        "all_parts_metaphone_match",
        "all_parts_nysiis_match",
    ]

    summary_counts = {
        "soundex_match (any part)":           any_true("soundex_match"),
        "metaphone_match (any part)":         any_true("metaphone_match"),
        "nysiis_match (any part)":            any_true("nysiis_match"),
        "match_rating_comparison (any part)": any_true("match_rating_comparison"),
        "all_parts_soundex_match":            sum(r["all_parts_soundex_match"]   for r in rows_out),
        "all_parts_metaphone_match":          sum(r["all_parts_metaphone_match"] for r in rows_out),
        "all_parts_nysiis_match":             sum(r["all_parts_nysiis_match"]    for r in rows_out),
    }

    examples = {
        col: rng.sample([r for r in rows_out if r[col] is True],
                        min(N_EXAMPLES, summary_counts[col]))
        for col in all_parts_metrics
    }

    print(f"\nWrote {n:,} rows to {args.output}")
    print(f"\n{'Metric':<42} {'n':>5}   pct")
    print("-" * 57)
    for label, count in summary_counts.items():
        note = "  ← excluded from all_parts_* (too permissive)" if "match_rating" in label else ""
        print(f"  {label:<40} {count:5d} / {n}  ({100*count/n:.1f}%){note}")
        if label in all_parts_metrics:
            for r in examples[label]:
                print(f"      {r['name']}  →  {r['perturbed']}")

    log = {
        "n_pairs": n,
        "input":   args.input,
        "output":  args.output,
        "match_rates": {
            m: {
                "n":   c,
                "pct": round(100 * c / n, 1),
                **({"examples": [{"name": r["name"], "perturbed": r["perturbed"]}
                                 for r in examples[m]]}
                   if m in all_parts_metrics else {}),
            }
            for m, c in summary_counts.items()
        },
    }
    log_path = Path(args.output).with_suffix(".log.json")
    log_path.write_text(json.dumps(log, indent=2))
    print(f"\nLog → {log_path}")


if __name__ == "__main__":
    main()
