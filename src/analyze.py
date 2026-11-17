"""Compute the probes and the paper's main results (Table 2) from cached ArcFace embeddings.

For each model under output/cached_embeds/:
  - Predicting reference similarity: 5-fold CV R² of OLS regression on
    dispersion only, centroid similarity only, and both.
  - Real vs. perturbed separability: 5-fold CV AUC / accuracy of a logistic
    regression on both probes, for all names and for high-/low-fame names
    (split at the median log-pageviews).

Writes output/analysis/summary.json and output/analysis/metrics.csv (per-name values).

Usage:
    uv run src/analyze.py
"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from cv import cv_ols_r2, cv_ols_r2_inductive, cv_separability_inductive
from embeddings import load_embed, load_embed_masked
from probes import calc_centroid_sim, calc_dispersion, calc_gt_sim, centroid

ROOT     = Path(__file__).parent.parent
FAME_CSV = ROOT / "data" / "wp_celeb_fame.csv"


def load_records(model_dir: Path, aligned_root: Path, fame: dict, strict: bool) -> list[dict]:
    """Per-name probe inputs for one model. Names with alignment failures are excluded."""
    records = []
    for person_dir in tqdm(sorted(p for p in model_dir.iterdir() if p.is_dir()), desc=model_dir.name):
        name = person_dir.name.replace("_", " ")
        if name not in fame:
            continue
        aligned = aligned_root / model_dir.name / person_dir.name
        E = load_embed_masked(person_dir / "af.pt", aligned, strict=strict)
        if E is None:
            continue
        E_gt = load_embed(person_dir / "af_gt.pt")
        E_p  = load_embed_masked(person_dir / "af_perturbed.pt", aligned / "perturbed", strict=strict)
        records.append({
            "name":      name,
            "log_views": fame[name],
            "disp":      calc_dispersion(E),
            "cn":        centroid(E),
            "gt_sim":    calc_gt_sim(E, E_gt) if E_gt is not None else None,
            "disp_p":    calc_dispersion(E_p) if E_p is not None else None,
            "cn_p":      centroid(E_p) if E_p is not None else None,
        })
    return records


def analyze_model(records: list[dict], k: int, perm_n: int) -> dict:
    # Predicting reference similarity (names with a reference photo)
    reg = [r for r in records if r["gt_sim"] is not None]
    disp = np.array([r["disp"] for r in reg])
    cn   = np.array([r["cn"] for r in reg])
    gt   = np.array([r["gt_sim"] for r in reg])
    r2 = {
        "dispersion":   cv_ols_r2([disp], gt, k=k, n_perm=perm_n),
        "centroid_sim": cv_ols_r2_inductive([], cn, gt, k=k, n_perm=perm_n),
        "both":         cv_ols_r2_inductive([disp], cn, gt, k=k, n_perm=perm_n),
    }

    # Real vs. perturbed separability (names with perturbed-name generations)
    sep = [r for r in records if r["disp_p"] is not None]
    lv  = np.array([r["log_views"] for r in sep])
    median = float(np.median(lv))
    subsets = {
        "all":  np.arange(len(sep)),
        "high": np.where(lv >  median)[0],
        "low":  np.where(lv <= median)[0],
    }
    disp_r = np.array([r["disp"] for r in sep])
    cn_r   = np.array([r["cn"] for r in sep])
    disp_p = np.array([r["disp_p"] for r in sep])
    cn_p   = np.array([r["cn_p"] for r in sep])
    separability = {}
    for subset, idx in subsets.items():
        res = cv_separability_inductive(disp_r[idx], cn_r[idx], disp_p[idx], cn_p[idx], k=k)
        separability[subset] = {"n": int(len(idx)), **res}

    return {"n": len(records), "n_reg": len(reg), "n_sep": len(sep),
            "median_log_views": median, "r2": r2, "separability": separability}


def main():
    cfg = OmegaConf.load(ROOT / "config" / "config.yaml")
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(sys.argv[1:]))
    analysis_dir = Path(cfg.get("analysis_dir", "output/analysis"))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    k, perm_n = int(cfg.kfold_k), int(cfg.perm_n)

    with open(FAME_CSV, newline="") as f:
        fame = {row["name"]: float(row["log_views"]) for row in csv.DictReader(f)}

    summary, all_records = {}, {}
    for model_dir in sorted(p for p in Path(cfg.embeds_dir).iterdir() if p.is_dir()):
        records = load_records(model_dir, Path(cfg.aligned_dir), fame, bool(cfg.strict_black))
        if len(records) < 2 * k:
            print(f"Skipping {model_dir.name}: too few names ({len(records)})")
            continue
        summary[model_dir.name] = analyze_model(records, k, perm_n)
        all_records[model_dir.name] = records

    with open(analysis_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Per-name values (centroid_sim here is in-sample: max similarity to all other names)
    with open(analysis_dir / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "name", "log_views", "gt_sim", "dispersion", "centroid_sim"])
        for model, records in all_records.items():
            C = np.array([r["cn"] for r in records])
            for i, r in enumerate(records):
                cen = calc_centroid_sim(r["cn"], np.delete(C, i, axis=0))
                w.writerow([model, r["name"], r["log_views"], r["gt_sim"], r["disp"], cen])

    # Table 2
    fmt = lambda m, s: f"{m:.3f}±{s:.3f}"
    print(f"\n{'Model':<22} {'R² disp':>12} {'R² cen':>12} {'R² both':>12}"
          f" {'AUC all':>12} {'Acc all':>12} {'AUC high':>12} {'Acc high':>12}")
    for model, s in summary.items():
        r2, sep = s["r2"], s["separability"]
        cells = [fmt(r2[p]["cv_r2_mean"], r2[p]["cv_r2_std"]) for p in ("dispersion", "centroid_sim", "both")]
        for subset in ("all", "high"):
            cells += [fmt(sep[subset]["cv_auc_mean"], sep[subset]["cv_auc_std"]),
                      fmt(sep[subset]["cv_accuracy_mean"], sep[subset]["cv_accuracy_std"])]
        print(f"{model:<22} " + " ".join(f"{c:>12}" for c in cells))
    print(f"\nSaved {analysis_dir / 'summary.json'} and {analysis_dir / 'metrics.csv'}")


if __name__ == "__main__":
    main()
