"""
Filter collected celebrity data to freely-licensed images and push to HF Hub.

Reads data/wp_celeb_fame.csv and data_prep/faces/{slug}/{infobox.jpg,aligned.jpg,license.json},
keeps only freely-licensed entries (CC*, Public domain, GFDL; drops non-free /
missing licenses), and skips names containing parentheses, commas, or " and "
(disambiguation suffixes, titles, or multi-person entries). Pushes to
malper/namesakes on Hugging Face Datasets.

Requires being logged in:  uv run hf auth login
"""

import csv
import json
import re
import sys
from pathlib import Path

from datasets import Dataset, Features, Image, Value

HF_REPO        = "malper/namesakes"
INPUT_CSV      = Path(__file__).parent.parent.parent / "data" / "wp_celeb_fame.csv"
FACES_DIR      = Path(__file__).parent.parent / "faces"
PERTURBED_CSV = Path(__file__).parent.parent.parent / "data" / "perturbed_names.csv"


def is_free(license_str: str) -> bool:
    s = license_str.strip().lower()
    if not s:
        return False
    return "non-free" not in s and "fair use" not in s


def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def main():
    if not INPUT_CSV.exists():
        print(f"Missing {INPUT_CSV} — run src/sample_celebs.py first.", file=sys.stderr)
        sys.exit(1)

    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    perturbed_names: dict[str, str] = {}
    if PERTURBED_CSV.exists():
        with PERTURBED_CSV.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                perturbed_names[row["name"]] = row["perturbed"]
    else:
        print(f"Warning: {PERTURBED_CSV} not found — doppleganger column will be empty.",
              file=sys.stderr)

    records = []
    skipped = {
        "parentheses_or_comma": 0,
        "no_image": 0,
        "no_aligned": 0,
        "no_license": 0,
        "non_free": 0,
    }

    for row in rows:
        name = row["name"]

        if "(" in name or ")" in name or "," in name:
            skipped["parentheses_or_comma"] += 1
            continue
        if " and " in name.lower():
            skipped["parentheses_or_comma"] += 1
            continue

        slug         = name.replace(" ", "_")
        img_path     = FACES_DIR / slug / "infobox.jpg"
        aligned_path = FACES_DIR / slug / "aligned.jpg"
        license_path = FACES_DIR / slug / "license.json"

        if not img_path.exists():
            skipped["no_image"] += 1
            continue
        if not aligned_path.exists():
            skipped["no_aligned"] += 1
            continue
        if not license_path.exists():
            skipped["no_license"] += 1
            continue

        with license_path.open(encoding="utf-8") as f:
            lic = json.load(f)

        if not is_free(lic.get("license", "")):
            skipped["non_free"] += 1
            continue

        records.append({
            "name":          name,
            "doppleganger":  perturbed_names.get(name, ""),  # published column name (= perturbed name)
            "views":         int(row["views"]),
            "log_views":     float(row["log_views"]),
            "image":         str(img_path),
            "aligned_image": str(aligned_path),
            "license":       lic.get("license", ""),
            "license_url":   lic.get("license_url", ""),
            "artist":        strip_html(lic.get("artist", "")),
            "image_name":    lic.get("image_name", ""),
        })

    total = len(rows)
    print(f"{len(records)}/{total} entries pass all filters")
    print(f"  Skipped: {skipped['parentheses_or_comma']} ambiguous names, "
          f"{skipped['no_image']} no image, "
          f"{skipped['no_aligned']} no aligned image, "
          f"{skipped['no_license']} no license, "
          f"{skipped['non_free']} non-free")

    if not records:
        print("Nothing to push.", file=sys.stderr)
        sys.exit(1)

    features = Features({
        "name":          Value("string"),
        "doppleganger":  Value("string"),
        "views":         Value("int32"),
        "log_views":     Value("float32"),
        "image":         Image(),
        "aligned_image": Image(),
        "license":       Value("string"),
        "license_url":   Value("string"),
        "artist":        Value("string"),
        "image_name":    Value("string"),
    })

    ds = Dataset.from_list(records, features=features)
    print(f"Pushing {len(ds)} rows to {HF_REPO} ...")
    ds.push_to_hub(HF_REPO)
    print("Done.")


if __name__ == "__main__":
    main()
