# Data Collection

This document describes how to rebuild the celebrity fame dataset used by the Namesakes project.
Three scripts are run in sequence; each produces a file consumed by the next.

---

## Overview

| Step | Script | Output |
|------|--------|--------|
| 1 | `data_prep/src/get_infobox_pages.py` | `data/infobox_pages.tsv` (infobox pages with view counts) |
| 2 | `data_prep/src/sample_celebs.py` | `data/wp_celeb_fame.csv` (uniformly subsampled, ~2 000 entries) |
| 3 | `data_prep/src/get_celeb_images.py` | `data_prep/faces/{Name}/infobox.jpg` + `license.json` |
| 4 | `data_prep/src/align_celeb_images.py` | `data_prep/faces/{Name}/aligned.jpg` |
| 5 | `data_prep/src/get_perturbed_names.py` | `data/perturbed_names.csv` |
| 5a | `data_prep/src/score_perturbed_phonetics.py` | `data/perturbed_phonetics.csv` (optional, not used downstream) |
| 6 | `data_prep/src/push_to_hub.py` | `malper/namesakes` on Hugging Face Hub |

---

## Required downloads

All files go in `data_prep/dumps/`:

```shell
mkdir -p data_prep/dumps
cd data_prep/dumps

# SQL dumps (for resolving Template:Infobox_person → page titles)
wget https://dumps.wikimedia.org/enwiki/20260101/enwiki-20260101-linktarget.sql.gz
wget https://dumps.wikimedia.org/enwiki/20260101/enwiki-20260101-templatelinks.sql.gz
wget https://dumps.wikimedia.org/enwiki/20260101/enwiki-20260101-page.sql.gz

# Pageview file (~700 MB compressed)
wget https://dumps.wikimedia.org/other/pageview_complete/2026/2026-01/pageviews-20260101-user.bz2
```

---

## Step 1 — Build infobox page list with view counts

Finds all namespace-0 pages that transclude `Template:Infobox_person` via the
SQL dumps, looks up each page's January 1 2026 view count from the pageview
file, and writes a filtered TSV.

```shell
uv run data_prep/src/get_infobox_pages.py
```

Optional flags:
```
--min-views INT   Exclude pages with fewer views than this (default: 10)
```

**Output**: `data/infobox_pages.tsv` — tab-separated `title\tviews`, sorted descending by views.
**Runtime**: ~10–15 min processing after download.

Verify:
```shell
wc -l data/infobox_pages.tsv   # expect several hundred thousand
head -5 data/infobox_pages.tsv # top pages by views
```

---

## Step 2 — Build celebrity fame table

Reads `data/infobox_pages.tsv`, filters by a minimum view count, and writes
the full filtered table with raw and log view counts.  Also saves histograms
of the view distribution before and after filtering to `data_prep/plots/`.

```shell
uv run data_prep/src/sample_celebs.py
```

Optional flags:
```
--min-views INT   Exclude pages below this view count (default: 10)
--n-bins INT      Number of log-views bins for uniform subsampling (default: 10)
--target-n INT    Approximate total sample size; each bin gets target-n / n-bins entries (default: 2000)
--seed INT        Random seed for subsampling (default: 42)
```

**Output**:
- `data/wp_celeb_fame.csv` — columns `name, views, log_views`, sorted descending by views
- `data_prep/plots/fame_distribution.png` — 3×2 plot: raw views (row 1), log-views fine-grained (row 2),
  and log-views at the n-bins resolution used for subsampling (row 3); before/after in each column

**Runtime**: seconds (purely local, no API calls).

Verify:
```shell
wc -l data/wp_celeb_fame.csv   # header + one row per page passing the filter
head -5 data/wp_celeb_fame.csv # top pages by views
```

---

## Step 3 — Download infobox images

For each celebrity in `data/wp_celeb_fame.csv`, fetches the Wikipedia infobox
image, resizes it so the longest side is ≤ 256 px, and saves it as JPEG.

```shell
uv run data_prep/src/get_celeb_images.py
```

**Output**: `data_prep/faces/{Name_Slug}/infobox.jpg` and `data_prep/faces/{Name_Slug}/license.json`
**Runtime**: ~3 s/person (sequential with rate-limit sleep).

Verify:
```shell
ls data_prep/faces/ | wc -l   # expect ~N subdirectories
```

Missing images are expected for pages whose Wikipedia article has no infobox photo.

---

## Step 4 — Align faces

Runs dlib facial landmark detection and alignment on each `infobox.jpg`,
saving a 256×256 aligned crop as `aligned.jpg` in the same directory.
Safe to resume — skips entries where `aligned.jpg` already exists.

Requires the dlib landmark weights (one-time download):
```shell
wget https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat -P weights/
```

Then:
```shell
uv run data_prep/src/align_celeb_images.py
```

Optional flags:
```
--landmark-fn PATH   Path to landmark .dat file (default: weights/shape_predictor_68_face_landmarks.dat)
```

**Output**: `data_prep/faces/{Name_Slug}/aligned.jpg`

---

## Step 5a — Perturbed phonetic analysis (optional, not used downstream)

Scores each perturbed pair for phonetic similarity using three metrics
(Soundex, Metaphone, NYSIIS via `jellyfish`). Useful for auditing how many
pairs are too phonetically close to serve as a useful probe — e.g. *Robin
Williams → Robina Walliams* would likely activate the same model attractor.

```shell
uv run data_prep/src/score_perturbed_phonetics.py
```

**Output**:
- `data/perturbed_phonetics.csv` — original CSV plus per-part phonetic
  encodings, per-part match flags, and three `all_parts_{metric}_match`
  summary columns (True when every name part matches on that metric)
- `data_prep/perturbed_phonetics.log.json` — match rates + random examples per metric

Typical rates (current dataset): ~30% of pairs match on Soundex across all
parts, ~19% on Metaphone, ~15% on NYSIIS (strictest). These pairs are
candidates for regeneration with a phonetic-distance constraint added to
`get_perturbed_names.py`, but this is not yet implemented.

---

## Step 5 — Push to Hugging Face Hub

Filters to freely-licensed entries (CC*, Public domain, GFDL; excludes non-free /
missing licenses), strips HTML from artist fields, and pushes to
[malper/namesakes](https://huggingface.co/datasets/malper/namesakes).

Log in first if needed:
```shell
uv run hf auth login
```

Then:
```shell
uv run data_prep/src/push_to_hub.py
```

Entries are excluded if:
- Name contains parentheses (disambiguation e.g. "John Smith (actor)"), commas (titles), or " and " (multi-person entries)
- `infobox.jpg`, `aligned.jpg`, or `license.json` is missing
- License is non-free or blank

**Output**: HF dataset with columns `name, views, log_views, image, aligned_image, license, license_url, artist, image_name`
