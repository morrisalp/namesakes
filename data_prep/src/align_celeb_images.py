"""
Run facial alignment on collected infobox images.

For each data_prep/faces/{slug}/infobox.jpg, detects and aligns the face using
dlib landmarks and saves the result as data_prep/faces/{slug}/aligned.jpg.
Skips entries where aligned.jpg already exists (safe to resume).

Requires the dlib landmark weights file:
    wget https://github.com/italojs/facial-landmarks-recognition/raw/master/shape_predictor_68_face_landmarks.dat -P weights/

Usage:
    uv run data_prep/src/align_celeb_images.py [--landmark-fn PATH]
"""

import argparse
import csv
import sys
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
from alignment import FaceAligner

INPUT_CSV   = Path(__file__).parent.parent.parent / "data" / "wp_celeb_fame.csv"
FACES_DIR   = Path(__file__).parent.parent / "faces"
LANDMARK_FN = Path(__file__).parent.parent.parent / "weights" / "shape_predictor_68_face_landmarks.dat"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--landmark-fn", type=Path, default=LANDMARK_FN,
                   help="Path to dlib landmark .dat file (default: %(default)s)")
    return p.parse_args()


def main():
    args = parse_args()

    if not args.landmark_fn.exists():
        print(f"Landmark file not found: {args.landmark_fn}", file=sys.stderr)
        print("Download it with:", file=sys.stderr)
        print("  wget https://github.com/italojs/facial-landmarks-recognition/raw/master/"
              "shape_predictor_68_face_landmarks.dat -P weights/", file=sys.stderr)
        sys.exit(1)

    if not INPUT_CSV.exists():
        print(f"Missing {INPUT_CSV} — run data_prep/src/sample_celebs.py first.", file=sys.stderr)
        sys.exit(1)

    with INPUT_CSV.open(newline="", encoding="utf-8") as f:
        names = [row["name"] for row in csv.DictReader(f)]

    print(f"Aligning faces for {len(names)} people...")
    print(f"Loading landmark model from {args.landmark_fn}")
    aligner = FaceAligner(str(args.landmark_fn))

    skipped = failed = saved = 0
    for name in tqdm(names, unit="person"):
        slug      = name.replace(" ", "_")
        img_path  = FACES_DIR / slug / "infobox.jpg"
        out_path  = FACES_DIR / slug / "aligned.jpg"

        if out_path.exists():
            skipped += 1
            continue

        if not img_path.exists():
            failed += 1
            continue

        img = Image.open(img_path).convert("RGB")
        aligned = aligner.align(img)

        # FaceAligner returns a black image when detection fails — skip those
        if aligned.getbbox() is None:
            failed += 1
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        aligned.save(out_path, "JPEG", quality=90)
        saved += 1

    print(f"Done. Saved {saved}, skipped {skipped} already done, {failed} failed/no image.")


if __name__ == "__main__":
    main()
