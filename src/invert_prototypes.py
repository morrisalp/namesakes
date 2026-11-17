"""
Generate a large batch of images for specific names, align, invert with E4E,
and average in W+ space to produce a prototype face.

Useful for visually inspecting what a model associates with a name.

Usage:
    uv run src/invert_prototypes.py \\
        --names "Emma Watson" "Kirsten Dunst" \\
        [--n-images 16|64]     (default: 16)
        [--model sdxl]         (default: sdxl)
        [--comfyui-url URL]    (default: http://127.0.0.1:8188)
        [--out-dir DIR]        (default: output/prototypes)
        [--batch-size N]       (default: 4)
        [--no-invert]          skip E4E inversion (just generate + align + grid)

Outputs per name under out_dir/model/name_slug/:
    images/         aligned generated images (0.png, 1.png, ...)
    grid.png        GT (top-left) + all aligned gens in a grid
    prototype.png   W+-averaged E4E prototype (omitted with --no-invert)
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from tqdm.auto import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from alignment import FaceAligner
from comfyui_api import (build_flux_workflow, build_sdxl_workflow,
                         generate_images)

mpl.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "savefig.bbox": "tight"})

ALIGNED_DIR = ROOT / "output" / "aligned"
PROMPT_TEMPLATE = "a picture of the person {}"

# Default names to run when --names is not specified.
# All low-fame names likely to be hallucinated by the model.
DEFAULT_NAMES = [
    "Donna Air",
    "Jeffrey Beall",
    "Annie Lowrey",
    "Leon Neyfakh",
    "Remo Ruffini",
    "Alan Yentob",
    "Abbi Jacobson",
    "Adam Ragusea",
]
BATCH_SIZE_DEFAULT = 4
N_COLS_GRID = 8   # columns in the generated-images grid


# ---------------------------------------------------------------------------
# Model config helpers
# ---------------------------------------------------------------------------

def load_model_cfg(model_key: str) -> dict:
    path = ROOT / "config" / "model" / f"{model_key}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown model '{model_key}'. "
                         f"Available: {[p.stem for p in (ROOT/'config'/'model').glob('*.yaml')]}")
    with open(path) as f:
        return yaml.safe_load(f)


def build_workflow(cfg: dict, prompt: str, seed: int, batch_size: int):
    t = cfg["type"]
    if t == "flux":
        return build_flux_workflow(
            unet=cfg["diffusion_model_path"],
            clip_l=cfg["clip_l_path"],
            t5xxl=cfg["t5xxl_path"],
            vae=cfg["vae_path"],
            prompt=prompt,
            width=cfg["width"], height=cfg["height"],
            batch_size=batch_size,
            steps=cfg["num_inference_steps"],
            guidance=cfg["guidance_scale"],
            seed=seed,
        )
    else:  # sdxl / sdxl-turbo
        return build_sdxl_workflow(
            checkpoint=cfg["checkpoint"],
            prompt=prompt,
            negative_prompt=cfg.get("negative_prompt", ""),
            width=cfg["width"], height=cfg["height"],
            batch_size=batch_size,
            steps=cfg["num_inference_steps"],
            cfg=cfg["guidance_scale"],
            seed=seed,
        )


# ---------------------------------------------------------------------------
# Grid helper
# ---------------------------------------------------------------------------

def save_grid(out_path: Path, gt: Image.Image | None,
              images: list[Image.Image], prototype: Image.Image | None,
              name: str):
    """GT (top-left) | aligned gens | prototype (bottom-left if present)."""
    n_cols = N_COLS_GRID
    # slots: images + optional prototype
    all_imgs = images + ([prototype] if prototype is not None else [])
    n_rows = math.ceil(len(all_imgs) / n_cols)

    has_gt = gt is not None
    n_cols_total = n_cols + (1 if has_gt else 0)

    fig, axes = plt.subplots(n_rows, n_cols_total,
                             figsize=(n_cols_total * 0.9, n_rows * 1.0))
    axes = np.array(axes).reshape(n_rows, n_cols_total)

    for ax in axes.flat:
        ax.set_axis_off()

    # GT column: image in row 0, label below; rows 1+ hidden
    if has_gt:
        axes[0, 0].imshow(np.array(gt))
        axes[0, 0].set_title("GT", fontsize=6)
        if prototype is not None and n_rows > 1:
            axes[-1, 0].imshow(np.array(prototype))
            axes[-1, 0].set_title("proto", fontsize=6)

    col_off = 1 if has_gt else 0
    for i, img in enumerate(images):
        r, c = divmod(i, n_cols)
        axes[r, c + col_off].imshow(np.array(img))

    fig.suptitle(name, fontsize=8, y=1.01)
    fig.savefig(out_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--n-images", type=int, default=16)
    parser.add_argument("--model", default="sdxl")
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8188")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE_DEFAULT)
    parser.add_argument("--no-invert", action="store_true",
                        help="Skip E4E inversion; just generate, align, and grid")
    parser.add_argument("--device", default="cuda",
                        help="Torch device for E4E inversion (default: cuda). Use cuda:1 if GPU 0 is occupied by ComfyUI.")
    args = parser.parse_args()

    names = args.names if args.names is not None else DEFAULT_NAMES

    model_cfg = load_model_cfg(args.model)
    model_slug = args.model

    out_root = Path(args.out_dir) if args.out_dir else ROOT / "output" / "prototypes"
    out_root = out_root / model_slug

    do_invert = not args.no_invert
    if do_invert:
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available; skipping E4E inversion.")
            do_invert = False
        else:
            e4e_path = ROOT / "weights" / "e4e_ffhq_encode.pt"
            if not e4e_path.exists():
                print(f"WARNING: E4E weights not found at {e4e_path}; skipping inversion.")
                do_invert = False

    if do_invert:
        sys.path.append(str(ROOT / "encoder4editing"))
        from models.psp import pSp
        from utils.common import tensor2im
        from invert import Inverter
        inverter = Inverter(str(e4e_path), args.device, pSp, tensor2im)

    aligner = FaceAligner(str(ROOT / "weights" / "shape_predictor_68_face_landmarks.dat"))

    n_batches = math.ceil(args.n_images / args.batch_size)

    for name in names:
        slug = name.replace(" ", "_")
        out_dir = out_root / slug
        img_dir = out_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        # ---- generate ----
        existing = list(img_dir.glob("*.png"))
        if len(existing) < args.n_images:
            print(f"Generating {args.n_images} images for '{name}'...")
            prompt = PROMPT_TEMPLATE.format(name)
            raw_imgs = []
            for i in tqdm(range(n_batches), desc=name):
                wf = build_workflow(model_cfg, prompt,
                                    seed=i * args.batch_size, batch_size=args.batch_size)
                raw_imgs += generate_images(args.comfyui_url, wf)
            raw_imgs = raw_imgs[:args.n_images]
        else:
            print(f"'{name}': {len(existing)} images already exist, skipping generation.")
            raw_imgs = [Image.open(p).convert("RGB") for p in sorted(existing)]

        # ---- align ----
        aligned = []
        for img in tqdm(raw_imgs, desc=f"  aligning {name}"):
            try:
                aligned.append(aligner.align(img))
            except Exception:
                pass  # skip failed alignments

        if not aligned:
            print(f"  WARNING: all alignments failed for '{name}', skipping.")
            continue

        for i, img in enumerate(aligned):
            img.save(img_dir / f"{i}.png")
        print(f"  {len(aligned)} aligned images saved.")

        # ---- load GT ----
        gt_path = ALIGNED_DIR / "sd_xl_base_1_0" / slug / "gt.png"
        gt = Image.open(gt_path).convert("RGB") if gt_path.exists() else None

        # ---- filter black images before inversion ----
        n_before = len(aligned)
        aligned = [img for img in aligned if np.array(img).mean() >= 10]
        if len(aligned) < n_before:
            print(f"  Skipped {n_before - len(aligned)} black image(s).")
        if not aligned:
            print(f"  WARNING: no valid images remain for '{name}', skipping.")
            continue

        # ---- invert ----
        prototype = None
        if do_invert:
            print(f"  Inverting {len(aligned)} images...")
            _, prototype, _ = inverter.invert_imgs(aligned, batch_size=args.batch_size)
            prototype.save(out_dir / "prototype.png")
            print(f"  Saved {out_dir}/prototype.png")
            torch.cuda.empty_cache()

        # ---- grid ----
        save_grid(out_dir / "grid.png", gt, aligned, prototype, name)
        print(f"  Saved {out_dir}/grid.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
