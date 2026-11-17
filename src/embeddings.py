"""Load cached embeddings, dropping generations whose face alignment failed (black image)."""
import numpy as np
import torch
from pathlib import Path


def load_embed(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=True).numpy()


def _is_black(img_path: Path, threshold: int = 10) -> bool:
    """Return True if the aligned image is fully black (alignment failure)."""
    try:
        from PIL import Image
        import numpy as _np
        arr = _np.array(Image.open(img_path).convert("L"))
        return arr.mean() < threshold
    except Exception:
        return False


def count_black_images(aligned_dir: Path, n: int, threshold: int = 10) -> int:
    """Return number of black (alignment-failed) images out of n."""
    if not aligned_dir.exists():
        return 0
    return sum(_is_black(aligned_dir / f"{i}.png") for i in range(n))


def load_embed_masked(embed_path: Path, aligned_dir: Path, strict: bool = False) -> np.ndarray | None:
    """Load embeddings and drop rows whose aligned image is black.

    Args:
        embed_path:  path to the .pt embedding file (shape k × dim)
        aligned_dir: directory containing 0.png, 1.png, … for this name/model
        strict:      if True, return None if *any* image is black (not just all)

    Returns None if the file is missing or all rows are masked out (or any, if strict).
    """
    E = load_embed(embed_path)
    if E is None:
        return None
    # If the aligned dir doesn't exist, fall back to unmasked (no images to check)
    if not aligned_dir.exists():
        return E
    valid = [i for i in range(len(E)) if not _is_black(aligned_dir / f"{i}.png")]
    if not valid:
        return None
    if strict and len(valid) < len(E):
        return None
    if len(valid) == len(E):
        return E
    return E[valid]

