"""Reference similarity and the two probes from the paper (Sec. 3).

E: [k, d] face embeddings of the k generations for one name.
"""
import numpy as np


def norm_rows(E: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)


def centroid(E: np.ndarray) -> np.ndarray:
    """L2-normalised mean of the row-normalised embeddings (ê in the paper)."""
    c = norm_rows(E).mean(axis=0)
    return c / (np.linalg.norm(c) + 1e-8)


def calc_gt_sim(E: np.ndarray, E_gt: np.ndarray) -> float:
    """Reference similarity s_gt: mean cosine similarity of generations to the reference photo."""
    return float((norm_rows(E) @ norm_rows(E_gt).T).mean())


def calc_dispersion(E: np.ndarray) -> float:
    """Dispersion δ: mean squared distance of the generations from their centroid."""
    return float(((E - E.mean(axis=0)) ** 2).sum(axis=1).mean())


def calc_centroid_sim(c: np.ndarray, other_centroids: np.ndarray) -> float:
    """Centroid similarity s_cen: max cosine similarity of this name's centroid to other names' centroids."""
    if len(other_centroids) == 0:
        return 0.0
    return float((other_centroids @ c).max())
