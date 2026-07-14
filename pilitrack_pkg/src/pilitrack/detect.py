"""Detection front-end.

Two responsibilities, kept behind simple function seams so the production
backends (Omnipose for cells; a trained ridge model for pili) can replace the
lightweight defaults without touching the rest of the pipeline.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato, threshold_otsu
from skimage.morphology import skeletonize
from skimage.measure import label

from .config import AcquisitionConfig


def _drop_small(mask: np.ndarray, min_size: int) -> np.ndarray:
    """Remove connected components with fewer than ``min_size`` pixels."""
    lbl, n = ndi.label(mask, structure=np.ones((3, 3), int))
    if n == 0:
        return mask
    sizes = np.bincount(lbl.ravel())
    keep = sizes >= min_size
    keep[0] = False
    return keep[lbl]


def segment_cells(cell_frame: np.ndarray, cfg: AcquisitionConfig) -> np.ndarray:
    """Return an integer label image of cell bodies for one frame.

    Default: Otsu + connected components (adequate for the synthetic movie and
    well-separated cells). For real rod-shaped Pseudomonas, swap this body for
    an Omnipose call and return its label image -- the interface is unchanged.
    """
    img = cell_frame.astype(float)
    try:
        thr = threshold_otsu(img)
    except ValueError:
        return np.zeros_like(img, dtype=int)
    mask = img > thr
    mask = _drop_small(mask, 4)
    return label(mask)


def skeletonize_probability(prob: np.ndarray, cfg: AcquisitionConfig) -> np.ndarray:
    """Skeletonize a pilus-probability map (e.g. an ilastik Pixel Classification
    export). ``prob`` is the single-class probability channel in [0, 1]."""
    mask = np.asarray(prob, dtype=float) > cfg.pilus_prob_threshold
    mask = _drop_small(mask, 3)
    return skeletonize(mask)


def detect_pili(fluor_frame: np.ndarray, cfg: AcquisitionConfig) -> np.ndarray:
    """Return a boolean skeleton mask of pilus filaments for one frame.

    Sato tubeness enhances thin bright ridges; threshold relative to the
    response max; skeletonize to 1-px centrelines.
    """
    img = fluor_frame.astype(float)
    ridge = sato(img, sigmas=cfg.ridge_sigmas, black_ridges=False)
    if ridge.max() <= 0:
        return np.zeros_like(img, dtype=bool)
    ridge = ridge / ridge.max()
    mask = ridge > cfg.detect_threshold
    mask = _drop_small(mask, 3)
    return skeletonize(mask)


def filament_components(skeleton: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected skeleton components (8-connectivity)."""
    lbl, n = ndi.label(skeleton, structure=np.ones((3, 3), int))
    return lbl, n
