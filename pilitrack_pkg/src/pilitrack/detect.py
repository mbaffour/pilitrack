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

    Sato tubeness enhances thin bright ridges; threshold relative to a robust
    high percentile of the response; skeletonize to 1-px centrelines.
    """
    img = fluor_frame.astype(float)
    ridge = sato(img, sigmas=cfg.ridge_sigmas, black_ridges=False)
    pos = ridge[ridge > 0]
    if pos.size == 0:
        return np.zeros_like(img, dtype=bool)
    # Normalize by a high percentile of the positive response, NOT the global
    # max: one hot/saturated pixel (or a cosmic ray) gives a huge Sato response
    # that would rescale every real pilus below the fixed threshold and blank the
    # whole frame. The 99.5th percentile is robust to such single-pixel outliers.
    hi = np.percentile(pos, 99.5)
    if hi <= 0:
        hi = float(ridge.max())
    ridge = np.clip(ridge / hi, 0, 1)
    mask = hysteresis_mask(ridge, cfg)
    mask = _drop_small(mask, 3)
    return skeletonize(mask)


def hysteresis_mask(norm_ridge: np.ndarray, cfg: AcquisitionConfig) -> np.ndarray:
    """Double-threshold (hysteresis) mask on a [0,1]-normalized ridge response.

    Keeps pixels above the low threshold that are connected to a *core* above
    ``detect_threshold`` — recovering the faint distal end of a pilus (which
    tapers below a single hard cut) and bridging momentary dropouts, without
    lowering the global noise floor (isolated faint pixels are not kept).
    Degenerates to a single threshold when ``hysteresis_low_frac`` is >=1 or <=0.
    """
    high = float(cfg.detect_threshold)
    frac = float(getattr(cfg, "hysteresis_low_frac", 1.0))
    if not (0.0 < frac < 1.0):
        return norm_ridge > high
    from skimage.filters import apply_hysteresis_threshold
    return apply_hysteresis_threshold(norm_ridge, high * frac, high)


def filament_components(skeleton: np.ndarray) -> tuple[np.ndarray, int]:
    """Label connected skeleton components (8-connectivity)."""
    lbl, n = ndi.label(skeleton, structure=np.ones((3, 3), int))
    return lbl, n
