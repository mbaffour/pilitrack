"""Run the two-channel pipeline on a *single*-channel labelled-pilus movie.

Labelled-pilus TIRF (Alexa488-maleimide on a PilA cysteine knock-in) is often a
single fluorescence channel in which the cell body appears as a bright compact
blob and pili as thin faint ridges radiating from it. The pipeline wants two
inputs (a cell-body channel and a pilus channel); here we serve both roles from
the one channel by separating them *morphologically*, and feed the results
through the pipeline's existing ``segment_fn`` / ``detect_fn`` seams. The
validated core (``detect.detect_pili`` / ``detect.segment_cells``) is left
untouched, so the synthetic tests are unaffected.

Cells   : grey-opening removes structures thinner than the disk (pili) and keeps
          compact blobs (cells) -> Otsu -> connected components.
Pili    : white top-hat removes structures *larger* than the disk (the cell
          body) and flattens the uneven TIRF illumination, *before* the Sato
          ridge filter; the ridge response is normalized by a high percentile
          (not the global max, which a single bright cell/artifact would set)
          and cell interiors are masked out so cell-edge bloom is not mistaken
          for a pilus. These are the real-SNR "extract better" fixes.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import sato, threshold_otsu
from skimage.measure import label
from skimage.morphology import disk, opening, skeletonize, white_tophat

from .detect import _drop_small


def _cell_mask(
    frame,
    cfg,
    open_radius_px: float = 6.0,
    method: str = "robust",
    flatten_sigma: float | None = None,
    winsor_percentile: float = 98.0,
) -> np.ndarray:
    """Boolean mask of compact cell bodies in one frame.

    Fluorescent cells vary enormously in brightness (surface-label load differs
    cell to cell; some saturate) and TIRF illumination is uneven, so a single
    global Otsu threshold catches only the brightest cells and misses the rest,
    while a log stretch over-amplifies diffuse background into false cells. The
    default ``'robust'`` does both corrections and nothing more:

    1. subtract a large-Gaussian background (removes the illumination gradient
       and diffuse haze) — ``flatten_sigma`` px, defaulted to ~2.5 µm worth of
       pixels from ``cfg.pixel_size_nm`` so it generalizes across magnifications;
    2. winsorize the bright outliers at ``winsor_percentile`` so a handful of
       saturated cells no longer drag the Otsu threshold above the dim ones;
    3. grey-opening (disk ``open_radius_px``) erases thin structures (pili);
    4. Otsu on the result — now a threshold that separates faint *and* bright
       cells uniformly, with tight footprints.

    ``method='otsu'`` is the legacy single global Otsu (cheap; catches only the
    brightest cells). ``method='log-otsu'`` is the log stretch (over-segments
    background — kept for comparison).
    """
    img = np.asarray(frame, dtype=np.float32)
    if method == "robust":
        sigma = flatten_sigma or (2500.0 / float(cfg.pixel_size_nm))
        flat = np.clip(img - ndi.gaussian_filter(img, float(sigma)), 0, None)
        work = np.minimum(flat, np.percentile(flat, float(winsor_percentile)))
    elif method == "log-otsu":
        floor = float(np.percentile(img, 1.0))
        work = np.log1p(np.clip(img - floor, 0, None))
    elif method == "otsu":
        work = img
    else:
        raise ValueError(f"unknown cell-mask method {method!r}")
    opened = opening(work, disk(int(round(open_radius_px))))
    try:
        thr = threshold_otsu(opened)
    except ValueError:
        return np.zeros(img.shape, dtype=bool)
    return opened > thr


def make_cell_segmenter(
    open_radius_px: float = 6.0,
    min_cell_area_px: int = 80,
    method: str = "robust",
    flatten_sigma: float | None = None,
    winsor_percentile: float = 98.0,
):
    """A ``segment_fn(frame, cfg) -> int label image`` for cell bodies.

    Uses the dynamic-range-robust mask (background-flatten + winsorize + Otsu)
    by default so faint and bright cells are both segmented across the whole
    field. Instance separation is connected-components only (touching cells may
    merge; watershed / Omnipose is the upgrade for dense fields — pass a
    ``segment_fn`` from ``pilitrack.backends.omnipose_backend`` instead, or a
    real cell/phase channel via ``cell_stack``).
    """
    def segment_fn(frame, cfg) -> np.ndarray:
        mask = _cell_mask(frame, cfg, open_radius_px, method=method,
                          flatten_sigma=flatten_sigma,
                          winsor_percentile=winsor_percentile)
        mask = _drop_small(mask, int(min_cell_area_px))
        return label(mask)

    return segment_fn


def cell_labels_from_projection(
    stack,
    cfg,
    proj: str = "median",
    open_radius_px: float = 6.0,
    min_cell_area_px: int = 30,
) -> np.ndarray:
    """Segment cells once on a temporal projection of the movie.

    A median (default) projection averages out moving pili while reinforcing
    stationary cell bodies, giving a clean single ``(Y, X)`` label image. Fast
    and low-noise, but assumes cells barely move over the acquisition; for
    twitching fields prefer the per-frame ``make_cell_segmenter``.
    """
    fn = {"median": np.median, "mean": np.mean, "max": np.max}[proj]
    proj_img = fn(np.asarray(stack, dtype=np.float32), axis=0)
    mask = _cell_mask(proj_img, cfg, open_radius_px)
    mask = _drop_small(mask, int(min_cell_area_px))
    return label(mask)


def make_pili_detector(
    tophat_radius_px: float = 6.0,
    norm_percentile: float = 99.5,
    exclude_cell_interior: bool = True,
    open_radius_px: float = 6.0,
    erosion_radius_px: float = 2.0,
):
    """A ``detect_fn(frame, cfg) -> bool skeleton`` tuned for real TIRF SNR.

    Fixes the built-in detector's weaknesses on a reused single channel:
    1. white top-hat removes the bright cell body and flattens illumination
       before the ridge filter;
    2. cell interiors (eroded, so the pilus base at the rim survives) are zeroed
       so cell-edge bloom is not detected as a pilus;
    3. the ridge response is normalized by ``norm_percentile`` instead of its
       global max, so one hot pixel cannot suppress every faint pilus.
    Threshold, skeletonize, drop-small — same contract as ``detect.detect_pili``.
    """
    tophat_fp = disk(int(round(tophat_radius_px)))
    erode_fp = disk(int(round(erosion_radius_px)))

    def detect_fn(frame, cfg) -> np.ndarray:
        img = np.asarray(frame, dtype=np.float32)
        flat = white_tophat(img, tophat_fp)
        ridge = sato(flat, sigmas=cfg.ridge_sigmas, black_ridges=False)

        if exclude_cell_interior:
            # cheap bright-cell mask (method='otsu'): the white top-hat already
            # suppresses dim cell bodies, so only bright cells can bloom into a
            # false ridge; masking their eroded interior keeps the pilus base.
            cm = _cell_mask(img, cfg, open_radius_px, method="otsu")
            if cm.any():
                interior = ndi.binary_erosion(cm, structure=erode_fp)
                ridge = ridge.copy()
                ridge[interior] = 0.0

        pos = ridge[ridge > 0]
        if pos.size == 0:
            return np.zeros(img.shape, dtype=bool)
        hi = np.percentile(pos, norm_percentile)
        if hi <= 0:
            hi = float(ridge.max())
        if hi <= 0:
            return np.zeros(img.shape, dtype=bool)

        ridge = np.clip(ridge / hi, 0.0, 1.0)
        mask = ridge > cfg.detect_threshold
        mask = _drop_small(mask, 3)
        return skeletonize(mask)

    return detect_fn
