"""End-to-end pipeline: two-channel stack -> the metrics requested.

Split into three stages so a curation step can sit in the middle:
  detect_and_link  -> per-frame filaments, cell labels, and linked tracks
  summarize        -> per-pilus / per-cell / population tables (optionally on a
                      culled subset of track ids)
  analyze_movie    -> convenience wrapper running both with no curation

Outputs (all over the analysis window):
  * per-pilus kinetics: length(t), max length, extension/retraction velocities
  * per-cell: number of pili, max pilus length
  * population: percentage of piliated cells
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AcquisitionConfig
from .detect import segment_cells, detect_pili, skeletonize_probability
from .measure import extract_filaments, associate_to_cells
from .track import link_tracks
from .kinetics import summarize_pilus


def _compact_labels(lbl) -> np.ndarray:
    """Downcast a cell-label image to the smallest integer dtype that holds it.

    ``skimage.measure.label`` (and a plain ``.astype(int)``) give int32/int64, so
    the per-frame label images of a 1952x1952 x 70-frame movie cost ~2 GB on
    their own. Cell counts are small, so uint8/uint16 is lossless and 4-8x
    smaller — the difference between a movie that fits in RAM and one that does
    not. Non-integer input is rounded; anything with negative labels is left
    alone."""
    a = np.asarray(lbl)
    if a.dtype == bool or a.size == 0:
        return a
    if not np.issubdtype(a.dtype, np.integer):
        a = np.rint(a).astype(np.int64)
    if int(a.min()) < 0:
        return a
    mx = int(a.max())
    dtype = np.uint8 if mx <= 255 else (np.uint16 if mx <= 65535 else np.uint32)
    return a.astype(dtype, copy=False)


def detect_and_link(
    fluor_stack: np.ndarray,
    cell_stack: np.ndarray,
    cfg: AcquisitionConfig,
    segment_fn=None,
    detect_fn=None,
    cell_label_stack: np.ndarray | None = None,
    pilus_prob_stack: np.ndarray | None = None,
    progress=None,
) -> dict:
    """Run detection + tracking and return intermediate artifacts.

    Detection backends are pluggable, in priority order:
      cells  : ``cell_label_stack`` (precomputed, e.g. Omnipose batch) >
               ``segment_fn(frame, cfg)`` callable > built-in Otsu default.
      pili   : ``pilus_prob_stack`` (precomputed, e.g. ilastik probabilities) >
               ``detect_fn(frame, cfg)`` callable > built-in Sato ridge default.

    ``progress(done, total)`` is called after each frame if given. A frame that
    raises is *isolated*: it is recorded in the returned ``failed_frames`` and
    contributes no pili, instead of aborting a long movie near the end. Failures
    are surfaced (report + app), never silently swallowed.
    """
    T = fluor_stack.shape[0]

    def _cells(t):
        if cell_label_stack is not None:
            return cell_label_stack[t]
        if segment_fn is not None:
            return segment_fn(cell_stack[t], cfg)
        return segment_cells(cell_stack[t], cfg)

    def _skel(t):
        if pilus_prob_stack is not None:
            return skeletonize_probability(pilus_prob_stack[t], cfg)
        if detect_fn is not None:
            return detect_fn(fluor_stack[t], cfg)
        return detect_pili(fluor_stack[t], cfg)

    per_frame_filaments = []
    per_frame_cell_labels = []
    failed_frames = []
    for t in range(T):
        try:
            cells = _compact_labels(_cells(t))
            skel = _skel(t)
            fils = associate_to_cells(extract_filaments(skel, cfg), cells, cfg)
        except Exception as exc:      # isolate the bad frame, keep the movie
            failed_frames.append({"frame": int(t),
                                  "error": f"{type(exc).__name__}: {exc}"})
            cells = np.zeros(fluor_stack.shape[1:], np.uint8)
            fils = []
        per_frame_filaments.append(fils)
        per_frame_cell_labels.append(cells)
        if progress is not None:
            progress(t + 1, T)

    tracks = link_tracks(per_frame_filaments, cfg)
    return {
        "per_frame_filaments": per_frame_filaments,
        "per_frame_cell_labels": per_frame_cell_labels,
        "tracks": tracks,
        "n_frames": T,
        "shape": fluor_stack.shape[1:],
        "failed_frames": failed_frames,
    }


def summarize(
    tracks: list,
    per_frame_cell_labels: list,
    cfg: AcquisitionConfig,
    n_frames: int,
    keep_ids: set | None = None,
) -> dict:
    """Turn tracks into the per-pilus / per-cell / population tables.

    ``keep_ids``: if given, only tracks whose ``track_id`` is in the set are
    included -- this is how the curation step drops spurious tracks before the
    kinetics are computed.
    """
    if keep_ids is not None:
        tracks = [tr for tr in tracks if tr.track_id in keep_ids]

    # --- per-pilus kinetics ---
    pilus_rows = []
    for tr in tracks:
        series_nm = tr.length_series(n_frames) * cfg.pixel_size_nm
        finite = np.where(~np.isnan(series_nm))[0]
        if finite.size < 2:
            continue
        # Take the span from first to last detection and interpolate any interior
        # gap frames (tracks may bridge up to cfg.max_gap_frames). Feeding the
        # gap-collapsed array to the kinetics would shrink a 2*dt interval to dt
        # and overreport that segment's velocity ~2x; the interpolated span keeps
        # the time axis uniform so velocities/phase durations stay correct.
        span = series_nm[finite[0]:finite[-1] + 1].copy()
        gap = np.isnan(span)
        if gap.any():
            idx = np.arange(span.size)
            span[gap] = np.interp(idx[gap], idx[~gap], span[~gap])
        summ = summarize_pilus(span, cfg)
        pilus_rows.append({
            "track_id": tr.track_id,
            "cell_id": tr.cell_id,
            "n_frames": int(finite.size),
            "max_length_nm": summ["max_length_nm"],
            "n_extension_events": summ["n_extension_events"],
            "n_retraction_events": summ["n_retraction_events"],
            "mean_extension_velocity_nm_s": summ["mean_extension_velocity_nm_s"],
            "mean_retraction_velocity_nm_s": summ["mean_retraction_velocity_nm_s"],
        })
    pilus_df = pd.DataFrame(pilus_rows)

    # A "real" pilus is a track that persists and reaches a real length.
    if not pilus_df.empty:
        qual = pilus_df[(pilus_df["n_frames"] >= cfg.min_piliation_frames)
                        & (pilus_df["max_length_nm"] >= cfg.min_pilus_length_nm)
                        & pilus_df["cell_id"].notna()].copy()
    else:
        qual = pilus_df

    # --- per-cell: number of distinct pili over the window + max length ---
    n_cells_total = int(max((cl.max() for cl in per_frame_cell_labels), default=0))
    cell_rows = []
    for c in range(1, n_cells_total + 1):
        owned = qual[qual["cell_id"] == c] if not qual.empty else qual
        n_pili = len(owned)
        cell_rows.append({
            "cell_id": c,
            "n_pili": n_pili,
            "max_pilus_length_nm": float(owned["max_length_nm"].max()) if n_pili else 0.0,
            "piliated": n_pili > 0,
        })
    cell_df = pd.DataFrame(cell_rows)

    # --- population ---
    n_piliated = int(cell_df["piliated"].sum()) if not cell_df.empty else 0
    pct = 100.0 * n_piliated / n_cells_total if n_cells_total else np.nan
    population = {
        "n_cells": n_cells_total,
        "n_piliated_cells": n_piliated,
        "percent_piliated": pct,
        "window_s": cfg.window_s,
    }
    return {"pilus": pilus_df, "cell": cell_df, "population": population}


def analyze_movie(
    fluor_stack: np.ndarray,
    cell_stack: np.ndarray,
    cfg: AcquisitionConfig,
    segment_fn=None,
    detect_fn=None,
    cell_label_stack: np.ndarray | None = None,
    pilus_prob_stack: np.ndarray | None = None,
) -> dict:
    """``fluor_stack``/``cell_stack``: (T, H, W). Returns three DataFrames.

    Convenience wrapper: detect + link + summarize with no curation.
    """
    art = detect_and_link(
        fluor_stack, cell_stack, cfg,
        segment_fn=segment_fn, detect_fn=detect_fn,
        cell_label_stack=cell_label_stack, pilus_prob_stack=pilus_prob_stack,
    )
    return summarize(art["tracks"], art["per_frame_cell_labels"],
                     cfg, art["n_frames"])
