"""Automatic quality control for a run.

"Clean data always" means never handing back numbers a bad input silently
corrupted. ``qc_metrics`` measures the movie and the detection result, and
``qc_flags`` turns those into explicit warnings (saturation, defocus, no cells,
sparse detection, out-of-range kinetics) that ride along in every manifest and
batch summary — so a questionable movie is *flagged*, not averaged in.
"""
from __future__ import annotations

import numpy as np

# Published P. aeruginosa T4P sanity envelope (nm, nm/s).
SANE_VELOCITY_NM_S = (50.0, 2000.0)
SANE_MAXLEN_NM = (300.0, 5000.0)
SATURATION_WARN_FRACTION = 0.005   # >0.5% of pixels saturated -> flag
LOW_DETECTION_RATE = 0.5           # <50% of frames show any pilus -> flag
# Physically impossible individual outputs (over-links / tracking jumps).
IMPLAUSIBLE_LENGTH_NM = 5000.0     # a single pilus longer than ~5 um
IMPLAUSIBLE_VELOCITY_NM_S = 2000.0 # a single event faster than ~2 um/s


def _saturation_level(stack) -> float:
    """Value at which this stack is 'saturated'."""
    if np.issubdtype(stack.dtype, np.integer):
        return float(np.iinfo(stack.dtype).max)
    return float(np.percentile(stack, 99.99))


def _focus_score(frame) -> float:
    """Variance-of-Laplacian sharpness proxy (higher = sharper/in focus)."""
    from scipy import ndimage as ndi

    f = np.asarray(frame, dtype=np.float32)
    lap = ndi.laplace(f)
    return float(lap.var())


def qc_metrics(stack, art: dict, res: dict, cfg) -> dict:
    """Quantitative QC of one movie + its analysis result.

    ``art`` is a ``detect_and_link`` result; ``res`` a ``summarize`` result.
    Returns a flat dict of metrics plus a ``flags`` list.
    """
    stack = np.asarray(stack)
    T = int(stack.shape[0])
    sat_level = _saturation_level(stack)
    saturated_fraction = float(np.mean(stack >= sat_level))
    mid = T // 2

    per_fil = art.get("per_frame_filaments", [])
    n_filaments = int(sum(len(f) for f in per_fil))
    frames_with_pili = int(sum(1 for f in per_fil if len(f) > 0))
    detection_rate = frames_with_pili / T if T else 0.0
    n_tracks = int(len(art.get("tracks", [])))

    pili = res.get("pilus")
    pop = res.get("population", {}) or {}
    n_cells = int(pop.get("n_cells", 0) or 0)
    percent_piliated = pop.get("percent_piliated", None)

    n_qualified = 0
    med_ext = med_ret = med_len = None
    n_implausible_length = n_implausible_velocity = n_negative_velocity = 0
    if pili is not None and not pili.empty:
        qual = pili[(pili["n_frames"] >= cfg.min_piliation_frames)
                    & (pili["max_length_nm"] >= cfg.min_pilus_length_nm)
                    & pili["cell_id"].notna()]
        n_qualified = int(len(qual))
        med_ext = _nanmed(pili["mean_extension_velocity_nm_s"])
        med_ret = _nanmed(pili["mean_retraction_velocity_nm_s"])
        med_len = _nanmed(pili["max_length_nm"])
        # physically-impossible individual outputs (over-links / tracking jumps)
        vel = np.concatenate([
            np.asarray(pili["mean_extension_velocity_nm_s"], float),
            np.asarray(pili["mean_retraction_velocity_nm_s"], float)])
        vel = vel[np.isfinite(vel)]
        n_implausible_length = int((np.asarray(pili["max_length_nm"], float)
                                    > IMPLAUSIBLE_LENGTH_NM).sum())
        n_implausible_velocity = int((vel > IMPLAUSIBLE_VELOCITY_NM_S).sum())
        n_negative_velocity = int((vel < 0).sum())

    metrics = {
        "n_frames": T,
        "shape_yx": (int(stack.shape[1]), int(stack.shape[2])),
        "intensity_median": float(np.median(stack)),
        "intensity_max": float(stack.max()),
        "saturated_fraction": saturated_fraction,
        "focus_score_midframe": _focus_score(stack[mid]),
        "n_filaments_total": n_filaments,
        "mean_filaments_per_frame": n_filaments / T if T else 0.0,
        "detection_rate": detection_rate,
        "n_tracks": n_tracks,
        "n_cells": n_cells,
        "n_qualified_pili": n_qualified,
        "qualified_fraction": (n_qualified / n_tracks) if n_tracks else 0.0,
        "percent_piliated": percent_piliated,
        "median_extension_velocity_nm_s": med_ext,
        "median_retraction_velocity_nm_s": med_ret,
        "median_max_length_nm": med_len,
        "n_implausible_length": n_implausible_length,
        "n_implausible_velocity": n_implausible_velocity,
        "n_negative_velocity": n_negative_velocity,
    }
    metrics["flags"] = qc_flags(metrics)
    return metrics


def qc_flags(metrics: dict) -> list[str]:
    """Turn QC metrics into human-readable warnings ('' none = clean)."""
    flags: list[str] = []
    if metrics["saturated_fraction"] > SATURATION_WARN_FRACTION:
        flags.append(
            f"saturation: {metrics['saturated_fraction']*100:.1f}% of pixels at "
            f"sensor max — intensities clipped; lengths near bright cells may be off")
    if metrics["n_cells"] == 0:
        flags.append("no cells segmented — pili-only (per-cell %piliated unavailable)")
    if metrics["detection_rate"] < LOW_DETECTION_RATE:
        flags.append(
            f"sparse detection: only {metrics['detection_rate']*100:.0f}% of frames "
            f"show a pilus — check focus / detect_threshold")
    for name, key in (("extension", "median_extension_velocity_nm_s"),
                      ("retraction", "median_retraction_velocity_nm_s")):
        v = metrics.get(key)
        if v is not None and not (SANE_VELOCITY_NM_S[0] <= v <= SANE_VELOCITY_NM_S[1]):
            flags.append(
                f"{name} velocity {v:.0f} nm/s outside {SANE_VELOCITY_NM_S} — "
                f"check dt_s / pixel_size_nm")
    ml = metrics.get("median_max_length_nm")
    if ml is not None and not (SANE_MAXLEN_NM[0] <= ml <= SANE_MAXLEN_NM[1]):
        flags.append(f"median max length {ml:.0f} nm outside {SANE_MAXLEN_NM} — "
                     f"check pixel_size_nm / detection")
    # physically-impossible individual outputs (the artifact tail)
    if metrics.get("n_implausible_length", 0) > 0:
        flags.append(
            f"{metrics['n_implausible_length']} pilus(i) longer than "
            f"{IMPLAUSIBLE_LENGTH_NM/1000:.0f} um — likely over-linked/crossing "
            f"tracks; report medians and inspect the length tail")
    if metrics.get("n_implausible_velocity", 0) > 0:
        flags.append(
            f"{metrics['n_implausible_velocity']} velocity event(s) faster than "
            f"{IMPLAUSIBLE_VELOCITY_NM_S/1000:.0f} um/s — likely single-frame "
            f"tracking jumps; use the median, not the mean")
    if metrics.get("n_negative_velocity", 0) > 0:
        flags.append(
            f"{metrics['n_negative_velocity']} negative velocity value(s) — "
            f"a kinetics classification error; please report this")
    return flags


def _nanmed(series):
    arr = np.asarray(series, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.median(arr)) if arr.size else None
