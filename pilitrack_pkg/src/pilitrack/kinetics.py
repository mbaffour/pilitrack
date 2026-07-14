"""Extract extension / dwell / retraction phases and velocities from a
single pilus length-vs-time trace.

This is the measurement that matters, so it is deliberately independent of the
imaging front-end: it takes a 1-D array of lengths (in nm) sampled at ``dt_s``
and returns labelled phases with per-phase slopes. It is unit-tested directly
against synthetic traces with known ground-truth velocities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from .config import AcquisitionConfig


def _odd_window(requested: int, n: int) -> int:
    """Largest usable odd Savitzky-Golay window <= n, >= 5 when possible."""
    w = max(5, requested if requested % 2 == 1 else requested + 1)
    if w > n:
        w = n if n % 2 == 1 else n - 1
    return max(3, w)


def _smooth_derivative(length_nm: np.ndarray, dt: float, window: int) -> np.ndarray:
    """Per-frame slope (nm/s) via a Savitzky-Golay derivative -- far more
    noise-robust than a raw finite difference for sign classification."""
    n = length_nm.size
    w = _odd_window(window, n)
    if w < 3 or n < 3:
        d = np.gradient(length_nm, dt)
        return d
    poly = min(2, w - 1)
    return savgol_filter(length_nm, w, poly, deriv=1, delta=dt, mode="interp")


@dataclass
class Phase:
    kind: str          # "extension" | "dwell" | "retraction"
    start_frame: int
    end_frame: int     # inclusive
    velocity_nm_s: float   # signed; ~0 for dwell

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


def _classify(slope_nm_s: float, eps: float) -> str:
    if slope_nm_s > eps:
        return "extension"
    if slope_nm_s < -eps:
        return "retraction"
    return "dwell"


def segment_trace(length_nm: np.ndarray, cfg: AcquisitionConfig) -> list[Phase]:
    """Segment one length(t) trace into contiguous kinetic phases.

    Strategy: smooth, take per-frame slope, classify each interval by sign,
    merge consecutive same-kind intervals, drop runs shorter than
    ``min_phase_frames`` (folding them into a neighbour), then fit the slope of
    each surviving run by least squares for a robust velocity estimate.
    """
    length_nm = np.asarray(length_nm, dtype=float)
    n = length_nm.size
    if n < 2:
        return []

    dt = cfg.dt_s
    eps = cfg.velocity_sign_eps_nm_s

    # per-frame slope (frame space, length n) then classify each frame
    deriv = _smooth_derivative(length_nm, dt, cfg.smoothing_window)
    labels = [_classify(s, eps) for s in deriv]

    # group consecutive equal labels into runs over frames [kind, f0, f1]
    runs: list[list] = []
    for i, lab in enumerate(labels):
        if runs and runs[-1][0] == lab:
            runs[-1][2] = i
        else:
            runs.append([lab, i, i])

    # drop too-short runs by merging into the longer neighbour
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for idx, (kind, a, b) in enumerate(runs):
            if (b - a + 1) < cfg.min_phase_frames:
                left = runs[idx - 1] if idx > 0 else None
                right = runs[idx + 1] if idx < len(runs) - 1 else None
                if left and (not right or (left[2] - left[1]) >= (right[2] - right[1])):
                    left[2] = b
                    runs.pop(idx)
                elif right:
                    right[1] = a
                    runs.pop(idx)
                changed = True
                break

    # fit each run's velocity on raw data, trimming transition frames at the
    # boundaries (they blend into the neighbouring dwell and bias the slope)
    phases: list[Phase] = []
    for kind, f0, f1 in runs:
        a, b = f0, f1
        if (b - a + 1) >= 4:              # long enough to trim both ends
            a, b = f0 + 1, f1 - 1
        frames = np.arange(a, b + 1)
        if frames.size >= 2:
            v = float(np.polyfit(frames * dt, length_nm[a:b + 1], 1)[0])
        else:
            v = float(np.polyfit(np.arange(f0, f1 + 1) * dt,
                                 length_nm[f0:f1 + 1], 1)[0]) if f1 > f0 else 0.0
        phases.append(Phase(_classify(v, eps), f0, f1, v))

    # merge adjacent phases that ended up the same kind after fitting, and
    # RE-CLASSIFY the merged run from its own slope -- keeping the old label
    # while recomputing the slope could leave an "extension" phase with a
    # negative velocity (or vice versa).
    merged: list[Phase] = []
    for p in phases:
        if merged and merged[-1].kind == p.kind:
            prev = merged[-1]
            f0, f1 = prev.start_frame, p.end_frame
            t = np.arange(f0, f1 + 1) * dt
            y = length_nm[f0:f1 + 1]
            v = float(np.polyfit(t, y, 1)[0]) if f1 > f0 else 0.0
            merged[-1] = Phase(_classify(v, eps), f0, f1, v)
        else:
            merged.append(p)
    return merged


def summarize_pilus(length_nm: np.ndarray, cfg: AcquisitionConfig) -> dict:
    """Per-pilus kinetic summary used by the pipeline."""
    phases = segment_trace(length_nm, cfg)
    # sign guard: an extension speed is the positive slope of an extension phase,
    # a retraction speed the positive magnitude of a negative-slope phase. This
    # can never report a negative extension/retraction velocity.
    ext = [p.velocity_nm_s for p in phases
           if p.kind == "extension" and p.velocity_nm_s > 0]
    ret = [-p.velocity_nm_s for p in phases
           if p.kind == "retraction" and p.velocity_nm_s < 0]
    length_nm = np.asarray(length_nm, dtype=float)
    return {
        "max_length_nm": float(np.nanmax(length_nm)) if length_nm.size else np.nan,
        "n_extension_events": len(ext),
        "n_retraction_events": len(ret),
        "mean_extension_velocity_nm_s": float(np.mean(ext)) if ext else np.nan,
        "mean_retraction_velocity_nm_s": float(np.mean(ret)) if ret else np.nan,
        "phases": phases,
    }
