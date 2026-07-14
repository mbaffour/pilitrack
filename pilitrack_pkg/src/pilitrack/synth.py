"""Synthetic data with ground truth, for validating the pipeline before any
real acquisition exists.

``make_kinetic_trace`` builds a length(t) trace from prescribed extension /
dwell / retraction cycles -> used to unit-test the velocity extraction.
``make_movie`` renders labelled cells + pili into a fluorescence stack so the
full image pipeline can be exercised end-to-end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AcquisitionConfig


@dataclass
class TraceTruth:
    length_nm: np.ndarray
    v_ext_nm_s: float
    v_ret_nm_s: float
    max_length_nm: float
    n_cycles: int


def make_kinetic_trace(
    cfg: AcquisitionConfig,
    v_ext_nm_s: float = 500.0,
    v_ret_nm_s: float = 500.0,
    dwell_s: float = 1.0,
    rest_s: float = 2.0,
    max_length_nm: float = 2000.0,
    n_cycles: int = 2,
    noise_nm: float = 0.0,
    rng: np.random.Generator | None = None,
) -> TraceTruth:
    """One pilus: repeated extend -> dwell -> retract -> rest cycles.

    Velocities are in nm/s (P. aeruginosa T4P run ~500 nm/s). Returns the
    sampled length(t) plus the ground-truth values it was built from.
    """
    rng = rng or np.random.default_rng(0)
    dt = cfg.dt_s
    seg_lengths_s = []
    seg_kinds = []
    t_ext = max_length_nm / v_ext_nm_s
    t_ret = max_length_nm / v_ret_nm_s
    for _ in range(n_cycles):
        seg_lengths_s += [t_ext, dwell_s, t_ret, rest_s]
        seg_kinds += ["ext", "dwell", "ret", "rest"]

    total_s = sum(seg_lengths_s)
    n = int(np.ceil(total_s / dt)) + 1
    t = np.arange(n) * dt

    length = np.zeros(n)
    cur = 0.0
    edges = np.cumsum([0.0] + seg_lengths_s)
    for ti, tt in enumerate(t):
        # find current segment
        seg = np.searchsorted(edges, tt, side="right") - 1
        seg = min(seg, len(seg_kinds) - 1)
        into = tt - edges[seg]
        kind = seg_kinds[seg]
        # length at start of this segment
        base = 0.0
        # reconstruct length at segment start deterministically
        L = 0.0
        for k in seg_kinds[:seg]:
            if k == "ext":
                L = max_length_nm
            elif k == "ret":
                L = 0.0
        if kind == "ext":
            length[ti] = min(max_length_nm, L + v_ext_nm_s * into)
        elif kind == "ret":
            length[ti] = max(0.0, max_length_nm - v_ret_nm_s * into)
        else:  # dwell holds max, rest holds 0
            length[ti] = max_length_nm if kind == "dwell" else 0.0

    if noise_nm > 0:
        length = length + rng.normal(0, noise_nm, size=length.shape)
        length = np.clip(length, 0, None)

    return TraceTruth(length, v_ext_nm_s, v_ret_nm_s, max_length_nm, n_cycles)


@dataclass
class MovieTruth:
    stack: np.ndarray             # (T, H, W) fluorescence (pili) channel
    cell_stack: np.ndarray        # (T, H, W) cell-body channel
    cell_poles_px: list           # [(y, x), ...] emission points
    pilus_angles: list            # radians, one per cell
    traces_nm: list               # ground-truth length(t) per cell's pilus
    cfg: AcquisitionConfig


def _draw_line(img, y0, x0, angle, length_px, amp, sigma):
    """Add a soft filament from (y0,x0) along ``angle`` of ``length_px``."""
    if length_px <= 0:
        return
    n = max(2, int(length_px * 3))
    ts = np.linspace(0, length_px, n)
    ys = y0 + ts * np.sin(angle)
    xs = x0 + ts * np.cos(angle)
    H, W = img.shape
    yy = np.clip(np.round(ys).astype(int), 0, H - 1)
    xx = np.clip(np.round(xs).astype(int), 0, W - 1)
    img[yy, xx] += amp


def make_movie(
    cfg: AcquisitionConfig,
    n_cells: int = 4,
    shape: tuple = (128, 128),
    piliated_fraction: float = 0.75,
    background: float = 20.0,
    pilus_amp: float = 200.0,
    noise: float = 8.0,
    seed: int = 0,
    **trace_kwargs,
) -> MovieTruth:
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    # build a reference trace to know the frame count
    ref = make_kinetic_trace(cfg, rng=rng, **trace_kwargs)
    T = ref.length_nm.size
    H, W = shape

    stack = np.zeros((T, H, W), dtype=float)
    cell_stack = np.zeros((T, H, W), dtype=float)
    poles, angles, traces = [], [], []

    margin = 20
    for c in range(n_cells):
        cy = rng.uniform(margin, H - margin)
        cx = rng.uniform(margin, W - margin)
        angle = rng.uniform(0, 2 * np.pi)
        # cell body: short rod ~ blob at (cy,cx)
        piliated = rng.random() < piliated_fraction
        tr = make_kinetic_trace(cfg, rng=rng, **trace_kwargs)
        poles.append((cy, cx))
        angles.append(angle)
        traces.append(tr.length_nm if piliated else np.zeros(T))
        yy, xx = np.ogrid[:H, :W]
        body = ((yy - cy) ** 2 + (xx - cx) ** 2) <= 3.0 ** 2
        for t in range(T):
            cell_stack[t][body] += 4000.0
            if piliated:
                L_px = tr.length_nm[t] / cfg.pixel_size_nm
                _draw_line(stack[t], cy, cx, angle, L_px, pilus_amp, 1.0)

    for t in range(T):
        stack[t] = gaussian_filter(stack[t], 1.1) + background
        stack[t] = rng.poisson(np.clip(stack[t], 0, None)) + rng.normal(0, noise, (H, W))
        cell_stack[t] = gaussian_filter(cell_stack[t], 3.0) + background

    stack = np.clip(stack, 0, None)
    return MovieTruth(stack, cell_stack, poles, angles, traces, cfg)
