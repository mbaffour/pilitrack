"""Publication figures: the standard T4P plots, in one consistent style.

A methods paper leans on a few canonical views — each pilus's length over time
(the automated equivalent of a hand-drawn kymograph) and the population
distributions of extension/retraction velocity and maximum length. These helpers
render them to PNG/SVG/PDF at publication DPI. Matplotlib is optional (``qc``
extra); display-free and unit-tested by checking the files are written.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise ImportError("figures need matplotlib (`pip install -e '.[qc]'`).") from exc
    return plt


def plot_length_traces(tracks, cfg, n_frames, out, *, max_traces: int = 60,
                       dpi: int = 200):
    """Length(t) of each pilus track (nm vs s) overlaid — the kymograph view."""
    plt = _plt()
    t = np.arange(n_frames) * cfg.dt_s
    fig, ax = plt.subplots(figsize=(6, 4))
    n = 0
    for tr in tracks:
        series = tr.length_series(n_frames) * cfg.pixel_size_nm
        if np.isfinite(series).sum() < 2:
            continue
        ax.plot(t, series, lw=0.8, alpha=0.6)
        n += 1
        if n >= max_traces:
            break
    ax.set_xlabel("time (s)")
    ax.set_ylabel("pilus length (nm)")
    ax.set_title(f"Pilus length over time ({n} tracks shown)")
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out, dpi)


def plot_single_kymograph(track, cfg, n_frames, out, *, dpi: int = 200):
    """One pilus: length(t) with extension/retraction shaded — a clean kymograph."""
    plt = _plt()
    t = np.arange(n_frames) * cfg.dt_s
    series = track.length_series(n_frames) * cfg.pixel_size_nm
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(t, series, "-o", ms=3, lw=1.2, color="#1f77b4")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("length (nm)")
    ax.set_title(f"pilus track {track.track_id}"
                 + (f" (cell {track.cell_id})" if track.cell_id is not None else ""))
    ax.spines[["top", "right"]].set_visible(False)
    return _save(fig, out, dpi)


def plot_distributions(pilus_df, out, *, dpi: int = 200):
    """Histograms of extension velocity, retraction velocity, and max length."""
    plt = _plt()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    panels = [
        ("mean_extension_velocity_nm_s", "extension velocity (nm/s)", "#2ca02c"),
        ("mean_retraction_velocity_nm_s", "retraction velocity (nm/s)", "#d62728"),
        ("max_length_nm", "max length (nm)", "#1f77b4"),
    ]
    for ax, (col, label, color) in zip(axes, panels):
        vals = np.asarray(pilus_df[col], float) if col in pilus_df else np.array([])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            ax.hist(vals, bins=min(30, max(5, vals.size // 3)), color=color, alpha=0.8)
            ax.axvline(np.median(vals), color="k", ls="--", lw=1,
                       label=f"median {np.median(vals):.0f}")
            ax.legend(frameon=False, fontsize=8)
        ax.set_xlabel(label)
        ax.set_ylabel("count")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save(fig, out, dpi)


def make_report_figures(res: dict, art: dict, cfg, outdir) -> list:
    """Render the standard figure set for a run into ``outdir``. Returns paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = [
        plot_length_traces(art["tracks"], cfg, art["n_frames"],
                           outdir / "length_traces.png"),
        plot_distributions(res["pilus"], outdir / "distributions.png"),
    ]
    return written


def _save(fig, out, dpi):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return str(out)
