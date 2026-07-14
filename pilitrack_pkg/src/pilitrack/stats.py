"""Statistical aggregation for publication tables.

Two things a paper needs beyond raw medians: **confidence intervals**
(bootstrap, no distributional assumption) and **hierarchical** aggregation —
pili within a cell are not independent, so pooling every pilus inflates n and
understates uncertainty. ``kinetics_table(level="per_cell")`` averages within
each cell first, then summarizes across cells; ``level="per_pilus"`` pools. Both
report n, median, mean, and a bootstrap CI so you can state effects honestly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# report-name -> per-pilus column in the summarize() pilus table
_METRICS = {
    "extension_velocity_nm_s": "mean_extension_velocity_nm_s",
    "retraction_velocity_nm_s": "mean_retraction_velocity_nm_s",
    "max_length_nm": "max_length_nm",
}
# report-name -> per-cell column produced by per_cell_table
_METRICS_CELL = {
    "extension_velocity_nm_s": "mean_ext_nm_s",
    "retraction_velocity_nm_s": "mean_ret_nm_s",
    "max_length_nm": "mean_max_length_nm",
}


def bootstrap_ci(values, statistic=np.median, n_boot: int = 2000,
                 ci: float = 95.0, seed: int = 0):
    """(point estimate, ci_low, ci_high) by bootstrap resampling. Deterministic
    for a given ``seed`` (reproducible CIs)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(v))
    if v.size == 1:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    boots = statistic(v[idx], axis=1)
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return (point, lo, hi)


def describe(values, statistic=np.median, **kw) -> dict:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    point, lo, hi = bootstrap_ci(v, statistic=statistic, **kw)
    return {
        "n": int(v.size),
        "median": float(np.median(v)) if v.size else float("nan"),
        "mean": float(np.mean(v)) if v.size else float("nan"),
        "point": point, "ci_low": lo, "ci_high": hi,
    }


def per_cell_table(pilus_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the per-pilus table to one row per cell (drops pili with no
    cell). Columns: ``cell_id, n_pili, mean_max_length_nm, mean_ext_nm_s,
    mean_ret_nm_s``."""
    if pilus_df is None or pilus_df.empty or "cell_id" not in pilus_df:
        return pd.DataFrame(columns=["cell_id", "n_pili", "mean_max_length_nm",
                                     "mean_ext_nm_s", "mean_ret_nm_s"])
    df = pilus_df[pilus_df["cell_id"].notna()]
    if df.empty:
        return pd.DataFrame(columns=["cell_id", "n_pili", "mean_max_length_nm",
                                     "mean_ext_nm_s", "mean_ret_nm_s"])
    g = df.groupby("cell_id")
    return g.agg(
        n_pili=("track_id", "count"),
        mean_max_length_nm=("max_length_nm", "mean"),
        mean_ext_nm_s=("mean_extension_velocity_nm_s", "mean"),
        mean_ret_nm_s=("mean_retraction_velocity_nm_s", "mean"),
    ).reset_index()


def kinetics_table(pilus_df: pd.DataFrame, level: str = "per_pilus",
                   n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Summary table (n, median, mean, bootstrap CI) for each kinetic metric.

    ``level='per_pilus'`` pools all pili; ``level='per_cell'`` averages within
    each cell first (hierarchical — the honest n is the number of cells)."""
    if level == "per_cell":
        src, cols = per_cell_table(pilus_df), _METRICS_CELL
    elif level == "per_pilus":
        src, cols = pilus_df, _METRICS
    else:
        raise ValueError(f"level must be 'per_pilus' or 'per_cell', got {level!r}")
    rows = []
    for name, col in cols.items():
        vals = src[col] if (src is not None and not src.empty and col in src) else []
        d = describe(vals, n_boot=n_boot, seed=seed)
        d.update(metric=name, level=level)
        rows.append(d)
    return pd.DataFrame(rows)[["metric", "level", "n", "median", "mean",
                               "point", "ci_low", "ci_high"]]


def combine_pili(pilus_dfs, names=None) -> pd.DataFrame:
    """Concatenate several movies' per-pilus tables, tagged by movie, for a
    pooled cross-movie analysis."""
    frames = []
    names = names or [f"movie{i}" for i in range(len(pilus_dfs))]
    for name, df in zip(names, pilus_dfs):
        if df is not None and not df.empty:
            d = df.copy()
            d.insert(0, "movie", name)
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
