"""Statistical aggregation: bootstrap CIs, per-cell / hierarchical tables."""
import numpy as np
import pandas as pd
import pytest

from pilitrack import stats


def _pilus_df():
    return pd.DataFrame({
        "track_id": [1, 2, 3, 4],
        "cell_id": [1, 1, 2, np.nan],
        "n_frames": [5, 5, 5, 5],
        "max_length_nm": [1000.0, 1200.0, 800.0, 2000.0],
        "mean_extension_velocity_nm_s": [300.0, 400.0, 350.0, np.nan],
        "mean_retraction_velocity_nm_s": [350.0, np.nan, 300.0, 400.0],
    })


def test_bootstrap_ci_deterministic_and_bracketing():
    vals = np.arange(1, 101).astype(float)
    p1 = stats.bootstrap_ci(vals, seed=0)
    p2 = stats.bootstrap_ci(vals, seed=0)
    assert p1 == p2                       # reproducible for a fixed seed
    point, lo, hi = p1
    assert point == pytest.approx(np.median(vals))
    assert lo < point < hi


def test_bootstrap_ci_edge_cases():
    assert stats.bootstrap_ci([]) == (float("nan"),) * 3 or np.isnan(
        stats.bootstrap_ci([])[0])
    assert stats.bootstrap_ci([7.0]) == (7.0, 7.0, 7.0)


def test_per_cell_table():
    pc = stats.per_cell_table(_pilus_df())
    assert set(pc["cell_id"]) == {1, 2}          # the NaN-cell pilus dropped
    row1 = pc[pc["cell_id"] == 1].iloc[0]
    assert row1["n_pili"] == 2
    assert row1["mean_max_length_nm"] == pytest.approx(1100.0)


def test_kinetics_table_levels_differ():
    df = _pilus_df()
    per_pilus = stats.kinetics_table(df, level="per_pilus")
    per_cell = stats.kinetics_table(df, level="per_cell")
    # per-pilus max_length pools 4 pili; per-cell aggregates to 2 cells
    mp = per_pilus[per_pilus["metric"] == "max_length_nm"].iloc[0]
    mc = per_cell[per_cell["metric"] == "max_length_nm"].iloc[0]
    assert mp["n"] == 4
    assert mc["n"] == 2                            # honest n = number of cells
    assert mp["median"] == pytest.approx(1100.0)   # median(1000,1200,800,2000)
    assert mc["median"] == pytest.approx(950.0)    # median(1100, 800)


def test_kinetics_table_bad_level():
    with pytest.raises(ValueError):
        stats.kinetics_table(_pilus_df(), level="nonsense")


def test_combine_pili_tags_movie():
    df = _pilus_df()
    out = stats.combine_pili([df, df], names=["m1", "m2"])
    assert set(out["movie"]) == {"m1", "m2"}
    assert len(out) == 8


def test_empty_inputs():
    empty = pd.DataFrame()
    assert stats.per_cell_table(empty).empty
    t = stats.kinetics_table(empty, level="per_pilus")
    assert (t["n"] == 0).all()
