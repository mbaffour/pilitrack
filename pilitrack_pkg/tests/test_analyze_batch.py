"""End-to-end orchestrator + batch + QC on synthetic single-channel movies."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from pilitrack.analyze import analyze_file, build_config
from pilitrack.config import AcquisitionConfig
from pilitrack.batch import run_batch, find_movies
from pilitrack.qc import qc_metrics, qc_flags
from pilitrack.io import config_from_meta


def _synthetic_single_channel(T=8, H=110, W=110, seed=0):
    """One bright cell blob + a pilus that extends then retracts, one channel."""
    rng = np.random.default_rng(seed)
    stack = np.zeros((T, H, W), np.float32)
    yy, xx = np.ogrid[:H, :W]
    cell = ((yy - 55) ** 2 + (xx - 55) ** 2) <= 9 ** 2
    for t in range(T):
        f = np.zeros((H, W), np.float32)
        f[cell] += 4000.0
        length = int(6 + 3 * min(t, T - 1 - t))  # grow then shrink
        for k in range(length):
            f[55, 64 + k] += 500.0
        f = gaussian_filter(f, 1.1) + 60.0
        stack[t] = f + rng.normal(0, 5, (H, W))
    return np.clip(stack, 0, None).astype(np.uint16)


def test_analyze_file_array_end_to_end(tmp_path):
    mov = _synthetic_single_channel()
    result = analyze_file(
        array=mov, array_axes="TYX", path="<array>",
        out=tmp_path / "out",
        overrides={"pixel_size_nm": 43.3, "dt_s": 0.4, "min_pilus_length_nm": 200.0},
        verbose=False)
    assert result["meta"]["single_channel"]
    assert result["res"]["population"]["n_cells"] >= 1
    assert not result["res"]["pilus"].empty
    # outputs written
    assert (tmp_path / "out" / "pili.csv").exists()
    assert (tmp_path / "out" / "manifest.json").exists()
    man = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert man["acquisition_config"]["pixel_size_nm"] == pytest.approx(43.3)
    assert man["detection_params"]["tophat_radius_px"] == 6.0
    # detect_threshold routed into the config (the detector reads cfg.detect_threshold)
    assert man["acquisition_config"]["detect_threshold"] == 0.3


def test_build_config_routes_detect_threshold_to_cfg():
    meta = {"pixel_size_nm": 43.3, "dt_s": 0.4, "duration_s": 10.0}
    cfg, det = build_config(meta, overrides={"detect_threshold": 0.42,
                                             "tophat_radius_px": 8.0})
    assert cfg.detect_threshold == 0.42        # config field
    assert det["tophat_radius_px"] == 8.0      # detector-only
    assert "detect_threshold" not in det
    # default applied when not given
    cfg2, _ = build_config(meta, overrides={})
    assert cfg2.detect_threshold == 0.30


def test_qc_flags_detect_problems():
    # saturation + no cells + out-of-range velocity
    m = {"saturated_fraction": 0.05, "n_cells": 0, "detection_rate": 1.0,
         "median_extension_velocity_nm_s": 9000.0,
         "median_retraction_velocity_nm_s": 400.0,
         "median_max_length_nm": 1000.0}
    flags = qc_flags(m)
    assert any("saturation" in f for f in flags)
    assert any("no cells" in f for f in flags)
    assert any("extension velocity" in f for f in flags)


def test_qc_metrics_on_synthetic():
    mov = _synthetic_single_channel()
    result = analyze_file(array=mov, array_axes="TYX", path="<array>",
                          overrides={"pixel_size_nm": 43.3, "dt_s": 0.4,
                                     "min_pilus_length_nm": 200.0}, verbose=False)
    qc = qc_metrics(mov, result["art"], result["res"], result["cfg"])
    assert 0.0 <= qc["detection_rate"] <= 1.0
    assert qc["n_frames"] == mov.shape[0]
    assert "flags" in qc


def test_run_batch_over_tiff_folder(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    folder = tmp_path / "movies"
    folder.mkdir()
    for i in range(2):
        mov = _synthetic_single_channel(seed=i)
        tifffile.imwrite(str(folder / f"movie_{i}.tif"), mov,
                         metadata={"axes": "TYX"})
    out = tmp_path / "batch"
    res = run_batch(folder, out, overrides={"pixel_size_nm": 43.3, "dt_s": 0.4,
                                            "min_pilus_length_nm": 200.0,
                                            "detect_threshold": 0.3},
                    save_overlays=False, verbose=False)
    assert len(res["summary"]) == 2
    assert (out / "summary.csv").exists()
    assert (out / "batch_manifest.json").exists()
    # each movie got its own result folder + manifest
    assert (out / "movie_0" / "manifest.json").exists()


def test_pilus_length_timeseries():
    from pilitrack.analyze import pilus_length_timeseries
    from pilitrack.track import Track
    cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=50.0)
    tr = Track(0, 3, frames=[0, 2], length_px=[10.0, 20.0],
               tips=[(0, 0), (0, 0)], base_yx=(0, 0))
    df = pilus_length_timeseries([tr], cfg, n_frames=3)
    assert list(df.columns) == ["track_id", "cell_id", "frame", "time_s", "length_nm"]
    assert len(df) == 2                                   # frame 1 absent (NaN) dropped
    assert df.loc[df.frame == 0, "length_nm"].iloc[0] == 500.0   # 10 px * 50 nm
    assert df.loc[df.frame == 2, "time_s"].iloc[0] == 1.0        # 2 * 0.5 s


def test_analyze_file_writes_length_timeseries(tmp_path):
    mov = _synthetic_single_channel()
    analyze_file(array=mov, array_axes="TYX", path="<array>", out=tmp_path / "o",
                 overrides={"pixel_size_nm": 43.3, "dt_s": 0.4,
                            "min_pilus_length_nm": 200.0}, verbose=False)
    ts = tmp_path / "o" / "pilus_length_over_time.csv"
    assert ts.exists()
    df = pd.read_csv(ts)
    assert {"track_id", "cell_id", "frame", "time_s", "length_nm"} <= set(df.columns)


def test_find_movies(tmp_path):
    (tmp_path / "a.nd2").write_bytes(b"x")
    (tmp_path / "b.tif").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    found = find_movies(tmp_path)
    names = {p.name for p in found}
    assert names == {"a.nd2", "b.tif"}
