"""Publication figures render to files."""
import os

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pilitrack.config import AcquisitionConfig
from pilitrack.pipeline import detect_and_link, summarize
from pilitrack.singlechannel import make_cell_segmenter, make_pili_detector
from pilitrack import figures

CFG = AcquisitionConfig(dt_s=0.4, pixel_size_nm=43.3, ridge_sigmas=(1.5, 2.25, 3.0),
                        min_pilus_length_nm=200.0, base_search_radius_px=9.0,
                        max_base_jump_px=7.5)


def _movie(T=8, H=120, W=120):
    yy, xx = np.ogrid[:H, :W]
    cell = ((yy - 55) ** 2 + (xx - 55) ** 2) <= 9 ** 2
    frames = []
    for t in range(T):
        f = np.zeros((H, W), np.float32)
        f[cell] += 4000.0
        for k in range(6 + 3 * min(t, T - 1 - t)):
            f[55, 64 + k] += 500.0
        frames.append(gaussian_filter(f, 1.1) + 60.0)
    return np.stack(frames).astype(np.uint16)


@pytest.fixture
def run():
    pytest.importorskip("matplotlib")
    mov = _movie()
    art = detect_and_link(mov, mov, CFG,
                          segment_fn=make_cell_segmenter(min_cell_area_px=30),
                          detect_fn=make_pili_detector())
    res = summarize(art["tracks"], art["per_frame_cell_labels"], CFG, art["n_frames"])
    return art, res


def test_make_report_figures(run, tmp_path):
    art, res = run
    paths = figures.make_report_figures(res, art, CFG, tmp_path / "figs")
    assert paths and all(os.path.exists(p) for p in paths)


def test_length_traces_and_distributions(run, tmp_path):
    art, res = run
    p1 = figures.plot_length_traces(art["tracks"], CFG, art["n_frames"], tmp_path / "t.png")
    p2 = figures.plot_distributions(res["pilus"], tmp_path / "d.png")
    assert os.path.exists(p1) and os.path.exists(p2)


def test_single_kymograph(run, tmp_path):
    art, res = run
    if not art["tracks"]:
        pytest.skip("no track detected in this synthetic run")
    p = figures.plot_single_kymograph(art["tracks"][0], CFG, art["n_frames"],
                                      tmp_path / "k.png")
    assert os.path.exists(p)


def test_distributions_handles_empty(tmp_path):
    pytest.importorskip("matplotlib")
    import pandas as pd
    p = figures.plot_distributions(pd.DataFrame(), tmp_path / "e.png")
    assert os.path.exists(p)
