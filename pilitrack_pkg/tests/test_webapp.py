"""Browser-app helpers (no Streamlit needed for these)."""
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from pilitrack.webapp import overlay_rgb, _measurements, analyze_for_web
from pilitrack.measure import Filament


def test_overlay_rgb_marks_pili_magenta():
    frame = np.zeros((20, 20), float)
    frame[5:15, 5:15] = 1000.0
    cells = np.zeros((20, 20), int)
    cells[6:14, 6:14] = 1
    fil = Filament(1, 5.0, (10, 3), (10, 8),
                   np.array([[10, c] for c in range(3, 9)]))  # left of the cell
    rgb = overlay_rgb(frame, cells, [fil])
    assert rgb.shape == (20, 20, 3) and rgb.dtype == np.uint8
    px = rgb[10, 3]           # a pilus pixel -> magenta [255, 38, 229]
    assert px[0] > 200 and px[2] > 200 and px[1] < 100


def test_measurements_summary():
    res = {"population": {"n_cells": 5, "n_piliated_cells": 3, "percent_piliated": 60.0},
           "cell": pd.DataFrame({"cell_id": [1, 2, 3], "n_pili": [2, 1, 3],
                                 "piliated": [True, True, True]}),
           "pilus": pd.DataFrame()}
    qc = {"median_max_length_nm": 1000.0, "median_extension_velocity_nm_s": 300.0,
          "median_retraction_velocity_nm_s": 350.0, "n_qualified_pili": 6}
    m = _measurements(res, qc)
    assert m["percent_piliated"] == "60%"
    assert m["pili_per_cell"] == "2.0"
    assert m["median_ext_nm_s"] == "300"


def test_analyze_for_web_on_tiff(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    H = W = 120
    yy, xx = np.ogrid[:H, :W]
    cell = ((yy - 55) ** 2 + (xx - 55) ** 2) <= 9 ** 2
    frames = []
    for _ in range(6):
        f = np.zeros((H, W), np.float32)
        f[cell] += 4000.0
        for k in range(20):
            f[55, 64 + k] += 500.0
        frames.append(gaussian_filter(f, 1.1) + 60.0)
    mov = np.clip(np.stack(frames), 0, None).astype(np.uint16)
    p = tmp_path / "m.ome.tif"
    tifffile.imwrite(str(p), mov, metadata={
        "axes": "TYX", "PhysicalSizeX": 0.0433, "PhysicalSizeXUnit": "µm",
        "TimeIncrement": 0.4})
    r = analyze_for_web(str(p), detect_threshold=0.30, fast=False)
    assert {"fluor", "cfg", "art", "res", "qc", "meta"} <= set(r)
    assert r["fluor"].shape[0] == 6
    assert "flags" in r["qc"]
