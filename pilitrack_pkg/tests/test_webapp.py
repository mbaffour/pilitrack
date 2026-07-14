"""Browser-app helpers (no Streamlit needed for these)."""
import numpy as np
import pandas as pd
import pytest
from scipy.ndimage import gaussian_filter

from pilitrack.webapp import (overlay_rgb, _measurements, analyze_for_web,
                              _click_to_image_yx, _draw_labels)
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


def test_click_to_image_yx_scales_from_display_to_full():
    # click at display (x=100, y=50) on a 500x250 shown image over a 1000x500 crop
    val = {"x": 100, "y": 50, "width": 500, "height": 250, "unix_time": 1}
    yx = _click_to_image_yx(val, Himg=500, Wimg=1000)   # (Himg, Wimg) full-res
    assert yx == [100.0, 200.0]                          # [y*2, x*2]


def test_click_to_image_yx_clamps_and_rejects_empty():
    assert _click_to_image_yx({}, 100, 100) is None
    # out-of-image click is clamped into bounds
    yx = _click_to_image_yx({"x": 999, "y": -5, "width": 100, "height": 100}, 100, 100)
    assert yx == [0.0, 99.0]


def test_draw_labels_returns_display_sized_image():
    bg = np.zeros((100, 200, 3), np.uint8)
    committed = [[[10, 20], [30, 40]]]
    wip = [[50, 60], [55, 65]]
    im, disp_h = _draw_labels(bg, committed, wip, disp_w=100)
    assert im.size == (100, disp_h) and disp_h == 50   # aspect preserved (200x100 -> 100x50)


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


def test_analyze_for_web_downsample_halves_and_rescales(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    mov = (np.random.default_rng(0).random((4, 160, 160)) * 800).astype(np.uint16)
    p = tmp_path / "big.ome.tif"
    tifffile.imwrite(str(p), mov, metadata={
        "axes": "TYX", "PhysicalSizeX": 0.0433, "PhysicalSizeXUnit": "µm",
        "TimeIncrement": 0.4})
    r1 = analyze_for_web(str(p), fast=False, downsample=1)
    r2 = analyze_for_web(str(p), fast=False, downsample=2)
    assert r2["fluor"].shape[1] == r1["fluor"].shape[1] // 2       # strided
    assert r2["cfg"].pixel_size_nm == pytest.approx(r1["cfg"].pixel_size_nm * 2)
