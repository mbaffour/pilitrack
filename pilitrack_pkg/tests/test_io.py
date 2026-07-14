"""Tests for the real-movie loader and auto-config.

The heavy paths (opening a real ND2) need the optional ``nd2`` package and a
sample file, so they are guarded / skipped. The load-bearing logic —
axis reordering, timestamp->dt, and the physical rescaling of the config — is
pure and tested directly, no file or ``nd2`` install required.
"""
import numpy as np
import pytest

from pilitrack.config import AcquisitionConfig
from pilitrack.io import _to_txy, _dt_from_events, _event_times, config_from_nd2
from pilitrack.singlechannel import make_cell_segmenter, make_pili_detector
from pilitrack.pipeline import detect_and_link, summarize


# --------------------------------------------------------------------------- #
# _to_txy — axis handling
# --------------------------------------------------------------------------- #
def test_to_txy_already_txy():
    a = np.zeros((7, 8, 9), np.uint16)
    out = _to_txy(a, {"T": 7, "Y": 8, "X": 9})
    assert out.shape == (7, 8, 9)


def test_to_txy_selects_channel():
    b = np.zeros((5, 2, 8, 8), np.uint16)  # T, C, Y, X
    b[:, 1] = 7
    out = _to_txy(b, {"T": 5, "C": 2, "Y": 8, "X": 8}, channel=1)
    assert out.shape == (5, 8, 8)
    assert (out == 7).all()


def test_to_txy_adds_missing_time_axis():
    c = np.ones((8, 8), np.uint16)
    out = _to_txy(c, {"Y": 8, "X": 8})
    assert out.shape == (1, 8, 8)


def test_to_txy_handles_z_plane():
    d = np.arange(2 * 3 * 4 * 4).reshape(2, 3, 4, 4)  # T, Z, Y, X
    out = _to_txy(d, {"T": 2, "Z": 3, "Y": 4, "X": 4}, z=1)
    assert out.shape == (2, 4, 4)
    assert np.array_equal(out, d[:, 1])


def test_to_txy_rejects_unknown_axis():
    with pytest.raises(ValueError):
        _to_txy(np.zeros((2, 2, 2)), {"P": 2, "Y": 2, "X": 2})


# --------------------------------------------------------------------------- #
# timestamps -> dt
# --------------------------------------------------------------------------- #
def test_dt_from_events_median():
    ev = [{"Time [s]": t} for t in [0.0, 0.44, 0.87, 1.31]]
    dt = _dt_from_events(ev)
    assert dt == pytest.approx(0.437, abs=0.01)


def test_dt_from_events_missing_column():
    assert _dt_from_events([{"Index": 0}, {"Index": 1}]) is None
    assert _dt_from_events([]) is None


def test_event_times_ignores_nonfinite():
    ev = [{"Time [s]": 0.0}, {"Time [s]": float("nan")}, {"Time [s]": 1.0}]
    times = _event_times(ev)
    assert times.tolist() == [0.0, 1.0]


# --------------------------------------------------------------------------- #
# config_from_nd2 — physical rescaling
# --------------------------------------------------------------------------- #
def _meta(px=43.333, dt=0.4376):
    return dict(pixel_size_nm=px, dt_s=dt, duration_s=30.0,
                channel_names=["488 nm"], n_timepoints=70, shape_yx=(1952, 1952))


def test_config_rescales_pixel_unit_params():
    cfg = config_from_nd2(_meta())
    # scale = 65 / 43.333 = 1.5
    assert cfg.pixel_size_nm == pytest.approx(43.333)
    assert cfg.dt_s == pytest.approx(0.4376)
    assert cfg.base_search_radius_px == pytest.approx(9.0, abs=0.05)
    assert cfg.max_base_jump_px == pytest.approx(7.5, abs=0.05)
    assert tuple(cfg.ridge_sigmas) == pytest.approx((1.5, 2.25, 3.0), abs=0.02)


def test_config_keeps_physical_length_params():
    cfg = config_from_nd2(_meta())
    assert cfg.min_pilus_length_nm == 300.0        # unchanged (already physical)
    assert cfg.max_pilus_length_nm == 10000.0
    # derived px value tracks the real pixel size
    assert cfg.min_pilus_length_px == pytest.approx(300.0 / 43.333, abs=0.01)


def test_config_raises_velocity_eps_above_pixel_quantization():
    # one pixel-step per frame ~ 43.333/0.4376 = 99 nm/s; eps = half of that ~ 49.5
    cfg = config_from_nd2(_meta())
    assert cfg.velocity_sign_eps_nm_s == pytest.approx(49.5, abs=1.0)
    assert cfg.velocity_sign_eps_nm_s >= 20.0       # never below the base floor


def test_config_overrides_win():
    cfg = config_from_nd2(_meta(), dt_s=1.0, detect_threshold=0.1,
                          min_pilus_length_nm=200.0)
    assert cfg.dt_s == 1.0
    assert cfg.detect_threshold == 0.1
    assert cfg.min_pilus_length_nm == 200.0


def test_config_requires_dt_when_missing():
    m = _meta()
    m["dt_s"] = None
    with pytest.raises(ValueError):
        config_from_nd2(m)
    # ... unless supplied as an override
    cfg = config_from_nd2(m, dt_s=0.5)
    assert cfg.dt_s == 0.5


# --------------------------------------------------------------------------- #
# single-channel backends drive the pipeline end-to-end (synthetic frame)
# --------------------------------------------------------------------------- #
def test_single_channel_backends_end_to_end():
    from scipy.ndimage import gaussian_filter
    H = W = 120
    img = np.full((H, W), 100.0, np.float32)
    yy, xx = np.ogrid[:H, :W]
    img[((yy - 60) ** 2 + (xx - 60) ** 2) <= 10 ** 2] += 3000.0   # cell blob
    for k in range(35):
        img[60, 70 + k] += 400.0                                   # pilus
    img = gaussian_filter(img, 1.2).astype(np.uint16)
    stack = np.stack([img] * 6)

    cfg = AcquisitionConfig(dt_s=0.44, pixel_size_nm=43.3,
                            ridge_sigmas=(1.5, 2.25, 3.0),
                            min_pilus_length_nm=200.0,
                            base_search_radius_px=9.0, max_base_jump_px=7.5)
    seg = make_cell_segmenter(open_radius_px=6, min_cell_area_px=30)
    det = make_pili_detector(tophat_radius_px=6)

    art = detect_and_link(stack, stack, cfg, segment_fn=seg, detect_fn=det)
    res = summarize(art["tracks"], art["per_frame_cell_labels"], cfg, art["n_frames"])
    assert res["population"]["n_cells"] >= 1
    assert not res["pilus"].empty


# --------------------------------------------------------------------------- #
# guarded smoke test against a real nd2 install (skipped without it)
# --------------------------------------------------------------------------- #
def test_load_nd2_smoke(tmp_path):
    pytest.importorskip("nd2")
    import os
    from pilitrack.io import load_nd2
    sample = (os.path.dirname(__file__)
              + "/../../Labelled data/trial01007.nd2")
    if not os.path.exists(sample):
        pytest.skip("no sample ND2 file available")
    stack, meta = load_nd2(sample, frames=slice(0, 2))
    assert stack.ndim == 3 and stack.shape[0] == 2
    assert meta["pixel_size_nm"] > 0
