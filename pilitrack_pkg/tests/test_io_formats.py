"""Multi-format loader: axis handling, channel resolution, TIFF, arrays."""
import numpy as np
import pytest

from pilitrack.io import (
    _to_tcyx, _guess_pili_channel, _guess_cell_channel,
    load_array, load_movie, config_from_meta, config_from_nd2,
)


# ---- _to_tcyx generalized axis handling ---- #
def test_to_tcyx_tyx_adds_c():
    out = _to_tcyx(np.zeros((5, 8, 8)), {"T": 5, "Y": 8, "X": 8})
    assert out.shape == (5, 1, 8, 8)


def test_to_tcyx_tcyx_identity():
    a = np.zeros((5, 2, 8, 8))
    out = _to_tcyx(a, {"T": 5, "C": 2, "Y": 8, "X": 8})
    assert out.shape == (5, 2, 8, 8)


def test_to_tcyx_projects_z():
    a = np.zeros((3, 4, 8, 8))  # T, Z, Y, X
    a[:, 2] = 9
    out = _to_tcyx(a, {"T": 3, "Z": 4, "Y": 8, "X": 8}, z="max")
    assert out.shape == (3, 1, 8, 8)
    assert (out == 9).all()


def test_to_tcyx_lone_z_is_time():
    # frames on Z with no T axis (plain multi-page/ImageJ stack) must stay as
    # time, NOT be max-projected into a single frame.
    out = _to_tcyx(np.zeros((7, 8, 8)), {"Z": 7, "Y": 8, "X": 8})
    assert out.shape == (7, 1, 8, 8)
    out2 = _to_tcyx(np.zeros((7, 2, 8, 8)), {"Z": 7, "C": 2, "Y": 8, "X": 8})
    assert out2.shape == (7, 2, 8, 8)


def test_to_tcyx_generic_sequence_axis_is_time():
    # tifffile labels a plain multi-page TIFF's page axis 'Q' (or 'I'); it must
    # be treated as time, not rejected as an unsupported axis.
    for stack_axis in ("Q", "I"):
        out = _to_tcyx(np.zeros((6, 8, 8)), {stack_axis: 6, "Y": 8, "X": 8})
        assert out.shape == (6, 1, 8, 8)


def test_config_from_meta_pixel_size_none_uses_override():
    cfg = config_from_meta({"pixel_size_nm": None, "dt_s": 0.4, "duration_s": 5.0},
                           pixel_size_nm=65.0)
    assert cfg.pixel_size_nm == 65.0
    # and with no override it should error clearly, not TypeError on float(None)
    with pytest.raises(ValueError):
        config_from_meta({"pixel_size_nm": None, "dt_s": 0.4})


def test_guess_pili_channel_avoids_phase_without_fluor_keyword():
    # neither name has a fluorophore keyword; must still avoid the phase channel
    assert _guess_pili_channel(["Phase", "Pili"], 2) == 1
    assert _guess_cell_channel(["Phase", "Pili"], 2, 1) == 0


def test_plain_multipage_tiff_loads_all_frames(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    mov = (np.random.default_rng(1).random((7, 32, 32)) * 300).astype(np.uint16)
    p = tmp_path / "plain.tif"
    tifffile.imwrite(str(p), mov)          # no ImageJ/OME metadata -> axes 'QYX'
    fluor, cell, meta = load_movie(str(p))
    assert meta["n_timepoints"] == 7 and fluor.shape[0] == 7


def test_frames_subset_slices_frame_times(tmp_path):
    # duration must reflect the selected frames, not the whole movie
    tifffile = pytest.importorskip("tifffile")
    mov = (np.random.default_rng(2).random((10, 16, 16)) * 300).astype(np.uint16)
    p = tmp_path / "t.ome.tif"
    tifffile.imwrite(str(p), mov, metadata={"axes": "TYX", "TimeIncrement": 0.5})
    _, _, meta = load_movie(str(p), frames=slice(2, 5))
    assert meta["n_timepoints"] == 3


def test_to_tcyx_selects_position():
    a = np.zeros((2, 3, 6, 6))  # P, T, Y, X
    a[1] = 5
    out = _to_tcyx(a, {"P": 2, "T": 3, "Y": 6, "X": 6}, position=1)
    assert out.shape == (3, 1, 6, 6)
    assert (out == 5).all()


# ---- channel role guessing ---- #
def test_guess_pili_channel_by_name():
    assert _guess_pili_channel(["Phase", "488 nm"], 2) == 1
    assert _guess_pili_channel(["GFP"], 1) == 0


def test_guess_cell_channel_prefers_transmitted():
    assert _guess_cell_channel(["488 nm", "Phase"], 2, pili_channel=0) == 1
    # two-channel fallback: the other channel
    assert _guess_cell_channel(["chA", "chB"], 2, pili_channel=0) == 1
    # single channel -> none
    assert _guess_cell_channel(["488 nm"], 1, pili_channel=0) is None


# ---- load_array ---- #
def test_load_array_single_channel():
    f, c, m = load_array(np.zeros((4, 8, 8), np.uint16), axes="TYX",
                         pixel_size_nm=50.0, dt_s=0.5)
    assert f.shape == (4, 8, 8)
    assert c is None
    assert m["single_channel"] and m["n_channels"] == 1


def test_load_array_dual_channel_roles():
    a = np.zeros((4, 2, 8, 8), np.uint16)
    a[:, 1] = 3
    f, c, m = load_array(a, axes="TCYX", channel_names=["488 nm", "Phase"])
    assert m["pili_channel"] == 0 and m["cell_channel"] == 1
    assert c is not None and (c == 3).all()
    assert not m["single_channel"]


# ---- TIFF round-trip through the real reader ---- #
def test_load_movie_ome_tiff_roundtrip(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = np.random.randint(0, 500, size=(6, 16, 16), dtype=np.uint16)
    path = tmp_path / "movie.ome.tif"
    tifffile.imwrite(str(path), data, metadata={
        "axes": "TYX", "PhysicalSizeX": 0.065, "PhysicalSizeXUnit": "µm",
        "TimeIncrement": 0.5})
    f, c, m = load_movie(str(path))
    assert f.shape == (6, 16, 16)
    assert c is None and m["single_channel"]
    assert m["pixel_size_nm"] == pytest.approx(65.0, abs=1.0)
    assert m["dt_s"] == pytest.approx(0.5, abs=0.01)
    # and a config can be derived from it
    cfg = config_from_meta(m)
    assert cfg.pixel_size_nm == pytest.approx(65.0, abs=1.0)


def test_load_movie_tiff_frames_roi(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    data = np.arange(8 * 20 * 20, dtype=np.uint16).reshape(8, 20, 20)
    path = tmp_path / "m.tif"
    tifffile.imwrite(str(path), data, metadata={"axes": "TYX"})
    f, c, m = load_movie(str(path), frames=slice(2, 5), roi=(4, 12, 6, 14))
    assert f.shape == (3, 8, 8)


def test_config_from_nd2_alias_is_config_from_meta():
    assert config_from_nd2 is config_from_meta


def test_load_movie_rejects_unknown_format(tmp_path):
    p = tmp_path / "x.foo"
    p.write_bytes(b"nope")
    with pytest.raises(ValueError):
        load_movie(str(p))
