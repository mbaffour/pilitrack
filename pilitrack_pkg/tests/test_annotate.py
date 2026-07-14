"""Hand-labeling core: geometry, folding annotations into the summary, I/O."""
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from pilitrack.config import AcquisitionConfig
from pilitrack.pipeline import detect_and_link
from pilitrack.singlechannel import make_cell_segmenter, make_pili_detector
from pilitrack import annotate
from pilitrack.annotate import (
    ManualPilus, Annotations, rasterize_polyline, polyline_length_px,
    manual_filament, apply_annotations, shapes_to_manual_pili,
    manual_pili_to_shapes, save_annotations, load_annotations)


CFG = AcquisitionConfig(dt_s=0.4, pixel_size_nm=43.3, ridge_sigmas=(1.5, 2.25, 3.0),
                        min_pilus_length_nm=200.0, base_search_radius_px=9.0,
                        max_base_jump_px=7.5)


def _cell_only_movie(T=6, H=120, W=120):
    """A movie with one bright cell blob and NO pili (so any pilus is manual)."""
    yy, xx = np.ogrid[:H, :W]
    cell = ((yy - 55) ** 2 + (xx - 55) ** 2) <= 9 ** 2
    frames = []
    for _ in range(T):
        f = np.zeros((H, W), np.float32)
        f[cell] += 4000.0
        frames.append(gaussian_filter(f, 1.1) + 60.0)
    return np.stack(frames).astype(np.uint16)


# ---- geometry ---- #
def test_rasterize_horizontal_line():
    coords = rasterize_polyline([[10, 5], [10, 20]], (40, 40))
    assert (coords[:, 0] == 10).all()
    assert coords[:, 1].min() == 5 and coords[:, 1].max() == 20


def test_polyline_length():
    assert polyline_length_px([[0, 0], [0, 3], [4, 3]]) == pytest.approx(7.0)


def test_manual_filament_length_and_ends():
    fil = manual_filament([[55, 64], [55, 80]], (120, 120))
    assert fil.length_px == pytest.approx(16.0)
    assert fil.base_yx == (55, 64) and fil.tip_yx == (55, 80)
    assert fil.coords.shape[0] > 0


# ---- fold into summary ---- #
def test_apply_annotations_adds_a_qualified_pilus():
    mov = _cell_only_movie()
    art = detect_and_link(mov, mov, CFG,
                          segment_fn=make_cell_segmenter(min_cell_area_px=30),
                          detect_fn=make_pili_detector())
    base = apply_annotations(art, Annotations(), CFG)
    n_before = len(base["summary"]["pilus"])

    # trace the SAME missed pilus on every frame, emanating from the cell edge
    pili = [ManualPilus(frame=t, points=[[55, 64], [55, 82]]) for t in range(mov.shape[0])]
    out = apply_annotations(art, Annotations(manual_pili=pili), CFG)
    pilus_df = out["summary"]["pilus"]
    assert len(pilus_df) > n_before
    # the manual pilus is associated to the cell and makes it piliated
    assert out["summary"]["population"]["n_piliated_cells"] >= 1
    # its measured length ~ 18 px * 43.3 nm ~ 780 nm
    assert pilus_df["max_length_nm"].max() > 500


def test_apply_annotations_respects_removed_ids_and_cell_override():
    mov = _cell_only_movie()
    art = detect_and_link(mov, mov, CFG,
                          segment_fn=make_cell_segmenter(min_cell_area_px=30),
                          detect_fn=make_pili_detector())
    pili = [ManualPilus(frame=t, points=[[55, 64], [55, 82]]) for t in range(mov.shape[0])]
    out = apply_annotations(art, Annotations(manual_pili=pili), CFG)
    tracks = out["art"]["tracks"]
    assert tracks, "expected at least one manual track"
    # removing every track id -> empty pilus table
    removed = Annotations(manual_pili=pili, removed_track_ids=[tr.track_id for tr in tracks])
    out2 = apply_annotations(art, Annotations(manual_pili=pili, removed_track_ids=removed.removed_track_ids), CFG)
    assert out2["summary"]["pilus"].empty
    # supplying an all-zero cell stack -> no association -> not piliated
    zeros = np.zeros_like(np.stack(art["per_frame_cell_labels"]))
    out3 = apply_annotations(art, Annotations(manual_pili=pili), CFG, cell_labels=zeros)
    assert out3["summary"]["population"]["n_cells"] == 0


# ---- napari shapes conversion ---- #
def test_shapes_roundtrip():
    pili = [ManualPilus(frame=3, points=[[10.0, 5.0], [10.0, 20.0]])]
    shapes = manual_pili_to_shapes(pili)
    assert shapes[0].shape == (2, 3)          # [t, y, x]
    back = shapes_to_manual_pili(shapes)
    assert back[0].frame == 3
    assert np.allclose(back[0].points, [[10, 5], [10, 20]])


def test_shapes_2d_uses_default_frame():
    out = shapes_to_manual_pili([np.array([[1.0, 2.0], [3.0, 4.0]])], default_frame=7)
    assert out[0].frame == 7


# ---- persistence ---- #
def test_save_load_annotations_json(tmp_path):
    ann = Annotations(manual_pili=[ManualPilus(2, [[1, 2], [3, 4]], cell_id=5)],
                      removed_track_ids=[9], movie="m.nd2", notes="hi")
    p = save_annotations(ann, tmp_path / "ann.json")
    loaded, cells = load_annotations(p)
    assert cells is None
    assert loaded.movie == "m.nd2" and loaded.removed_track_ids == [9]
    assert loaded.manual_pili[0].cell_id == 5
    assert loaded.manual_pili[0].frame == 2


def test_save_load_annotations_with_cells(tmp_path):
    pytest.importorskip("tifffile")
    ann = Annotations(manual_pili=[])
    cells = np.zeros((3, 8, 8), np.int32)
    cells[:, 2:5, 2:5] = 1
    p = save_annotations(ann, tmp_path / "ann.json", cell_labels=cells)
    _, loaded_cells = load_annotations(p)
    assert loaded_cells is not None and loaded_cells.shape == (3, 8, 8)
    assert loaded_cells.max() == 1
