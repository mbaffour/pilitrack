"""Pre-annotation: detection -> editable labels -> correct -> (no double count)."""
import numpy as np
import pytest

from pilitrack.config import AcquisitionConfig
from pilitrack.measure import Filament
from pilitrack.annotate import (annotations_from_art, apply_annotations,
                                _order_skeleton_path)


def _art_one_filament(shape=(40, 40)):
    coords = np.array([[20, c] for c in range(5, 25)])
    f = Filament(1, 20.0, (20, 5), (20, 24), coords, cell_id=3)
    return {"per_frame_filaments": [[f]],
            "per_frame_cell_labels": [np.zeros(shape, int)],
            "shape": shape, "n_frames": 1, "tracks": []}


def test_order_skeleton_path_base_to_tip():
    coords = np.array([[10, c] for c in range(3, 10)])
    p = _order_skeleton_path(coords, (10, 3))
    assert p[0] == [10, 3] and p[-1] == [10, 9]


def test_annotations_from_art_makes_editable_traces():
    ann = annotations_from_art(_art_one_filament())
    assert len(ann.manual_pili) == 1
    mp = ann.manual_pili[0]
    assert mp.frame == 0 and mp.cell_id == 3
    pts = np.asarray(mp.points)
    assert pts.shape[0] >= 2
    ends = sorted([pts[0].tolist(), pts[-1].tolist()])
    assert ends[0][1] < 10 and ends[-1][1] > 18       # spans the filament
    assert ann.meta["source"] == "auto-detection"


def test_replace_auto_avoids_double_count():
    art = _art_one_filament()
    cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=50.0, min_pilus_length_nm=200.0,
                            base_search_radius_px=9.0)
    ann = annotations_from_art(art)
    # seeded/complete set: exactly the one filament, not auto + seeded copy
    out = apply_annotations(art, ann, cfg, replace_auto=True)
    assert sum(len(f) for f in out["art"]["per_frame_filaments"]) == 1
    # add mode would count both
    out2 = apply_annotations(art, ann, cfg, replace_auto=False)
    assert sum(len(f) for f in out2["art"]["per_frame_filaments"]) == 2


def test_prelabel_cli_end_to_end(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from scipy.ndimage import gaussian_filter
    from pilitrack.annotate import prelabel_main, load_annotations
    H = W = 120
    yy, xx = np.ogrid[:H, :W]
    cell = ((yy - 55) ** 2 + (xx - 55) ** 2) <= 9 ** 2
    frames = []
    for _ in range(4):
        f = np.zeros((H, W), np.float32)
        f[cell] += 4000.0
        for k in range(22):
            f[55, 64 + k] += 300.0
        frames.append(gaussian_filter(f, 1.1) + 60.0)
    mov = np.clip(np.stack(frames), 0, None).astype(np.uint16)
    movie = tmp_path / "m.ome.tif"
    tifffile.imwrite(str(movie), mov, metadata={
        "axes": "TYX", "PhysicalSizeX": 0.0433, "PhysicalSizeXUnit": "µm",
        "TimeIncrement": 0.4})
    out = tmp_path / "pre.json"
    prelabel_main([str(movie), "--out", str(out)])
    assert out.exists()
    loaded, _ = load_annotations(out)
    assert len(loaded.manual_pili) >= 1            # detected the pilus as a label
    assert loaded.meta.get("source") == "auto-detection"
