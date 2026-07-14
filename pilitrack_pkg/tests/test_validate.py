"""Validation of automated detection vs hand-labeled ground truth."""
import numpy as np
import pytest

from pilitrack.config import AcquisitionConfig
from pilitrack.measure import Filament
from pilitrack.annotate import Annotations, ManualPilus, rasterize_polyline
from pilitrack.validate import detection_metrics, length_agreement, validate

CFG = AcquisitionConfig(pixel_size_nm=50.0, dt_s=0.5)
SHAPE = (40, 40)


def _fil(label, p0, p1, length_px=None):
    coords = rasterize_polyline([p0, p1], SHAPE)
    if length_px is None:
        length_px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
    return Filament(label, length_px, tuple(p0), tuple(p1), coords)


def _art(filaments):
    return {"per_frame_filaments": [filaments],
            "per_frame_cell_labels": [np.zeros(SHAPE, int)],
            "shape": SHAPE, "n_frames": 1, "tracks": []}


def test_tp_fp_fn_counts():
    # auto A matches a truth; auto B is a false positive; a 2nd truth is missed
    A = _fil(1, [10, 5], [10, 25])
    B = _fil(2, [30, 5], [30, 15])
    art = _art([A, B])
    truth = Annotations(manual_pili=[
        ManualPilus(0, [[10, 5], [10, 25]]),   # matches A
        ManualPilus(0, [[20, 5], [20, 25]]),   # missed
    ])
    m = detection_metrics(art, truth, CFG, frames=[0], tol_px=2.0, overlap_frac=0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5 and m["f1"] == 0.5
    assert len(m["matched"]) == 1


def test_perfect_detection():
    A = _fil(1, [10, 5], [10, 25])
    art = _art([A])
    truth = Annotations(manual_pili=[ManualPilus(0, [[10, 5], [10, 25]])])
    m = detection_metrics(art, truth, CFG, frames=[0], tol_px=2.0)
    assert m["precision"] == 1.0 and m["recall"] == 1.0 and m["f1"] == 1.0


def test_all_missed_when_no_auto():
    art = _art([])
    truth = Annotations(manual_pili=[ManualPilus(0, [[10, 5], [10, 25]])])
    m = detection_metrics(art, truth, CFG, frames=[0], tol_px=2.0)
    assert m["tp"] == 0 and m["fn"] == 1 and m["recall"] == 0.0


def test_all_fp_when_no_truth():
    A = _fil(1, [10, 5], [10, 25])
    m = detection_metrics(_art([A]), Annotations(manual_pili=[]), CFG,
                          frames=[0], tol_px=2.0)
    assert m["fp"] == 1 and m["tp"] == 0


def test_length_agreement_bias():
    # auto measures 22 px where the hand trace is 20 px -> +2 px * 50 nm = +100 nm
    A = _fil(1, [10, 5], [10, 25], length_px=22.0)
    art = _art([A])
    truth = Annotations(manual_pili=[ManualPilus(0, [[10, 5], [10, 25]])])
    m = detection_metrics(art, truth, CFG, frames=[0], tol_px=2.0)
    la = length_agreement(m["matched"], CFG)
    assert la["n"] == 1
    assert la["bias_nm"] == pytest.approx(100.0)
    assert la["mae_nm"] == pytest.approx(100.0)


def test_validate_combines():
    A = _fil(1, [10, 5], [10, 25])
    art = _art([A])
    truth = Annotations(manual_pili=[ManualPilus(0, [[10, 5], [10, 25]])])
    rep = validate(art, truth, CFG, frames=[0], tol_px=2.0)
    assert rep["detection"]["f1"] == 1.0
    assert rep["n_matched"] == 1
    assert "length_agreement" in rep


def test_validate_cli_end_to_end(tmp_path):
    """pilitrack-validate: real movie file + labels -> a scored report on disk."""
    tifffile = pytest.importorskip("tifffile")
    from scipy.ndimage import gaussian_filter
    from pilitrack.annotate import save_annotations
    from pilitrack.validate import main

    # synthetic movie: a bright cell with a pilus the detector should find
    H = W = 120
    yy, xx = np.ogrid[:H, :W]
    cell = ((yy - 55) ** 2 + (xx - 55) ** 2) <= 9 ** 2
    frames = []
    for _ in range(3):
        f = np.zeros((H, W), np.float32)
        f[cell] += 4000.0
        for k in range(20):
            f[55, 64 + k] += 500.0
        frames.append(gaussian_filter(f, 1.1) + 60.0)
    mov = np.clip(np.stack(frames), 0, None).astype(np.uint16)
    movie = tmp_path / "m.ome.tif"
    tifffile.imwrite(str(movie), mov, metadata={
        "axes": "TYX", "PhysicalSizeX": 0.0433, "PhysicalSizeXUnit": "µm",
        "TimeIncrement": 0.4})

    # ground truth: trace the pilus on frames 0 and 1
    ann = Annotations(manual_pili=[ManualPilus(0, [[55, 64], [55, 82]]),
                                   ManualPilus(1, [[55, 64], [55, 82]])])
    labels = tmp_path / "ann.json"
    save_annotations(ann, labels)

    out = tmp_path / "val.json"
    report = main([str(movie), "--labels", str(labels), "--frames", "0", "1",
                   "--out", str(out)])
    assert out.exists()
    det = report["detection"]
    assert {"precision", "recall", "f1", "tp", "fp", "fn"} <= set(det)
    assert det["recall"] > 0     # the traced pilus is detected
