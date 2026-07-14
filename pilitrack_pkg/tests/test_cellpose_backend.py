"""Cellpose backend seam — verified with a fake model (no torch needed)."""
import numpy as np

import pilitrack.backends.cellpose_backend as cb
from pilitrack.pipeline import analyze_movie
from pilitrack.config import AcquisitionConfig
from pilitrack.synth import make_movie


class _FakeCellpose:
    def eval(self, img, **kw):
        masks = np.zeros(np.asarray(img).shape, dtype=np.int32)
        masks[2:6, 2:6] = 1          # one "cell"
        return masks, None, None     # (masks, flows, styles)


def test_make_cellpose_segmenter_contract(monkeypatch):
    monkeypatch.setattr(cb, "_load_model", lambda model_type, gpu: _FakeCellpose())
    seg = cb.make_cellpose_segmenter(model_type="cyto3")
    labels = seg(np.zeros((8, 8), np.float32), None)
    assert labels.dtype == int or np.issubdtype(labels.dtype, np.integer)
    assert int(labels.max()) == 1


def test_segment_stack_cellpose(monkeypatch):
    monkeypatch.setattr(cb, "_load_model", lambda model_type, gpu: _FakeCellpose())
    stack = np.zeros((3, 8, 8), np.float32)
    out = cb.segment_stack_cellpose(stack, AcquisitionConfig())
    assert out.shape == (3, 8, 8)
    assert (out.max(axis=(1, 2)) == 1).all()


def test_cellpose_segfn_drops_into_pipeline(monkeypatch):
    # a Cellpose-style segment_fn feeds analyze_movie exactly like any backend
    monkeypatch.setattr(cb, "_load_model", lambda model_type, gpu: _FakeCellpose())
    cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=65.0, min_pilus_length_nm=200.0)
    mov = make_movie(cfg, n_cells=4, shape=(120, 120), seed=2)
    seg = cb.make_cellpose_segmenter()
    res = analyze_movie(mov.stack, mov.cell_stack, cfg, segment_fn=seg)
    assert "percent_piliated" in res["population"]
