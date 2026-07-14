import numpy as np

from pilitrack.config import AcquisitionConfig
from pilitrack.detect import skeletonize_probability, segment_cells
from pilitrack.pipeline import analyze_movie
from pilitrack.synth import make_movie


def _movie():
    cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=65.0, min_pilus_length_nm=200.0)
    mov = make_movie(cfg, n_cells=5, shape=(160, 160), piliated_fraction=0.8,
                     v_ext_nm_s=500, v_ret_nm_s=500, max_length_nm=2000,
                     n_cycles=2, seed=3)
    return cfg, mov


def test_skeletonize_probability():
    cfg = AcquisitionConfig()
    prob = np.zeros((40, 40))
    prob[20, 5:30] = 0.9          # a bright horizontal filament
    skel = skeletonize_probability(prob, cfg)
    assert skel.sum() > 0


def test_custom_segment_and_detect_callables():
    cfg, mov = _movie()
    calls = {"seg": 0, "det": 0}

    def seg_fn(frame, c):
        calls["seg"] += 1
        return segment_cells(frame, c)

    def det_fn(frame, c):
        calls["det"] += 1
        from pilitrack.detect import detect_pili
        return detect_pili(frame, c)

    res = analyze_movie(mov.stack, mov.cell_stack, cfg,
                        segment_fn=seg_fn, detect_fn=det_fn)
    assert calls["seg"] == mov.stack.shape[0]   # called once per frame
    assert calls["det"] == mov.stack.shape[0]
    assert res["population"]["n_cells"] >= 3


def test_precomputed_stacks_path():
    """Simulates Omnipose label stack + ilastik probability stack inputs."""
    cfg, mov = _movie()
    T = mov.stack.shape[0]

    # stand-in for an Omnipose label stack: Otsu labels per frame
    cell_labels = np.stack([segment_cells(mov.cell_stack[t], cfg) for t in range(T)])

    # stand-in for an ilastik probability stack: normalise the pilus channel
    prob = mov.stack - mov.stack.min()
    prob = prob / (prob.max() + 1e-9)

    res = analyze_movie(mov.stack, mov.cell_stack, cfg,
                        cell_label_stack=cell_labels, pilus_prob_stack=prob)
    assert res["population"]["n_cells"] >= 3
    assert "percent_piliated" in res["population"]
