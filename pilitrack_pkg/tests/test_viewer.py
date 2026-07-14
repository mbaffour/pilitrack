import numpy as np

from pilitrack.config import AcquisitionConfig
from pilitrack.pipeline import detect_and_link
from pilitrack.synth import make_movie
from pilitrack.viewer import (
    build_skeleton_label_stack, build_tracks_array, curated_summary)


def _artifacts():
    cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=65.0, min_pilus_length_nm=200.0)
    mov = make_movie(cfg, n_cells=5, shape=(160, 160), piliated_fraction=0.8,
                     v_ext_nm_s=500, v_ret_nm_s=500, max_length_nm=2000,
                     n_cycles=2, seed=5)
    art = detect_and_link(mov.stack, mov.cell_stack, cfg)
    return cfg, mov, art


def test_skeleton_label_stack_shape_and_ids():
    cfg, mov, art = _artifacts()
    stack = build_skeleton_label_stack(art["per_frame_filaments"],
                                       art["shape"], art["n_frames"])
    assert stack.shape == (art["n_frames"],) + tuple(art["shape"])
    assert stack.max() >= 1                      # at least one track painted


def test_tracks_array_format():
    cfg, mov, art = _artifacts()
    arr = build_tracks_array(art["tracks"])
    assert arr.ndim == 2 and arr.shape[1] == 4   # [track_id, t, y, x]
    if arr.shape[0]:
        assert arr[:, 1].min() >= 0              # frame indices non-negative


def test_culling_changes_summary():
    cfg, mov, art = _artifacts()
    full = curated_summary(art, cfg, remove_ids=set())
    if art["tracks"]:
        drop = {art["tracks"][0].track_id}
        culled = curated_summary(art, cfg, remove_ids=drop)
        # removing a track can only keep or reduce the pilus count
        assert len(culled["pilus"]) <= len(full["pilus"])
