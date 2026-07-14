import numpy as np

from pilitrack.config import AcquisitionConfig
from pilitrack.pipeline import analyze_movie
from pilitrack.synth import make_movie


def test_end_to_end_runs_and_finds_pili():
    cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=65.0,
                            min_pilus_length_nm=200.0)
    mov = make_movie(cfg, n_cells=5, shape=(160, 160),
                     piliated_fraction=0.8, v_ext_nm_s=500, v_ret_nm_s=500,
                     max_length_nm=2000, n_cycles=2, seed=1)
    res = analyze_movie(mov.stack, mov.cell_stack, cfg)

    # cells were detected
    assert res["population"]["n_cells"] >= 4
    # at least some pili tracked with a plausible max length
    assert not res["pilus"].empty
    assert res["pilus"]["max_length_nm"].max() > 800
    # a nonzero piliation percentage was computed
    assert res["population"]["percent_piliated"] > 0
