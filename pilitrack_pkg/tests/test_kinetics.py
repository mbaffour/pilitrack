import numpy as np
import pytest

from pilitrack.config import AcquisitionConfig
from pilitrack.kinetics import segment_trace, summarize_pilus
from pilitrack.synth import make_kinetic_trace


@pytest.mark.parametrize("v_ext,v_ret", [(500, 500), (700, 300), (300, 900)])
def test_recovers_velocities_clean(v_ext, v_ret):
    cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=65.0)
    tr = make_kinetic_trace(cfg, v_ext_nm_s=v_ext, v_ret_nm_s=v_ret,
                            max_length_nm=2000, n_cycles=3)
    s = summarize_pilus(tr.length_nm, cfg)
    assert s["mean_extension_velocity_nm_s"] == pytest.approx(v_ext, rel=0.15)
    assert s["mean_retraction_velocity_nm_s"] == pytest.approx(v_ret, rel=0.15)


def test_counts_events():
    cfg = AcquisitionConfig(dt_s=0.5)
    tr = make_kinetic_trace(cfg, n_cycles=3, max_length_nm=2000)
    s = summarize_pilus(tr.length_nm, cfg)
    assert s["n_extension_events"] == 3
    assert s["n_retraction_events"] == 3


def test_robust_to_noise():
    cfg = AcquisitionConfig(dt_s=0.5)
    tr = make_kinetic_trace(cfg, v_ext_nm_s=500, v_ret_nm_s=500,
                            max_length_nm=2000, n_cycles=3, noise_nm=40.0)
    s = summarize_pilus(tr.length_nm, cfg)
    assert s["mean_extension_velocity_nm_s"] == pytest.approx(500, rel=0.25)
    assert s["mean_retraction_velocity_nm_s"] == pytest.approx(500, rel=0.25)


def test_max_length():
    cfg = AcquisitionConfig(dt_s=0.5)
    tr = make_kinetic_trace(cfg, max_length_nm=1500, n_cycles=2)
    s = summarize_pilus(tr.length_nm, cfg)
    assert s["max_length_nm"] == pytest.approx(1500, abs=60)


def test_flat_trace_no_events():
    cfg = AcquisitionConfig(dt_s=0.5)
    flat = np.full(40, 800.0)
    phases = segment_trace(flat, cfg)
    assert all(p.kind == "dwell" for p in phases)
