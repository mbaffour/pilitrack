"""Regression tests for bugs surfaced by the definitive-run audit:
negative extension/retraction velocities, and QC missing impossible outputs."""
import numpy as np

from pilitrack.config import AcquisitionConfig
from pilitrack.kinetics import summarize_pilus, segment_trace
from pilitrack.qc import qc_flags


def test_velocities_never_negative_on_noisy_traces():
    """summarize_pilus must never report a negative extension/retraction speed,
    even on noisy random-walk length traces (the merge/reclassify fix)."""
    cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=65.0)
    rng = np.random.default_rng(1)
    for _ in range(120):
        trace = np.clip(np.cumsum(rng.normal(0, 130, 40)) + 600.0, 0, None)
        s = summarize_pilus(trace, cfg)
        for k in ("mean_extension_velocity_nm_s", "mean_retraction_velocity_nm_s"):
            v = s[k]
            assert (v != v) or v >= 0, (k, v)


def test_phase_kind_matches_velocity_sign():
    """After segmentation, an 'extension' phase has a non-negative slope and a
    'retraction' phase a non-positive slope (no mislabelled merged phases)."""
    cfg = AcquisitionConfig(dt_s=0.5, pixel_size_nm=65.0)
    rng = np.random.default_rng(4)
    eps = cfg.velocity_sign_eps_nm_s
    for _ in range(80):
        trace = np.clip(np.cumsum(rng.normal(0, 140, 45)) + 700.0, 0, None)
        for p in segment_trace(trace, cfg):
            if p.kind == "extension":
                assert p.velocity_nm_s >= -eps
            elif p.kind == "retraction":
                assert p.velocity_nm_s <= eps


def test_qc_flags_impossible_outputs():
    m = {"saturated_fraction": 0.0, "n_cells": 5, "detection_rate": 1.0,
         "median_extension_velocity_nm_s": 300.0,
         "median_retraction_velocity_nm_s": 300.0, "median_max_length_nm": 1000.0,
         "n_implausible_length": 3, "n_implausible_velocity": 2,
         "n_negative_velocity": 1}
    flags = qc_flags(m)
    assert any("longer than" in f for f in flags)
    assert any("faster than" in f for f in flags)
    assert any("negative velocity" in f for f in flags)


def test_qc_no_flags_when_clean():
    m = {"saturated_fraction": 0.0, "n_cells": 5, "detection_rate": 1.0,
         "median_extension_velocity_nm_s": 300.0,
         "median_retraction_velocity_nm_s": 300.0, "median_max_length_nm": 1000.0,
         "n_implausible_length": 0, "n_implausible_velocity": 0,
         "n_negative_velocity": 0}
    assert qc_flags(m) == []
