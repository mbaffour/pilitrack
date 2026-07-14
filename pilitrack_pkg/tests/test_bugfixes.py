"""Regression tests for bugs surfaced by the definitive-run audit:
negative extension/retraction velocities, and QC missing impossible outputs."""
import numpy as np

from pilitrack.config import AcquisitionConfig
from pilitrack.kinetics import summarize_pilus, segment_trace
from pilitrack.pipeline import summarize
from pilitrack.analyze import build_config
from pilitrack.qc import qc_flags


class _FakeTrack:
    def __init__(self, frames, length_px, cell_id=1, track_id=1):
        self.track_id = track_id
        self.cell_id = cell_id
        self.frames = frames
        self.length_px = length_px

    def length_series(self, n):
        out = np.full(n, np.nan)
        for f, L in zip(self.frames, self.length_px):
            out[f] = L
        return out


def test_gap_bridged_track_does_not_inflate_velocity():
    """A track with a missing interior frame must not have its velocity doubled:
    the gap frame is interpolated so the dt axis stays uniform (pipeline.py)."""
    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=100.0)  # 1 px/frame = 250 nm/s
    dense = _FakeTrack([0, 1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15, 16])
    gapped = _FakeTrack([0, 1, 2, 4, 5, 6], [10, 11, 12, 14, 15, 16])  # frame 3 missing
    cells = [np.zeros((5, 5), int)] * 7
    v_dense = summarize([dense], cells, cfg, 7)["pilus"].iloc[0]["mean_extension_velocity_nm_s"]
    v_gap = summarize([gapped], cells, cfg, 7)["pilus"].iloc[0]["mean_extension_velocity_nm_s"]
    assert abs(v_gap - v_dense) < 30            # essentially identical, not ~2x
    assert v_gap < 350                          # would be ~500 if the gap collapsed


def test_build_config_ignores_loader_only_overrides():
    """A loader-only key (channel_names) must not reach AcquisitionConfig(**)."""
    meta = dict(pixel_size_nm=65.0, dt_s=0.4, duration_s=5.0)
    cfg, _ = build_config(meta, overrides={"channel_names": ["a", "b"],
                                           "pixel_size_nm": 50.0})
    assert cfg.pixel_size_nm == 50.0            # real config override still applied


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
