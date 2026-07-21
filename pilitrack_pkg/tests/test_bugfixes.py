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


def test_detect_pili_robust_to_a_hot_pixel():
    """A single saturated pixel must not blank the whole frame (percentile
    normalization, not global-max)."""
    import numpy as np
    from skimage.draw import line
    from pilitrack.detect import detect_pili, filament_components
    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=65.0)
    frame = np.full((64, 64), 100.0)
    rr, cc = line(10, 8, 40, 30)               # a faint diagonal pilus
    frame[rr, cc] = 400.0
    clean = detect_pili(frame, cfg)
    frame[5, 55] = 60000.0                      # one hot/saturated pixel
    withhot = detect_pili(frame, cfg)
    _, n_clean = filament_components(clean)
    _, n_hot = filament_components(withhot)
    assert n_clean >= 1                         # pilus found without the hot pixel
    assert n_hot >= 1                           # and still found WITH it (was 0 before)


def test_geodesic_length_corrected_for_oblique_lines():
    """The corrected estimator must be within ~3% of true length on an oblique
    straight skeleton (the naive 1/sqrt2 sum overestimated by ~8%)."""
    import numpy as np
    from skimage.draw import line
    from skimage.morphology import skeletonize
    from pilitrack.measure import _geodesic_length_px
    # ~22 deg line, the worst case for the naive estimator
    y1, x1 = 20, 50
    img = np.zeros((y1 + 3, x1 + 3), bool)
    rr, cc = line(0, 0, y1, x1)
    img[rr, cc] = True
    L = _geodesic_length_px(np.argwhere(skeletonize(img)))
    true = np.hypot(y1, x1)
    assert abs(L - true) / true < 0.03


def test_config_rejects_zero_dt_and_pixel_size():
    import pytest
    with pytest.raises(ValueError):
        AcquisitionConfig(dt_s=0.0)
    with pytest.raises(ValueError):
        AcquisitionConfig(pixel_size_nm=0.0)


def test_saturation_level_detects_sub_full_range_clipping():
    """A 12-bit camera in a uint16 container clips at 4095, not 65535."""
    import numpy as np
    from pilitrack.qc import _saturation_level
    stack = np.full((3, 16, 16), 500, np.uint16)
    stack[:, :8, :8] = 4095                     # a clipped block, not 65535
    lvl = _saturation_level(stack)
    assert lvl == 4095.0
    assert np.mean(stack >= lvl) > 0.2          # saturation is now detectable


def test_ml_scaled_sigmas_matches_physical_scale():
    from pilitrack.ml import _scaled_sigmas
    model = {"sigmas": (1.0, 2.0, 4.0), "pixel_size_nm": 130.0}
    # target movie at 65 nm/px -> half the pixel size -> double the sigmas
    assert _scaled_sigmas(model, 65.0) == (2.0, 4.0, 8.0)
    assert _scaled_sigmas(model, 130.0) == (1.0, 2.0, 4.0)   # same scale -> unchanged
    assert _scaled_sigmas({"sigmas": (1.0, 2.0)}, 65.0) == (1.0, 2.0)  # no train px


def test_link_tracks_honors_manual_track_id():
    """Two filaments hand-tagged with the same track_id must group into ONE
    track even when their bases are farther apart than max_base_jump_px."""
    import numpy as np
    from pilitrack.measure import Filament
    from pilitrack.track import link_tracks
    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=65.0, max_base_jump_px=3.0)
    co = np.array([[0, 0]])
    f0 = Filament(1, 5.0, (10.0, 10.0), (10.0, 15.0), co, cell_id=1, track_id=7)
    f1 = Filament(2, 6.0, (40.0, 40.0), (40.0, 46.0), co, cell_id=1, track_id=7)  # far base
    tracks = link_tracks([[f0], [f1]], cfg)
    grouped = [tr for tr in tracks if tr.track_id == 7]
    assert len(grouped) == 1 and grouped[0].frames == [0, 1]


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


def test_qc_organism_envelope_widens_ranges():
    """A 20 um / 2200 nm/s pilus is implausible for P. aeruginosa but normal for
    Neisseria — the organism envelope must change what QC flags."""
    from pilitrack.qc import envelope_for, qc_flags
    base = {"saturated_fraction": 0.0, "n_cells": 1, "detection_rate": 1.0,
            "median_extension_velocity_nm_s": 2200.0,
            "median_retraction_velocity_nm_s": 300.0,
            "median_max_length_nm": 20000.0, "n_implausible_length": 0,
            "n_implausible_velocity": 0, "n_negative_velocity": 0}
    pa = dict(base)                                     # defaults -> P. aeruginosa
    ng_env = envelope_for("N. gonorrhoeae")
    ng = dict(base, sane_velocity_nm_s=ng_env["velocity_nm_s"],
              sane_maxlen_nm=ng_env["maxlen_nm"])
    assert any("length" in f for f in qc_flags(pa))     # flagged for P. aeruginosa
    assert not any("length" in f for f in qc_flags(ng)) # fine for Neisseria


def test_crossing_pili_are_split_into_two():
    """Two crossing pili (an X) must resolve into TWO through-filaments of the
    right length, not one summed blob — and a single line stays one filament."""
    from skimage.draw import line
    from skimage.morphology import skeletonize
    from pilitrack.measure import extract_filaments
    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=65.0)
    img = np.zeros((70, 70), bool)
    for (y0, x0, y1, x1) in [(5, 5, 50, 50), (5, 50, 50, 5)]:
        rr, cc = line(y0, x0, y1, x1)
        img[rr, cc] = True
    fils = extract_filaments(skeletonize(img), cfg)
    assert len(fils) == 2
    for f in fils:
        assert 55 < f.length_px < 75          # ~one diagonal each (~63 px)
    one = np.zeros((70, 70), bool)
    rr, cc = line(5, 5, 50, 50)
    one[rr, cc] = True
    assert len(extract_filaments(skeletonize(one), cfg)) == 1


def test_phase_table_surfaces_extend_dwell_retract_events():
    """events.csv must contain the per-event phases (kind, dwell, velocity) for a
    pilus that extends then retracts, consistent with the length series."""
    from pilitrack.analyze import phase_table
    from pilitrack.synth import make_kinetic_trace

    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=65.0, max_pilus_length_nm=20000.0)
    truth = make_kinetic_trace(cfg, v_ext_nm_s=500, v_ret_nm_s=500,
                               max_length_nm=2000, n_cycles=1)
    length_px = truth.length_nm / cfg.pixel_size_nm

    class _Tr:                       # minimal track exposing length_series
        track_id, cell_id = 7, 3
        def length_series(self, n):
            return length_px

    df = phase_table([_Tr()], cfg, length_px.size)
    assert set(df.columns) >= {"track_id", "cell_id", "kind", "duration_s",
                               "velocity_nm_s", "delta_length_nm"}
    kinds = set(df["kind"])
    assert "extension" in kinds and "retraction" in kinds
    assert (df["track_id"] == 7).all() and (df["cell_id"] == 3).all()
    # an extension event has positive velocity and positive length change
    ext = df[df["kind"] == "extension"].iloc[0]
    assert ext["velocity_nm_s"] > 0 and ext["delta_length_nm"] > 0


def test_hysteresis_mask_recovers_faint_tail_but_not_noise():
    """Opt-in double-threshold keeps a faint tail connected to a bright core,
    rejects isolated faint noise, and matches a single cut when disabled."""
    from pilitrack.detect import hysteresis_mask
    resp = np.zeros((5, 12))
    resp[2, 0:4] = 0.9          # bright core (> high=0.3)
    resp[2, 4:9] = 0.22         # faint tail (between low=0.15 and high=0.3)
    on = AcquisitionConfig(dt_s=0.4, pixel_size_nm=65.0,
                           detect_threshold=0.30, hysteresis_low_frac=0.5)
    off = AcquisitionConfig(dt_s=0.4, pixel_size_nm=65.0,
                            detect_threshold=0.30, hysteresis_low_frac=1.0)
    assert hysteresis_mask(resp, on).sum() == 9      # core(4) + connected tail(5)
    assert hysteresis_mask(resp, off).sum() == 4     # core only
    noise = np.zeros((5, 12))
    noise[1, 1] = noise[3, 8] = 0.22                 # faint, no bright core
    assert hysteresis_mask(noise, on).sum() == 0


def test_orientation_continuity_keeps_identity_through_a_crossing():
    """Where two pili cross, base distance alone can prefer the SWAP; the
    orientation-continuity term must keep each pilus with its own track."""
    from pilitrack.track import link_tracks
    from pilitrack.measure import Filament

    def fil(base, tip, L, cid=1):
        f = Filament(1, L, base, tip, np.array([base, tip]))
        f.cell_id, f.track_id = cid, None
        return f

    def frames():                      # fresh filaments (track_id must be unset)
        return [[fil((50, 50), (50, 60), 10.0), fil((50, 52), (50, 42), 10.0)],
                [fil((50, 51.5), (50, 61), 10.0), fil((50, 50.5), (50, 40), 10.0)]]

    def tip_dirs(weight):
        cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=65.0, max_base_jump_px=5.0,
                                link_orientation_weight_px=weight, linker="lap")
        trs = [t for t in link_tracks(frames(), cfg) if len(t.tips) > 1]
        return [{"+x" if tp[1] > 50.8 else "-x" for tp in tr.tips} for tr in trs]

    assert any(len(d) > 1 for d in tip_dirs(0.0))       # base-only: identities swap
    assert all(len(d) == 1 for d in tip_dirs(2.0))      # with continuity: kept


def test_compact_labels_downcasts_losslessly():
    """Per-frame label images must use the smallest lossless dtype — int64 labels
    for a 2k x 2k x 70 movie cost ~2 GB on their own."""
    from pilitrack.pipeline import _compact_labels
    for mx, want in ((3, np.uint8), (300, np.uint16), (70000, np.uint32)):
        a = np.zeros((4, 4), np.int64)
        a[0, 0] = mx
        out = _compact_labels(a)
        assert out.dtype == want
        assert int(out.max()) == mx            # lossless
    assert _compact_labels(np.zeros((2, 2), bool)).dtype == bool
    assert _compact_labels(np.array([[-1, 2]])).dtype == np.int64   # left alone


def test_one_bad_frame_does_not_abort_the_movie():
    """A frame that raises must be isolated and REPORTED, not abort a long run."""
    from pilitrack.pipeline import detect_and_link
    from pilitrack.detect import detect_pili

    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=65.0)
    stack = (np.random.default_rng(0).random((5, 48, 48)) * 500).astype(np.uint16)
    seen = []

    def flaky(frame, c):
        if len(seen) == 2:
            seen.append(1)
            raise RuntimeError("simulated bad frame")
        seen.append(1)
        return detect_pili(frame, c)

    ticks = []
    art = detect_and_link(stack, stack, cfg, detect_fn=flaky,
                          progress=lambda d, t: ticks.append((d, t)))
    assert art["n_frames"] == 5                       # movie completed
    assert len(art["failed_frames"]) == 1
    assert art["failed_frames"][0]["frame"] == 2
    assert "simulated bad frame" in art["failed_frames"][0]["error"]
    assert ticks[-1] == (5, 5)                        # progress reported
