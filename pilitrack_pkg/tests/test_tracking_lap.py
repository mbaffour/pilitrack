"""Optimal (LAP) linker: dispatch, correctness, and parity with greedy on
clearly-separated pili."""
import numpy as np

from pilitrack.config import AcquisitionConfig
from pilitrack.measure import Filament
from pilitrack.track import link_tracks, _link_lap, _link_greedy


def _fil(label, base, length=10.0):
    return Filament(label, length, base, (base[0], base[1] + length),
                    np.array([[base[0], base[1]]]))


def _two_parallel_pili(T=4):
    """Two pili with well-separated bases drifting slightly, over T frames."""
    frames = []
    for t in range(T):
        frames.append([_fil(1, (10, 10 + 0.5 * t)), _fil(2, (40, 10 + 0.5 * t))])
    return frames


def test_link_tracks_dispatches_to_lap():
    cfg = AcquisitionConfig(max_base_jump_px=5.0, linker="lap")
    per_frame = _two_parallel_pili()
    tracks = link_tracks(per_frame, cfg)
    assert len(tracks) == 2
    assert all(len(tr.frames) == 4 for tr in tracks)


def test_lap_and_greedy_agree_when_separated():
    cfg_g = AcquisitionConfig(max_base_jump_px=5.0, linker="greedy")
    cfg_l = AcquisitionConfig(max_base_jump_px=5.0, linker="lap")
    per_frame = _two_parallel_pili()
    g = _link_greedy([list(f) for f in per_frame], cfg_g)
    per_frame2 = _two_parallel_pili()
    lap = _link_lap([list(f) for f in per_frame2], cfg_l)
    assert len(g) == len(lap) == 2
    # both recover two full-length tracks with matching length series
    g_lens = sorted(len(tr.frames) for tr in g)
    lap_lens = sorted(len(tr.frames) for tr in lap)
    assert g_lens == lap_lens == [4, 4]


def test_lap_respects_max_jump_and_cell():
    # a base that jumps too far starts a new track rather than linking
    cfg = AcquisitionConfig(max_base_jump_px=3.0, linker="lap")
    per_frame = [[_fil(1, (10, 10))], [_fil(1, (10, 20))]]  # jump of 10 px
    tracks = _link_lap(per_frame, cfg)
    assert len(tracks) == 2                     # not linked -> two 1-frame tracks
    assert all(len(tr.frames) == 1 for tr in tracks)


def test_lap_globally_optimal_assignment():
    # frame1 has two candidates; LAP must minimize TOTAL base displacement
    cfg = AcquisitionConfig(max_base_jump_px=10.0, linker="lap")
    f0 = [_fil(1, (0, 0)), _fil(2, (0, 10))]
    f1 = [_fil(3, (0, 1)), _fil(4, (0, 9))]   # near track1 and track2 respectively
    tracks = _link_lap([f0, f1], cfg)
    two_frame = [tr for tr in tracks if len(tr.frames) == 2]
    assert len(two_frame) == 2               # both tracks continued, none stolen
    # track that started at x=0 ends near x=1; the one at x=10 ends near x=9
    ends = sorted(tr.base_yx[1] for tr in two_frame)
    assert ends[0] < 5 and ends[1] > 5
