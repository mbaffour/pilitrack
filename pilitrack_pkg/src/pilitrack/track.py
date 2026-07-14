"""Link per-frame filaments into per-pilus tracks.

Greedy nearest-base assignment within a jump radius, with short gap bridging.
A pilus keeps its identity while its base stays put; the tip moves as it
extends/retracts. This is the identity model that makes length(t) meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AcquisitionConfig


@dataclass
class Track:
    track_id: int
    cell_id: int | None
    frames: list = field(default_factory=list)       # frame indices
    length_px: list = field(default_factory=list)     # length at each frame
    tips: list = field(default_factory=list)          # (y, x) tip per frame
    base_yx: tuple = (0.0, 0.0)

    def length_series(self, n_frames: int) -> np.ndarray:
        """Dense length(t) over the window, NaN where the pilus is absent."""
        out = np.full(n_frames, np.nan)
        for f, L in zip(self.frames, self.length_px):
            out[f] = L
        return out


def link_tracks(per_frame_filaments: list, cfg: AcquisitionConfig) -> list[Track]:
    """Link per-frame filaments into tracks.

    ``cfg.linker`` selects the frame-to-frame assignment: ``"greedy"`` (default)
    or ``"lap"`` — a globally optimal one-to-one Hungarian assignment that is
    more robust when several pili are close together or cross (greedy can grab
    the wrong nearby base first).
    """
    if getattr(cfg, "linker", "greedy") == "lap":
        return _link_lap(per_frame_filaments, cfg)
    return _link_greedy(per_frame_filaments, cfg)


def _link_greedy(per_frame_filaments: list, cfg: AcquisitionConfig) -> list[Track]:
    """``per_frame_filaments``: list (len T) of lists of Filament objects."""
    tracks: list[Track] = []
    active: list[Track] = []
    next_id = 0

    for t, filaments in enumerate(per_frame_filaments):
        used = set()
        # try to extend active tracks
        for tr in active:
            best, best_d = None, cfg.max_base_jump_px + 1e-9
            for j, f in enumerate(filaments):
                if j in used:
                    continue
                if tr.cell_id is not None and f.cell_id != tr.cell_id:
                    continue
                d = float(np.hypot(f.base_yx[0] - tr.base_yx[0],
                                   f.base_yx[1] - tr.base_yx[1]))
                if d < best_d:
                    best, best_d, best_j = f, d, j
            if best is not None:
                tr.frames.append(t)
                tr.length_px.append(best.length_px)
                tr.tips.append(best.tip_yx)
                tr.base_yx = best.base_yx
                best.track_id = tr.track_id
                used.add(best_j)

        # start new tracks for unmatched filaments
        for j, f in enumerate(filaments):
            if j in used:
                continue
            tr = Track(next_id, f.cell_id, [t], [f.length_px], [f.tip_yx], f.base_yx)
            f.track_id = next_id
            next_id += 1
            tracks.append(tr)
            active.append(tr)

        # retire tracks not seen within the gap tolerance
        active = [tr for tr in active if (t - tr.frames[-1]) <= cfg.max_gap_frames]

    return tracks


def _link_lap(per_frame_filaments: list, cfg: AcquisitionConfig) -> list[Track]:
    """Optimal (Hungarian) base-to-base assignment per frame.

    Same identity model and constraints as greedy (same cell, within
    ``max_base_jump_px``, gap bridging) but minimizes the *total* base
    displacement over all matches at once, so nearby/crossing pili are less
    likely to steal each other's identity.
    """
    from scipy.optimize import linear_sum_assignment

    tracks: list[Track] = []
    active: list[Track] = []
    next_id = 0
    R = cfg.max_base_jump_px
    BIG = 1e6

    for t, filaments in enumerate(per_frame_filaments):
        used: set = set()
        if active and filaments:
            cost = np.full((len(active), len(filaments)), BIG)
            for i, tr in enumerate(active):
                for j, f in enumerate(filaments):
                    if tr.cell_id is not None and f.cell_id != tr.cell_id:
                        continue
                    d = float(np.hypot(f.base_yx[0] - tr.base_yx[0],
                                       f.base_yx[1] - tr.base_yx[1]))
                    if d <= R:
                        cost[i, j] = d
            rows, cols = linear_sum_assignment(cost)
            for i, j in zip(rows, cols):
                if cost[i, j] >= BIG:
                    continue
                tr, f = active[i], filaments[j]
                tr.frames.append(t)
                tr.length_px.append(f.length_px)
                tr.tips.append(f.tip_yx)
                tr.base_yx = f.base_yx
                f.track_id = tr.track_id
                used.add(j)

        for j, f in enumerate(filaments):
            if j in used:
                continue
            tr = Track(next_id, f.cell_id, [t], [f.length_px], [f.tip_yx], f.base_yx)
            f.track_id = next_id
            next_id += 1
            tracks.append(tr)
            active.append(tr)

        active = [tr for tr in active if (t - tr.frames[-1]) <= cfg.max_gap_frames]

    return tracks
