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

    Filaments that already carry a ``track_id`` (a hand annotation grouping the
    same pilus across frames, e.g. across a large base excursion the automatic
    linker would split) are honored: they are grouped directly into their manual
    track and only the remaining filaments go through the automatic linker.
    """
    manual: dict[int, list] = {}
    auto = [[] for _ in per_frame_filaments]
    for t, filaments in enumerate(per_frame_filaments):
        for f in filaments:
            mid = getattr(f, "track_id", None)
            if mid is not None:
                manual.setdefault(int(mid), []).append((t, f))
            else:
                auto[t].append(f)

    linker = _link_lap if getattr(cfg, "linker", "greedy") == "lap" else _link_greedy
    if not manual:
        return linker(per_frame_filaments, cfg)

    # auto-link the rest, then renumber so auto ids can't collide with manual ids
    auto_tracks = linker(auto, cfg)
    offset = max(manual) + 1
    remap = {}
    for tr in auto_tracks:
        remap[tr.track_id] = tr.track_id + offset
        tr.track_id += offset
    for fils in auto:
        for f in fils:
            if f.track_id in remap:
                f.track_id = remap[f.track_id]
    manual_tracks = [_track_from_filaments(mid, grp) for mid, grp in manual.items()]
    return manual_tracks + auto_tracks


def _track_from_filaments(track_id: int, frame_fils: list) -> "Track":
    """Build one Track from hand-grouped ``(frame, filament)`` pairs."""
    frame_fils = sorted(frame_fils, key=lambda tf: tf[0])
    cell_id = next((f.cell_id for _, f in frame_fils if f.cell_id is not None), None)
    tr = Track(track_id, cell_id,
               [t for t, _ in frame_fils],
               [f.length_px for _, f in frame_fils],
               [f.tip_yx for _, f in frame_fils],
               frame_fils[-1][1].base_yx)
    for _, f in frame_fils:
        f.track_id = track_id
    return tr


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
