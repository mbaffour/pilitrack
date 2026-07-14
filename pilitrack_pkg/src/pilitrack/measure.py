"""Measure filament length and associate pili with cells.

Length is the geodesic path along the skeleton (not endpoint distance), because
pili flex by Brownian motion and a chord underestimates true length.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .config import AcquisitionConfig
from .detect import filament_components

_SQRT2 = np.sqrt(2.0)


@dataclass
class Filament:
    label: int
    length_px: float
    base_yx: tuple          # endpoint nearest a cell (assigned later)
    tip_yx: tuple
    coords: np.ndarray      # (k, 2) skeleton pixels
    cell_id: int | None = None
    track_id: int | None = None


def _endpoints(coords: np.ndarray) -> list[tuple]:
    """Skeleton pixels with exactly one 8-neighbour in the component."""
    s = set(map(tuple, coords))
    ends = []
    for (y, x) in s:
        nb = sum(
            (y + dy, x + dx) in s
            for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0)
        )
        if nb <= 1:
            ends.append((y, x))
    return ends


def _order_skeleton(coords: np.ndarray):
    """Order skeleton pixels into a single path from an endpoint.

    Returns an ordered ``(k, 2)`` float array, or ``None`` if the skeleton
    branches (a pixel with more than one unvisited neighbour) or does not form
    one connected path — those fall back to the plain step sum.
    """
    s = set(map(tuple, coords))
    if len(s) < 2:
        return None
    ends = _endpoints(coords)
    start = ends[0] if ends else tuple(coords[0])
    order = [start]
    seen = {start}
    cur = start
    while True:
        nbrs = [(cur[0] + dy, cur[1] + dx)
                for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                if not (dy == 0 and dx == 0)
                and (cur[0] + dy, cur[1] + dx) in s
                and (cur[0] + dy, cur[1] + dx) not in seen]
        if not nbrs:
            break
        if len(nbrs) > 1:          # branch point -> not a simple path
            return None
        cur = nbrs[0]
        seen.add(cur)
        order.append(cur)
    if len(seen) != len(s):        # didn't cover every pixel (branch/disconnect)
        return None
    return np.asarray(order, dtype=float)


def _geodesic_length_px(coords: np.ndarray) -> float:
    """Sub-pixel-corrected path length of a skeleton, in pixels.

    For a simple (unbranched) path we use the Vossepoel-Smeulders corrected
    chain-code estimator ``L = 0.980*N_e + 1.406*N_o - 0.091*N_c`` (orthogonal,
    diagonal, and corner counts). It cuts the naive ``1/sqrt2`` step-sum's ~8%
    oblique-angle length overestimate — which biased every reported length and
    velocity — to under ~2%. Branched skeletons (rare; filament crossings) fall
    back to the plain nearest-neighbour step sum.
    """
    s = set(map(tuple, coords))
    if len(s) <= 1:
        return 0.0
    ordered = _order_skeleton(coords)
    if ordered is not None and len(ordered) >= 2:
        d = np.diff(ordered, axis=0)
        ne = int(np.sum(np.abs(d[:, 0]) + np.abs(d[:, 1]) == 1))
        no = int(len(d) - ne)
        dirs = np.sign(d).astype(int)
        nc = int(np.sum(np.any(dirs[1:] != dirs[:-1], axis=1))) if len(d) > 1 else 0
        return max(0.0, 0.980 * ne + 1.406 * no - 0.091 * nc)
    # branched / disconnected skeleton: plain nearest-neighbour step sum
    total = 0.0
    seen = set()
    ends = _endpoints(coords)
    start = ends[0] if ends else tuple(coords[0])
    stack = [start]
    seen.add(start)
    while stack:
        y, x = stack.pop()
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                nb = (y + dy, x + dx)
                if nb in s and nb not in seen:
                    total += _SQRT2 if (dy != 0 and dx != 0) else 1.0
                    seen.add(nb)
                    stack.append(nb)
    return total


def extract_filaments(skeleton: np.ndarray, cfg: AcquisitionConfig) -> list[Filament]:
    lbl, n = filament_components(skeleton)
    out: list[Filament] = []
    for i in range(1, n + 1):
        coords = np.argwhere(lbl == i)
        ends = _endpoints(coords)
        # a real pilus skeleton is a simple path (2 endpoints); reject branchy
        # blobs that would otherwise overcount length wildly
        if len(ends) > cfg.max_branch_endpoints:
            continue
        length_px = _geodesic_length_px(coords)
        length_nm = length_px * cfg.pixel_size_nm
        if length_px < cfg.min_pilus_length_px or length_nm > cfg.max_pilus_length_nm:
            continue
        if len(ends) >= 2:
            a, b = ends[0], ends[-1]
        else:
            a = b = tuple(coords[0])
        out.append(Filament(i, length_px, a, b, coords))
    return out


def associate_to_cells(
    filaments: list[Filament], cell_labels: np.ndarray, cfg: AcquisitionConfig
) -> list[Filament]:
    """Assign each filament to the nearest cell if a base endpoint lies within
    ``base_search_radius_px`` of that cell. The endpoint closer to a cell
    becomes the base; the other becomes the tip."""
    if cell_labels.max() == 0:
        return filaments
    # distance transform per background: nearest cell id for every pixel
    inds = ndi.distance_transform_edt(
        cell_labels == 0, return_distances=True, return_indices=True
    )
    dist, (iy, ix) = inds[0], inds[1]
    r = cfg.base_search_radius_px
    for f in filaments:
        cand = []
        for end in (f.base_yx, f.tip_yx):
            y, x = end
            d = dist[y, x]
            cid = cell_labels[iy[y, x], ix[y, x]]
            cand.append((d, cid, end))
        cand.sort(key=lambda t: t[0])
        d0, cid0, base = cand[0]
        if d0 <= r and cid0 > 0:
            f.cell_id = int(cid0)
            f.base_yx = base
            f.tip_yx = cand[1][2]
    return filaments
