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


def _neighbors8(y, x):
    return [(y + dy, x + dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)
            if not (dy == 0 and dx == 0)]


def _connected_paths(pixel_set):
    """8-connected components of a set of pixels, each ordered as a path."""
    remaining = set(pixel_set)
    out = []
    while remaining:
        seed = next(iter(remaining))
        comp, stack = set(), [seed]
        while stack:
            p = stack.pop()
            if p in comp:
                continue
            comp.add(p)
            for nb in _neighbors8(*p):
                if nb in remaining and nb not in comp:
                    stack.append(nb)
        remaining -= comp
        ordered = _order_skeleton(np.array(sorted(comp)))
        out.append([tuple(int(v) for v in p) for p in ordered] if ordered is not None
                   else [tuple(int(v) for v in p) for p in comp])
    return out


def _pixel_clusters(pixel_set):
    """8-connected clusters of pixels (unordered sets)."""
    remaining = set(pixel_set)
    out = []
    while remaining:
        seed = next(iter(remaining))
        comp, stack = set(), [seed]
        while stack:
            p = stack.pop()
            if p in comp:
                continue
            comp.add(p)
            for nb in _neighbors8(*p):
                if nb in remaining and nb not in comp:
                    stack.append(nb)
        remaining -= comp
        out.append(comp)
    return out


def _seg_outward_dir(seg, end_idx):
    """Unit direction pointing from a segment's ``end_idx`` end into the segment
    (i.e. away from the junction the end touches)."""
    pts = seg if end_idx == 0 else seg[::-1]
    k = min(len(pts) - 1, 4)
    if k <= 0:
        return np.array([0.0, 0.0])
    v = np.array(pts[k], float) - np.array(pts[0], float)
    nrm = float(np.hypot(v[0], v[1]))
    return v / nrm if nrm > 0 else np.array([0.0, 0.0])


def _decompose_component(coords, max_junctions: int = 8):
    """Split one skeleton component into through-paths, resolving crossings by
    orientation continuity (SOAX/KnotResolver style): break the skeleton at
    junction pixels, then re-join the two segments that pass most straight
    through each junction (dot of outward directions most negative). Returns a
    list of coordinate arrays, each a single filament. Simple (unbranched)
    components are returned unchanged; hopeless blobs (many junctions) are
    dropped by returning ``[]``."""
    s = set(tuple(int(v) for v in p) for p in coords)
    if len(s) < 2:
        return [np.array(sorted(s))] if s else []
    deg = {p: sum((nb in s) for nb in _neighbors8(*p)) for p in s}
    junctions = {p for p in s if deg[p] >= 3}
    if not junctions:
        return [np.array(sorted(s))]
    if len(junctions) > max_junctions:          # a blob, not filaments
        return []

    segments = _connected_paths(s - junctions)
    parent = list(range(len(segments)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    import itertools
    used_end = set()
    bridges = []                                # (si, sj, frozenset(cluster))
    # a crossing skeletonizes to a small BLOCK of junction pixels, so pair arms
    # per junction *cluster*, not per pixel (each pixel only touches one arm).
    for cluster in _pixel_clusters(junctions):
        ends, seen = [], set()
        for si, seg in enumerate(segments):
            for ei, ep in ((0, seg[0]), (1, seg[-1])):
                if (si, ei) in seen:
                    continue
                if any(nb in cluster for nb in _neighbors8(*ep)):
                    ends.append((si, ei, _seg_outward_dir(seg, ei)))
                    seen.add((si, ei))
        avail = [e for e in ends if (e[0], e[1]) not in used_end]
        cand = sorted(
            ((float(np.dot(a[2], b[2])), ia, ib)
             for (ia, a), (ib, b) in itertools.combinations(enumerate(avail), 2)),
            key=lambda t: t[0])
        taken = set()
        for dot, ia, ib in cand:
            if ia in taken or ib in taken or dot > -0.2:   # not a straight pass-through
                continue
            taken.add(ia); taken.add(ib)
            a, b = avail[ia], avail[ib]
            used_end.add((a[0], a[1])); used_end.add((b[0], b[1]))
            parent[find(a[0])] = find(b[0])
            bridges.append((a[0], b[0], cluster))

    groups = {}
    for si, seg in enumerate(segments):
        groups.setdefault(find(si), set()).update(seg)
    for si, sj, cluster in bridges:
        groups[find(si)].update(cluster)
    # junction pixels never used in a pairing (e.g. a T-branch) attach to any
    # touching group so they aren't lost
    assigned = set().union(*groups.values()) if groups else set()
    for j in junctions:
        if j in assigned:
            continue
        for si, seg in enumerate(segments):
            if any(j in _neighbors8(*ep) for ep in (seg[0], seg[-1])):
                groups[find(si)].add(j)
                break

    return [np.array(sorted(g)) for g in groups.values() if len(g) >= 2]


def extract_filaments(skeleton: np.ndarray, cfg: AcquisitionConfig) -> list[Filament]:
    lbl, n = filament_components(skeleton)
    out: list[Filament] = []
    next_label = 1
    for i in range(1, n + 1):
        coords = np.argwhere(lbl == i)
        ends = _endpoints(coords)
        s = set((int(p[0]), int(p[1])) for p in coords)
        deg = {p: sum((nb in s) for nb in _neighbors8(*p)) for p in s}
        jclusters = _pixel_clusters({p for p in s if deg[p] >= 3})

        # Resolve genuine crossings (a few junction clusters, e.g. an X or T of
        # pili) into separate through-paths. Very branchy components (rings /
        # blobs) keep the original behavior: one filament if not too branchy,
        # else dropped — so this only ADDS separation for clean crossings.
        paths = []
        if jclusters and len(jclusters) <= 2 and len(ends) <= 6:
            paths = _decompose_component(coords)
        if not paths:
            paths = ([np.asarray(coords)]
                     if (not jclusters or len(ends) <= cfg.max_branch_endpoints)
                     else [])

        for path in paths:
            pends = _endpoints(path)
            length_px = _geodesic_length_px(path)
            length_nm = length_px * cfg.pixel_size_nm
            if length_px < cfg.min_pilus_length_px or length_nm > cfg.max_pilus_length_nm:
                continue
            if len(pends) >= 2:
                a, b = pends[0], pends[-1]
            else:
                a = b = tuple(int(v) for v in path[0])
            out.append(Filament(next_label, length_px, a, b, path))
            next_label += 1
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
