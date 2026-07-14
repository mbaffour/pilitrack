"""Hand-labeling: fold human corrections into the automated result.

Automated detection will always miss faint or crossing pili and mis-segment
touching cells. This module lets a person fix that and have the fix flow through
the *same* measurement path, so the reported numbers include the corrections:

  * **trace a missed pilus** — a drawn centerline becomes a real ``Filament``
    (arc length, endpoints), is associated to a cell, linked across frames, and
    measured for length + extension/retraction velocity like any other;
  * **edit cells** — a corrected cell-label stack (painted in the GUI) replaces
    the auto segmentation for association and per-cell counts;
  * **fix tracks** — remove false-positive track ids before summarizing.

Everything here is display-free and unit-tested; the napari GUI in ``viewer`` is
a thin layer that collects the drawings and calls ``apply_annotations``.
Annotations persist to JSON (and an optional cell-label TIFF) so work is saved,
results are reproducible, and the labels double as training data.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .measure import Filament, associate_to_cells
from .track import link_tracks
from .pipeline import summarize


@dataclass
class ManualPilus:
    """A hand-traced pilus centerline on one frame (image coords, (y, x))."""
    frame: int
    points: list                      # [[y, x], ...] polyline vertices
    cell_id: int | None = None        # force association (else nearest cell)
    track_id: int | None = None       # optional manual grouping across frames


@dataclass
class Annotations:
    """All hand corrections for one movie."""
    manual_pili: list = field(default_factory=list)      # list[ManualPilus]
    removed_track_ids: list = field(default_factory=list)
    movie: str | None = None
    notes: str = ""


# --------------------------------------------------------------------------- #
# Geometry: a drawn polyline -> a Filament
# --------------------------------------------------------------------------- #
def rasterize_polyline(points, shape) -> np.ndarray:
    """Integer pixel coords ``(k, 2)`` along a ``(y, x)`` polyline (deduped,
    clipped to ``shape``)."""
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 1:
        return np.zeros((0, 2), dtype=int)
    H, W = shape
    out, seen = [], set()
    segments = zip(pts[:-1], pts[1:]) if pts.shape[0] >= 2 else [(pts[0], pts[0])]
    for (y0, x0), (y1, x1) in segments:
        n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
        for y, x in zip(np.linspace(y0, y1, n), np.linspace(x0, x1, n)):
            yi = int(min(max(round(y), 0), H - 1))
            xi = int(min(max(round(x), 0), W - 1))
            if (yi, xi) not in seen:
                seen.add((yi, xi))
                out.append((yi, xi))
    return np.asarray(out, dtype=int) if out else np.zeros((0, 2), dtype=int)


def polyline_length_px(points) -> float:
    """Arc length (px) of a ``(y, x)`` polyline — the traced pilus length."""
    p = np.asarray(points, dtype=float)
    if p.shape[0] < 2:
        return 0.0
    return float(np.sum(np.sqrt(np.sum(np.diff(p, axis=0) ** 2, axis=1))))


def manual_filament(points, shape, label: int = -1) -> Filament:
    """Build a ``Filament`` from a drawn centerline. Length is the exact polyline
    arc length; endpoints are the first/last vertex (base assigned on
    association)."""
    coords = rasterize_polyline(points, shape)
    length_px = polyline_length_px(points)
    a = (int(round(points[0][0])), int(round(points[0][1])))
    b = (int(round(points[-1][0])), int(round(points[-1][1])))
    return Filament(label, length_px, a, b, coords)


# --------------------------------------------------------------------------- #
# Fold annotations into a detect_and_link result and re-summarize
# --------------------------------------------------------------------------- #
def apply_annotations(art: dict, annotations: Annotations, cfg,
                      cell_labels=None, keep_ids=None) -> dict:
    """Merge manual pili + (optional) edited cell labels + track removals into an
    ``art`` (``detect_and_link`` result) and re-run linking + summary.

    ``cell_labels``: a ``(T, Y, X)`` int stack (e.g. painted in the GUI) that
    replaces the auto cell segmentation. Returns ``{"art": new_art,
    "summary": {...}}`` where ``new_art`` has the manual filaments merged in.
    """
    n = art["n_frames"]
    shape = art["shape"]
    cells = (list(cell_labels) if cell_labels is not None
             else list(art["per_frame_cell_labels"]))

    per_fil = [list(fils) for fils in art["per_frame_filaments"]]
    next_label = 1 + max(
        (f.label for fils in per_fil for f in fils if f.label is not None),
        default=0)
    for mp in annotations.manual_pili:
        if len(mp.points) < 2 or not (0 <= mp.frame < n):
            continue
        fil = manual_filament(mp.points, shape, label=next_label)
        next_label += 1
        (fil,) = associate_to_cells([fil], np.asarray(cells[mp.frame]), cfg)
        if mp.cell_id is not None:
            fil.cell_id = int(mp.cell_id)
        if mp.track_id is not None:
            fil.track_id = int(mp.track_id)
        per_fil[mp.frame].append(fil)

    tracks = link_tracks(per_fil, cfg)
    removed = set(annotations.removed_track_ids or [])
    all_ids = {tr.track_id for tr in tracks}
    keep = (set(keep_ids) if keep_ids is not None else all_ids) - removed
    summary = summarize(tracks, cells, cfg, n, keep_ids=keep)

    new_art = dict(art)
    new_art["per_frame_filaments"] = per_fil
    new_art["per_frame_cell_labels"] = cells
    new_art["tracks"] = tracks
    return {"art": new_art, "summary": summary}


# --------------------------------------------------------------------------- #
# napari shapes <-> ManualPilus
# --------------------------------------------------------------------------- #
def shapes_to_manual_pili(shapes_data, default_frame: int = 0) -> list:
    """Convert napari Shapes-layer path arrays to ``ManualPilus`` list.

    Each shape is ``(V, 3)`` ``[t, y, x]`` (movie viewer) or ``(V, 2)`` ``[y, x]``.
    """
    out = []
    for arr in shapes_data:
        a = np.asarray(arr, dtype=float)
        if a.ndim != 2 or a.shape[0] < 2:
            continue
        if a.shape[1] >= 3:
            frame = int(round(float(np.median(a[:, 0]))))
            pts = a[:, 1:3]
        else:
            frame = default_frame
            pts = a[:, 0:2]
        out.append(ManualPilus(frame=frame, points=pts.tolist()))
    return out


def manual_pili_to_shapes(manual_pili) -> list:
    """Inverse of ``shapes_to_manual_pili`` — rebuild napari path arrays."""
    shapes = []
    for mp in manual_pili:
        pts = np.asarray(mp.points, dtype=float)
        t = np.full((pts.shape[0], 1), float(mp.frame))
        shapes.append(np.hstack([t, pts]))
    return shapes


# --------------------------------------------------------------------------- #
# Persistence (JSON, + optional cell-label TIFF)
# --------------------------------------------------------------------------- #
def annotations_to_dict(ann: Annotations) -> dict:
    return {
        "movie": ann.movie,
        "notes": ann.notes,
        "removed_track_ids": list(ann.removed_track_ids),
        "manual_pili": [asdict(mp) if isinstance(mp, ManualPilus)
                        else dict(mp) for mp in ann.manual_pili],
    }


def annotations_from_dict(d: dict) -> Annotations:
    pili = [ManualPilus(frame=int(mp["frame"]), points=mp["points"],
                        cell_id=mp.get("cell_id"), track_id=mp.get("track_id"))
            for mp in d.get("manual_pili", [])]
    return Annotations(manual_pili=pili,
                       removed_track_ids=list(d.get("removed_track_ids", [])),
                       movie=d.get("movie"), notes=d.get("notes", ""))


def save_annotations(ann: Annotations, path, cell_labels=None) -> str:
    """Write annotations JSON; if ``cell_labels`` given, a sibling
    ``<stem>_cells.tif`` with the edited cell-label stack."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(annotations_to_dict(ann), indent=2))
    if cell_labels is not None:
        import tifffile
        tifffile.imwrite(str(path.with_name(path.stem + "_cells.tif")),
                         np.asarray(cell_labels).astype(np.int32),
                         photometric="minisblack")  # (T,Y,X) labels, not RGB
    return str(path)


def load_annotations(path):
    """Load ``(Annotations, cell_labels_or_None)`` written by ``save_annotations``."""
    path = Path(path)
    ann = annotations_from_dict(json.loads(path.read_text()))
    cells_path = path.with_name(path.stem + "_cells.tif")
    cell_labels = None
    if cells_path.exists():
        import tifffile
        cell_labels = tifffile.imread(str(cells_path))
    return ann, cell_labels
