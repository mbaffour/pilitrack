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
    # provenance so a label can be tied back to the exact pixels it describes
    # (movie hash, pixel size, dt, ROI, which frames are fully labeled, versions)
    meta: dict = field(default_factory=dict)


def pili_mask(annotations: "Annotations", frame: int, shape, width_px: int = 3) -> np.ndarray:
    """Binary pilus mask for one frame: every trace on that frame rasterized and
    dilated to ~pilus width. This is the pixel target an ML detector trains on."""
    m = np.zeros(shape, dtype=bool)
    for mp in annotations.manual_pili:
        if int(mp.frame) != int(frame) or len(mp.points) < 2:
            continue
        coords = rasterize_polyline(mp.points, shape)
        if coords.size:
            m[coords[:, 0], coords[:, 1]] = True
    if width_px and width_px > 1:
        import scipy.ndimage as ndi
        m = ndi.binary_dilation(m, iterations=int(width_px) // 2)
    return m


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
                      cell_labels=None, keep_ids=None,
                      replace_auto: bool = False) -> dict:
    """Merge manual pili + (optional) edited cell labels + track removals into an
    ``art`` (``detect_and_link`` result) and re-run linking + summary.

    ``cell_labels``: a ``(T, Y, X)`` int stack (e.g. painted in the GUI) that
    replaces the auto cell segmentation. ``replace_auto=True`` treats the manual
    pili as the *complete* label set (used after seeding the editable layer from
    the detection — otherwise the seeded copies would double-count the auto
    detections). Returns ``{"art": new_art, "summary": {...}}``.
    """
    n = art["n_frames"]
    shape = art["shape"]
    cells = (list(cell_labels) if cell_labels is not None
             else list(art["per_frame_cell_labels"]))

    per_fil = ([[] for _ in range(n)] if replace_auto
               else [list(fils) for fils in art["per_frame_filaments"]])
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
# Pre-annotation: turn the detection into editable labels to correct
# --------------------------------------------------------------------------- #
def _order_skeleton_path(coords, base_yx) -> list:
    """Order skeleton pixels into a path starting from the base endpoint."""
    pts = [tuple(int(v) for v in c) for c in np.asarray(coords)]
    if len(pts) <= 1:
        return [list(base_yx)]
    s = set(pts)
    start = min(pts, key=lambda p: (p[0] - base_yx[0]) ** 2 + (p[1] - base_yx[1]) ** 2)
    ordered, seen, cur = [start], {start}, start
    while True:
        nxt = [(cur[0] + dy, cur[1] + dx)
               for dy in (-1, 0, 1) for dx in (-1, 0, 1)
               if not (dy == 0 and dx == 0)]
        nxt = [p for p in nxt if p in s and p not in seen]
        if not nxt:
            break
        cur = nxt[0]
        seen.add(cur)
        ordered.append(cur)
    return [list(p) for p in ordered]


def annotations_from_art(art: dict, *, simplify_tol: float = 1.5) -> Annotations:
    """Convert a ``detect_and_link`` result into **editable** annotations — the
    auto-detected pili as centerline traces a person can correct (delete false
    ones, adjust, add missed ones). This is the pre-annotation step: detect ->
    JSON -> human corrects -> train. Carries each filament's cell/track id."""
    from skimage.measure import approximate_polygon

    pili = []
    for t, fils in enumerate(art.get("per_frame_filaments", [])):
        for f in fils:
            path = np.asarray(_order_skeleton_path(f.coords, f.base_yx), dtype=float)
            if simplify_tol and path.shape[0] > 2:
                path = approximate_polygon(path, simplify_tol)
            if path.shape[0] < 2:
                path = np.array([list(f.base_yx), list(f.tip_yx)], dtype=float)
            pili.append(ManualPilus(
                frame=int(t), points=path.tolist(),
                cell_id=(int(f.cell_id) if f.cell_id is not None else None),
                track_id=(int(f.track_id) if f.track_id is not None else None)))
    return Annotations(manual_pili=pili, meta={"source": "auto-detection"})


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
        "meta": dict(ann.meta),
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
                       movie=d.get("movie"), notes=d.get("notes", ""),
                       meta=dict(d.get("meta", {})))


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


# --------------------------------------------------------------------------- #
# CLI: pilitrack-prelabel movie.nd2  ->  editable labels JSON to correct
# --------------------------------------------------------------------------- #
def prelabel_main(argv=None):
    """Auto-detect pili on a movie and write an **editable** labels JSON — the
    pre-annotation. A person then opens it (``pilitrack-gui --model`` / the GUI
    Load button), corrects it, and saves; those corrections become training data.
    Runs headless, so you can pre-label a whole batch of movies at once."""
    import argparse

    from .io import load_movie
    from .analyze import build_config, _backends, DEFAULT_DETECT_THRESHOLD
    from .pipeline import detect_and_link

    p = argparse.ArgumentParser(description=prelabel_main.__doc__)
    p.add_argument("path", help="movie file (.nd2/.tif/.czi)")
    p.add_argument("--out", default="prelabels.json")
    p.add_argument("--fast", action="store_true", help="center 512 crop + first 12 frames")
    p.add_argument("--roi", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"), default=None)
    p.add_argument("--frames", type=int, nargs=2, metavar=("START", "STOP"), default=None)
    p.add_argument("--detect-threshold", type=float, default=None)
    args = p.parse_args(argv)

    frames = slice(*args.frames) if args.frames else None
    roi = tuple(args.roi) if args.roi else None
    if args.fast and roi is None:
        _, _, m0 = load_movie(args.path, frames=slice(0, 1))
        H, W = m0["shape_yx"]
        c = 256
        y0, x0 = max(0, H // 2 - c), max(0, W // 2 - c)
        roi = (y0, y0 + 2 * c, x0, x0 + 2 * c)
        if frames is None:
            frames = slice(0, 12)

    fluor, cell, meta = load_movie(args.path, frames=frames, roi=roi)
    cfg, detection = build_config(meta, overrides={
        "detect_threshold": args.detect_threshold or DEFAULT_DETECT_THRESHOLD})
    seg, det = _backends(meta["single_channel"], detection)
    art = detect_and_link(fluor, fluor if meta["single_channel"] else cell, cfg,
                          segment_fn=seg, detect_fn=det)
    ann = annotations_from_art(art)
    ann.movie = str(args.path)
    ann.meta.update({"roi": list(roi) if roi else None,
                     "pixel_size_nm": cfg.pixel_size_nm, "dt_s": cfg.dt_s})
    save_annotations(ann, args.out)
    print(f"Wrote {len(ann.manual_pili)} pre-labelled pili to {args.out}.\n"
          f"Correct them: pilitrack-gui \"{args.path}\" --load {args.out}")
    return ann


if __name__ == "__main__":
    prelabel_main()
