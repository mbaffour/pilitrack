"""napari curation viewer.

The pure data-prep functions (label stack, tracks array, curated summary) carry
no napari dependency and are unit-tested. The interactive window lives in
``launch_viewer`` behind a guarded import, since napari needs a display.

Curation flow:
  * scrub the time axis to watch detected pili and their track IDs;
  * note the IDs of spurious tracks (crossing pili linked as one, noise ridges,
    breaks mid-retraction);
  * type them into the widget and recompute -- kinetics run on the kept set.
"""
from __future__ import annotations

import numpy as np

from .config import AcquisitionConfig
from .pipeline import detect_and_link, summarize


def build_skeleton_label_stack(per_frame_filaments: list, shape: tuple,
                               n_frames: int) -> np.ndarray:
    """(T, H, W) integer stack where each pilus's pixels carry ``track_id + 1``
    (0 = background). napari colours each track distinctly, so identity is
    visible frame to frame."""
    H, W = shape
    lbl = np.zeros((n_frames, H, W), dtype=np.int32)
    for t, fils in enumerate(per_frame_filaments):
        for f in fils:
            if f.track_id is None:
                continue
            ys = f.coords[:, 0]
            xs = f.coords[:, 1]
            lbl[t, ys, xs] = f.track_id + 1
    return lbl


def build_tracks_array(tracks: list) -> np.ndarray:
    """(N, 4) array [track_id, t, y, x] from pilus tip positions -- the format
    napari's Tracks layer expects for a 2D+time movie."""
    rows = []
    for tr in tracks:
        for t, (y, x) in zip(tr.frames, tr.tips):
            rows.append([tr.track_id, t, float(y), float(x)])
    if not rows:
        return np.empty((0, 4), dtype=float)
    return np.asarray(rows, dtype=float)


def curated_summary(art: dict, cfg: AcquisitionConfig,
                    remove_ids: set | None = None) -> dict:
    """Summarize kinetics on the kept tracks (all track ids minus removed)."""
    all_ids = {tr.track_id for tr in art["tracks"]}
    keep = all_ids - set(remove_ids or set())
    return summarize(art["tracks"], art["per_frame_cell_labels"],
                     cfg, art["n_frames"], keep_ids=keep)


def launch_viewer(fluor_stack: np.ndarray, cell_stack: np.ndarray,
                  cfg: AcquisitionConfig, **backend_kwargs):
    """Open a napari window for detection review + track culling.

    Returns the napari ``Viewer``. Requires ``pip install "pilitrack[viewer]"``
    (napari + magicgui + a Qt backend). Everything the window shows is built by
    the tested helpers above.
    """
    try:
        import napari
        from magicgui import magicgui
    except Exception as exc:  # pragma: no cover - needs a display
        raise ImportError(
            'napari not available. Install with: pip install "napari[all]" magicgui'
        ) from exc

    art = detect_and_link(fluor_stack, cell_stack, cfg, **backend_kwargs)
    skel_stack = build_skeleton_label_stack(
        art["per_frame_filaments"], art["shape"], art["n_frames"])
    cell_stack_lbl = np.stack(art["per_frame_cell_labels"]).astype(np.int32)
    tracks_arr = build_tracks_array(art["tracks"])

    state = {"summary": curated_summary(art, cfg), "removed": set()}

    viewer = napari.Viewer()
    viewer.add_image(fluor_stack, name="pili (fluorescence)", colormap="green",
                     blending="additive")
    viewer.add_labels(cell_stack_lbl, name="cells", opacity=0.3)
    viewer.add_labels(skel_stack, name="pili by track")
    if tracks_arr.shape[0]:
        viewer.add_tracks(tracks_arr, name="pilus tips")

    @magicgui(call_button="Recompute kinetics on kept tracks",
              remove_ids={"label": "Remove track IDs (comma-separated)"})
    def curate(remove_ids: str = ""):
        ids = {int(x) for x in remove_ids.replace(" ", "").split(",") if x != ""}
        state["removed"] = ids
        state["summary"] = curated_summary(art, cfg, ids)
        pop = state["summary"]["population"]
        n_pili = len(state["summary"]["pilus"])
        napari.utils.notifications.show_info(
            f"Kept {len(art['tracks']) - len(ids)} tracks | "
            f"{n_pili} pili | {pop['percent_piliated']:.0f}% piliated")

    @magicgui(call_button="Export CSVs", directory={"mode": "d"})
    def export(directory=".") -> None:
        from pathlib import Path
        d = Path(directory)
        state["summary"]["pilus"].to_csv(d / "pili.csv", index=False)
        state["summary"]["cell"].to_csv(d / "cells.csv", index=False)
        napari.utils.notifications.show_info(f"Wrote CSVs to {d}")

    viewer.window.add_dock_widget(curate, name="curate")
    viewer.window.add_dock_widget(export, name="export")
    viewer._pilitrack_state = state  # accessible after the window closes
    return viewer


def _results_text(summary, n_manual, n_removed) -> str:
    pop = summary["population"]
    pili = summary["pilus"]

    def med(col):
        if col in pili and not pili.empty:
            v = np.asarray(pili[col], float)
            v = v[np.isfinite(v)]
            return float(np.median(v)) if v.size else float("nan")
        return float("nan")

    pct = pop.get("percent_piliated", float("nan"))
    return "<br>".join([
        "<b>Results</b>",
        f"cells: {pop['n_cells']}",
        f"% piliated: {pct:.0f}" if pct == pct else "% piliated: n/a",
        f"pili (rows): {len(pili)}",
        f"median extension: {med('mean_extension_velocity_nm_s'):.0f} nm/s",
        f"median retraction: {med('mean_retraction_velocity_nm_s'):.0f} nm/s",
        f"median max length: {med('max_length_nm'):.0f} nm",
        f"<i>manual traced: {n_manual} &nbsp; removed: {n_removed}</i>",
    ])


def launch_annotator(fluor_stack: np.ndarray, cell_stack: np.ndarray,
                     cfg: AcquisitionConfig, annotations=None,
                     movie_path: str | None = None, **backend_kwargs):
    """Open a napari window for detection review **and hand-labeling**.

    On top of ``launch_viewer`` this lets a person *correct* the automated
    result and have the corrections flow through the same measurement path:

      * **trace a missed pilus** in the "manual pili (draw)" Shapes layer (select
        it, pick the *path* tool, click along the filament) — it becomes a real,
        measured track;
      * **edit cells** by painting in the "cells (editable)" Labels layer (add a
        missed cell, erase a false one, draw a 0-valued split line between two
        merged cells);
      * **fix tracks** by typing false-positive track ids to remove;
      * **Save / Load annotations** so work persists and is reproducible.

    Hit *Recompute* after any edit to fold it into the kinetics. Runs on a
    laptop; for very large movies load a cropped/downsampled ROI first (see
    ``examples/run_annotate.py``). Needs ``pip install "pilitrack[viewer]"``.
    """
    try:
        import napari
        from magicgui import magicgui
    except Exception as exc:  # pragma: no cover - needs a display
        raise ImportError(
            'napari not available. Install with: pip install "napari[all]" magicgui'
        ) from exc

    from . import annotate as _annotate

    base_art = detect_and_link(fluor_stack, cell_stack, cfg, **backend_kwargs)
    ann = annotations or _annotate.Annotations(movie=movie_path)

    viewer = napari.Viewer()
    viewer.add_image(fluor_stack, name="pili (fluorescence)", colormap="green",
                     blending="additive")
    cells_layer = viewer.add_labels(
        np.stack(base_art["per_frame_cell_labels"]).astype(np.int32),
        name="cells (editable)", opacity=0.35)
    skel_layer = viewer.add_labels(
        build_skeleton_label_stack(base_art["per_frame_filaments"],
                                   base_art["shape"], base_art["n_frames"]),
        name="pili by track")
    tracks_arr = build_tracks_array(base_art["tracks"])
    tracks_layer = viewer.add_tracks(tracks_arr, name="pilus tips") \
        if tracks_arr.shape[0] else None
    shapes_layer = viewer.add_shapes(
        _annotate.manual_pili_to_shapes(ann.manual_pili) or None,
        name="manual pili (draw)", shape_type="path",
        edge_color="magenta", edge_width=2)

    state = {"art": base_art, "summary": None, "removed": set(ann.removed_track_ids)}

    def _recompute():
        manual = _annotate.shapes_to_manual_pili(list(shapes_layer.data))
        cells = np.asarray(cells_layer.data)
        annotations_now = _annotate.Annotations(
            manual_pili=manual, removed_track_ids=list(state["removed"]),
            movie=movie_path)
        out = _annotate.apply_annotations(base_art, annotations_now, cfg,
                                          cell_labels=cells)
        state["art"] = out["art"]
        state["summary"] = out["summary"]
        state["annotations"] = annotations_now
        skel_layer.data = build_skeleton_label_stack(
            out["art"]["per_frame_filaments"], out["art"]["shape"],
            out["art"]["n_frames"])
        cells_layer.data = np.stack(out["art"]["per_frame_cell_labels"]).astype(np.int32)
        new_tracks = build_tracks_array(out["art"]["tracks"])
        if tracks_layer is not None and new_tracks.shape[0]:
            tracks_layer.data = new_tracks
        pop = out["summary"]["population"]
        if state.get("results_label") is not None:
            state["results_label"].setText(
                _results_text(out["summary"], len(manual), len(state["removed"])))
        napari.utils.notifications.show_info(
            f"{len(out['art']['tracks'])} tracks | "
            f"{len(out['summary']['pilus'])} pili | +{len(manual)} manual | "
            f"{pop.get('percent_piliated', float('nan')):.0f}% piliated")

    from qtpy.QtWidgets import QLabel
    results_label = QLabel()
    results_label.setWordWrap(True)
    state["results_label"] = results_label
    _recompute()  # seed the summary from any preloaded annotations

    @magicgui(call_button="Recompute (fold in traced pili + cell edits)")
    def recompute():
        _recompute()

    @magicgui(call_button="Remove track IDs",
              remove_ids={"label": "false-positive track IDs (comma-sep)"})
    def cull(remove_ids: str = ""):
        state["removed"] = {int(x) for x in remove_ids.replace(" ", "").split(",") if x}
        _recompute()

    @magicgui(call_button="Save annotations", path={"label": "annotations.json"})
    def save(path: str = "annotations.json"):
        _recompute()
        _annotate.save_annotations(state["annotations"], path,
                                   cell_labels=np.asarray(cells_layer.data))
        napari.utils.notifications.show_info(f"Saved annotations -> {path}")

    @magicgui(call_button="Load annotations", path={"label": "annotations.json"})
    def load(path: str = "annotations.json"):
        loaded, cells = _annotate.load_annotations(path)
        shapes_layer.data = _annotate.manual_pili_to_shapes(loaded.manual_pili)
        if cells is not None:
            cells_layer.data = np.asarray(cells).astype(np.int32)
        state["removed"] = set(loaded.removed_track_ids)
        _recompute()

    @magicgui(call_button="Export CSVs", directory={"mode": "d"})
    def export(directory=".") -> None:
        from pathlib import Path
        d = Path(directory)
        state["summary"]["pilus"].to_csv(d / "pili.csv", index=False)
        state["summary"]["cell"].to_csv(d / "cells.csv", index=False)
        napari.utils.notifications.show_info(f"Wrote CSVs to {d}")

    @magicgui(call_button="Export figures", directory={"mode": "d"})
    def figs(directory=".") -> None:
        from . import figures as _figures
        paths = _figures.make_report_figures(state["summary"], state["art"],
                                             cfg, directory)
        napari.utils.notifications.show_info(
            f"Wrote {len(paths)} figures to {directory}")

    @magicgui(call_button="Save for training", directory={"mode": "d"})
    def save_training(directory=".") -> None:
        from pathlib import Path
        from . import dataset as _dataset
        _recompute()
        ann = state["annotations"]
        frames = sorted({int(mp.frame) for mp in ann.manual_pili}) or None
        name = Path(movie_path).stem if movie_path else "labels"
        meta = _dataset.save_training_bundle(
            Path(directory) / name, stack=fluor_stack, annotations=ann, cfg=cfg,
            movie_path=movie_path,
            cell_labels=np.asarray(cells_layer.data), frames=frames)
        napari.utils.notifications.show_info(
            f"Saved training bundle ({len(meta.get('labeled_frames') or [])} "
            f"frames) to {directory}")

    viewer.window.add_dock_widget(results_label, name="results", area="right")
    for w, name in [(recompute, "recompute"), (cull, "fix tracks"),
                    (save, "save"), (load, "load"), (export, "CSVs"),
                    (figs, "figures"), (save_training, "save for training")]:
        viewer.window.add_dock_widget(w, name=name, area="right")
    viewer._pilitrack_state = state
    return viewer
