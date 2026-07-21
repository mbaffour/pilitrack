"""Browser version of pilitrack — a small local Streamlit app.

Runs a tiny local server and opens in your browser (`pilitrack-web`); your data
never leaves the machine. Pick or upload a movie, click Analyze, and get the five
measurements, per-frame overlays, publication figures, and one-click CSV
downloads. The heavy work reuses the tested pipeline; only the UI lives here.

The pure helpers ``analyze_for_web`` and ``overlay_rgb`` carry no Streamlit
dependency and are unit-tested; ``main`` (the page) is guarded so the module
imports without Streamlit installed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# absolute imports so `streamlit run webapp.py` (no package context) also works
from pilitrack.io import load_movie, config_from_meta
from pilitrack.analyze import (build_config, _backends, DEFAULT_DETECT_THRESHOLD,
                               pilus_length_timeseries, phase_table)
from pilitrack.pipeline import detect_and_link, summarize
from pilitrack.qc import qc_metrics

DEFAULT_MOVIE = str(Path(__file__).resolve().parents[3] / "Labelled data" / "trial01007.nd2")


def analyze_for_web(path, *, detect_threshold=None, fast=True, frames=None,
                    roi=None, pili_channel=None, cell_channel=None,
                    model=None, downsample=1, organism=None) -> dict:
    """Load + analyze a movie for the web UI. Returns everything the page renders
    (fluor stack, cfg, detect_and_link art, summary, qc, meta). ``model`` (a
    trained detector or path) replaces the ridge filter when given.

    Big movies are handled without loading the whole file: ND2 crops are read
    **lazily** (only the ROI/frames are materialized), and ``downsample`` strides
    the pixels (pixel size is rescaled to match), so a 500 MB movie stays light.
    """
    is_nd2 = isinstance(path, str) and path.lower().endswith(".nd2")
    if fast and roi is None:
        _, _, m0 = load_movie(path, frames=slice(0, 1), as_dask=is_nd2)
        H, W = m0["shape_yx"]
        c = 256
        y0, x0 = max(0, H // 2 - c), max(0, W // 2 - c)
        roi = (y0, y0 + 2 * c, x0, x0 + 2 * c)
        if frames is None:
            frames = slice(0, 15)
    # lazily materialize just the crop for ND2 (avoids loading the full file)
    fluor, cell, meta = load_movie(path, pili_channel=pili_channel,
                                   cell_channel=cell_channel, frames=frames,
                                   roi=roi, as_dask=(is_nd2 and roi is not None))
    d = max(1, int(downsample))
    if d > 1:
        fluor = np.ascontiguousarray(fluor[:, ::d, ::d])
        if cell is not None:
            cell = np.ascontiguousarray(cell[:, ::d, ::d])
        meta = dict(meta)
        meta["shape_yx"] = (fluor.shape[1], fluor.shape[2])
        if meta.get("pixel_size_nm"):
            meta["pixel_size_nm"] = meta["pixel_size_nm"] * d
    # Record the crop origin + stride so hand-labels drawn on this (possibly
    # cropped/downsampled) view can be mapped back to the full movie: a label
    # at (yl, xl) here is full-movie (roi_y0 + yl*d, roi_x0 + xl*d).
    meta = dict(meta)
    meta["roi"] = roi                       # (y0, y1, x0, x1) in full px, or None
    meta["downsample"] = d
    overrides = {"detect_threshold": detect_threshold or DEFAULT_DETECT_THRESHOLD}
    cfg, detection = build_config(meta, overrides=overrides)
    seg, det = _backends(meta["single_channel"], detection)
    prob = None
    if model is not None:
        from pilitrack import ml
        prob = ml.predict_prob_stack(ml.resolve_model(model), fluor,
                                     pixel_size_nm=cfg.pixel_size_nm)
    art = detect_and_link(fluor, fluor if meta["single_channel"] else cell, cfg,
                          segment_fn=seg,
                          detect_fn=(None if prob is not None else det),
                          pilus_prob_stack=prob)
    res = summarize(art["tracks"], art["per_frame_cell_labels"], cfg, art["n_frames"])
    qc = qc_metrics(fluor, art, res, cfg, organism=organism)
    return {"fluor": fluor, "cfg": cfg, "art": art, "res": res, "qc": qc, "meta": meta}


def overlay_rgb(fluor_frame, cell_labels, filaments) -> np.ndarray:
    """Render one frame the way the app shows it: green fluorescence, cyan cell
    outlines, magenta detected pili — a uint8 (H, W, 3) image."""
    from skimage.segmentation import find_boundaries

    img = np.asarray(fluor_frame, float)
    lo, hi = np.percentile(img, [2, 99.5])
    g = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)
    rgb = np.zeros((*img.shape, 3), float)
    rgb[..., 1] = g
    rgb[..., 0] = g * 0.12
    rgb[..., 2] = g * 0.22
    cells = np.asarray(cell_labels)
    if cells.max() > 0:
        rgb[find_boundaries(cells, mode="outer")] = [0.1, 0.9, 0.95]
    for f in filaments:
        c = np.asarray(f.coords)
        if c.size:
            rgb[c[:, 0], c[:, 1]] = [1.0, 0.15, 0.9]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def _frame_stats(art, cfg, fr) -> dict:
    """Per-frame biology readout: how many pili/cells are detected in frame
    ``fr`` and their lengths in µm. Pure (no Streamlit) so it can be unit-tested."""
    fils = art["per_frame_filaments"][fr]
    lengths_um = sorted(float(f.length_px) * cfg.pixel_size_nm / 1000.0 for f in fils)
    cells = art["per_frame_cell_labels"][fr]
    n_cells = int(np.asarray(cells).max()) if cells is not None and np.asarray(cells).size else 0
    return {
        "frame": int(fr),
        "n_pili": len(fils),
        "n_cells": n_cells,
        "lengths_um": lengths_um,
        "mean_length_um": float(np.mean(lengths_um)) if lengths_um else 0.0,
    }


def _click_to_image_yx(val, Himg, Wimg):
    """Map a ``streamlit-image-coordinates`` click to full-resolution image
    coordinates ``[y, x]``.

    ``val`` is the component's return dict ``{x, y, width, height, ...}`` where
    ``x, y`` are in *displayed* pixels and ``width, height`` are the displayed
    image size — so the click scales back to the real image regardless of how it
    was resized for display. Returns ``None`` for a non-point payload.
    """
    if not val or "x" not in val or "y" not in val:
        return None
    dw = float(val.get("width") or Wimg) or Wimg
    dh = float(val.get("height") or Himg) or Himg
    y = min(max(float(val["y"]) * Himg / dh, 0.0), Himg - 1)
    x = min(max(float(val["x"]) * Wimg / dw, 0.0), Wimg - 1)
    return [round(y, 1), round(x, 1)]


def _draw_labels(bg, committed, wip, disp_w):
    """Render the labeling canvas: the frame overlay ``bg`` (H×W×3 uint8) resized
    to ``disp_w`` wide, with finished manual pili (``committed``: list of ``[y,x]``
    polylines) drawn in yellow and the in-progress points (``wip``: ``[y,x]``) in
    orange. Returns ``(PIL image, disp_h)``."""
    from PIL import Image, ImageDraw

    Himg, Wimg = bg.shape[:2]
    disp_h = max(1, int(disp_w * Himg / Wimg))
    im = Image.fromarray(bg).resize((disp_w, disp_h))
    draw = ImageDraw.Draw(im)
    sx, sy = disp_w / Wimg, disp_h / Himg

    def D(pt):
        return (pt[1] * sx, pt[0] * sy)          # [y, x] -> (x_disp, y_disp)

    for poly in committed:
        if len(poly) >= 2:
            draw.line([D(p) for p in poly], fill=(255, 230, 0), width=2)
    if wip:
        if len(wip) >= 2:
            draw.line([D(p) for p in wip], fill=(255, 140, 0), width=2)
        for p in wip:
            x, y = D(p)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 140, 0))
    return im, disp_h


def _measurements(res, qc) -> dict:
    pop = res["population"]
    cells = res["cell"]
    pil = cells[cells["piliated"]] if not cells.empty else cells
    return {
        "piliated_cells": f"{pop['n_piliated_cells']} / {pop['n_cells']}",
        "percent_piliated": (f"{pop['percent_piliated']:.0f}%"
                             if pop["percent_piliated"] == pop["percent_piliated"] else "n/a"),
        "pili_per_cell": (f"{pil['n_pili'].mean():.1f}" if not pil.empty else "n/a"),
        "median_length_nm": _fmt(qc["median_max_length_nm"]),
        "median_ext_nm_s": _fmt(qc["median_extension_velocity_nm_s"]),
        "median_ret_nm_s": _fmt(qc["median_retraction_velocity_nm_s"]),
        "n_qualified": qc["n_qualified_pili"],
    }


def _fmt(v):
    return "n/a" if v is None or (isinstance(v, float) and v != v) else f"{v:.0f}"


# --------------------------------------------------------------------------- #
# Streamlit page (guarded — needs `pip install streamlit`)
# --------------------------------------------------------------------------- #
def main():
    try:
        import streamlit as st
    except Exception as exc:  # pragma: no cover
        raise ImportError("The browser app needs Streamlit: pip install streamlit "
                          "(or `pip install -e '.[web]'`).") from exc
    import io
    import json
    import os
    import tempfile

    st.set_page_config(page_title="pilitrack", page_icon="🔬", layout="wide")
    st.markdown("## 🔬 pilitrack — type-IV pili from a movie")
    st.caption("Runs locally in your browser · your data stays on this machine.")

    with st.sidebar:
        st.header("Movie")
        default = DEFAULT_MOVIE if Path(DEFAULT_MOVIE).exists() else ""
        path = st.text_input("File path", value=default,
                             help="ND2 / TIFF / OME-TIFF / CZI on this computer")
        up = st.file_uploader("…or upload", type=["nd2", "tif", "tiff", "czi"])
        st.caption("**Big movie?** Use the **File path** box above — the file stays "
                   "on disk (no upload, no size limit) and only the part you view "
                   "is loaded. Uploading copies the whole file into memory.")
        st.header("Detector")
        det_choice = st.radio(
            "Pilus detector",
            ["Built-in ridge (recommended)", "Trained model — upload .joblib",
             "Synthetic bootstrap (experimental)"],
            help="Ridge works well on real data today. Upload a model trained on "
                 "YOUR labels for the best results. The synthetic bootstrap is a "
                 "no-labels starting point — it may not match your data (train on "
                 "real labels to beat the ridge).")
        model_file = None
        if det_choice.startswith("Trained model"):
            model_file = st.file_uploader("Trained model (.joblib)", type=["joblib"])
        st.header("Settings")
        from pilitrack.qc import T4P_ENVELOPES, DEFAULT_ORGANISM
        organism = st.selectbox(
            "Organism (QC ranges)", list(T4P_ENVELOPES),
            index=list(T4P_ENVELOPES).index(DEFAULT_ORGANISM),
            help="Sets the biological sanity ranges QC flags against — Neisseria "
                 "pili are longer/faster than P. aeruginosa, so pick your species "
                 "to avoid false flags.")
        thr = st.slider("Detection threshold", 0.15, 0.60, 0.30, 0.05,
                        help="Higher = stricter (less noise, may miss faint pili)")
        fast = st.checkbox("Fast preview (center crop + first frames)", value=True,
                          help="Uncheck for the whole movie — slower and heavier")
        ds = st.select_slider("Downsample (for big/whole-field views)",
                              options=[1, 2, 4], value=1,
                              help="2× or 4× shrinks pixels so the full field fits "
                                   "in memory; pixel size is rescaled to match.")
        go = st.button("Analyze", type="primary", width="stretch")

    if go:
        src = path
        if up is not None:
            suffix = "." + up.name.split(".")[-1]
            # remove the previous uploaded temp file before writing a new one so
            # re-uploads don't pile up full-size copies in the OS temp dir
            prev = st.session_state.get("_upload_tmp")
            if prev:
                try:
                    os.unlink(prev)
                except OSError:
                    pass
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(up.getbuffer())
            tmp.close()
            src = tmp.name
            st.session_state["_upload_tmp"] = src
        if not src or not Path(src).exists():
            st.error("Pick a movie file that exists, or upload one.")
            return
        model = None
        if det_choice.startswith("Synthetic bootstrap"):
            with st.spinner("Preparing the synthetic model (first time trains it)…"):
                from pilitrack import ml
                model = ml.bootstrap_synthetic_model()
        elif det_choice.startswith("Trained model"):
            if model_file is None:
                st.warning("Upload a .joblib model, or pick a different detector.")
                return
            from pilitrack import ml
            mt = tempfile.NamedTemporaryFile(delete=False, suffix=".joblib")
            mt.write(model_file.getbuffer())
            mt.close()
            model = ml.load_model(mt.name)
        with st.spinner("Analyzing… (first run also loads the movie)"):
            try:
                st.session_state["res"] = analyze_for_web(
                    src, detect_threshold=thr, fast=fast, model=model,
                    downsample=ds, organism=organism)
                # a fresh analysis -> drop hand traces tied to the previous movie
                # so they can't leak into (and corrupt) this movie's labels
                for k in ("manual_pili", "wip_points", "_last_click"):
                    st.session_state.pop(k, None)
            except Exception as exc:
                st.exception(exc)
                return

    result = st.session_state.get("res")
    if not result:
        st.info("Pick a movie in the sidebar and press **Analyze**.")
        return

    res, qc, art, cfg, fluor = (result["res"], result["qc"], result["art"],
                                result["cfg"], result["fluor"])
    m = _measurements(res, qc)

    st.subheader("Measurements")
    c = st.columns(5)
    c[0].metric("Piliated cells", m["piliated_cells"], m["percent_piliated"])
    c[1].metric("Pili / cell", m["pili_per_cell"])
    c[2].metric("Pilus length", f"{m['median_length_nm']} nm")
    c[3].metric("Extension", f"{m['median_ext_nm_s']} nm/s")
    c[4].metric("Retraction", f"{m['median_ret_nm_s']} nm/s")
    st.caption(f"From {m['n_qualified']} qualified pili · medians (robust to the noise tail).")
    for f in qc["flags"]:
        st.warning("⚠ " + f)

    T = art["n_frames"]
    Hs, Ws = fluor.shape[1], fluor.shape[2]
    st.caption(f"Loaded **{T}** frame{'s' if T != 1 else ''} · {Hs}×{Ws} px · "
               f"{cfg.pixel_size_nm:.1f} nm/px · dt {cfg.dt_s:.3f} s")
    if T == 1:
        st.warning("Only **1 frame** was loaded — length is measurable, but "
                   "extension/retraction velocities and tracking need a "
                   "time-lapse. If your movie really has many frames, uncheck "
                   "**Fast preview**, or its time axis may be stored unusually "
                   "(e.g. as Z) — tell me the file format and I'll adjust.")

    t_over, t_label, t_fig, t_data = st.tabs(
        ["Frames", "Label (draw)", "Distributions", "Data & downloads"])

    with t_over:
        T = art["n_frames"]
        fr = st.slider("Frame", 0, T - 1, 0) if T > 1 else 0
        rgb = overlay_rgb(fluor[fr], art["per_frame_cell_labels"][fr],
                          art["per_frame_filaments"][fr])
        st.image(rgb, caption=f"Frame {fr} · green = cells, cyan = outline, magenta = pili",
                 width="stretch")
        # per-frame biology readout so you can step through the analysis frame by
        # frame and see what was measured, not just a picture
        sfr = _frame_stats(art, cfg, fr)
        s1, s2, s3 = st.columns(3)
        s1.metric("Pili this frame", sfr["n_pili"])
        s2.metric("Cells this frame", sfr["n_cells"])
        s3.metric("Mean length", f"{sfr['mean_length_um']:.2f} µm")
        if sfr["lengths_um"]:
            st.caption("Lengths (µm): "
                       + ", ".join(f"{v:.2f}" for v in sfr["lengths_um"]))

    with t_label:
        try:
            from streamlit_image_coordinates import streamlit_image_coordinates as st_coords
        except Exception:
            st.info("In-browser labeling needs `streamlit-image-coordinates` "
                    "(`pip install streamlit-image-coordinates`, or "
                    "`pip install -e '.[web]'`). The desktop app `pilitrack-gui` "
                    "also does full labeling.")
        else:
            from pilitrack.annotate import (annotations_from_art, annotations_to_dict,
                                            Annotations, ManualPilus)
            from pilitrack.dataset import save_training_bundle
            store = st.session_state.setdefault("manual_pili", {})   # frame -> [ManualPilus]
            wips = st.session_state.setdefault("wip_points", {})     # frame -> [[y,x]]
            T = art["n_frames"]
            lf = st.slider("Frame to label", 0, T - 1, 0, key="label_frame") if T > 1 else 0
            wip = wips.setdefault(lf, [])
            committed = [mp.points for mp in store.get(lf, [])]

            bg = overlay_rgb(fluor[lf], art["per_frame_cell_labels"][lf],
                             art["per_frame_filaments"][lf])
            Himg, Wimg = bg.shape[:2]
            disp_w = min(700, Wimg * 2) or 512
            st.caption("Magenta = already detected. **Click along a pilus the detector "
                       "MISSED** — point by point, base to tip — then press **Finish "
                       "pilus**. Yellow = finished, orange = in progress.")
            pil, _ = _draw_labels(bg, committed, wip, disp_w)
            val = st_coords(pil, width=disp_w, key=f"coords_{lf}", cursor="crosshair")

            # only act on a genuinely new click (the component re-returns its last
            # value on every rerun, so de-dupe by the click's timestamp/position)
            if val:
                stamp = (lf, val.get("unix_time"), val.get("x"), val.get("y"))
                if st.session_state.get("_last_click") != stamp:
                    st.session_state["_last_click"] = stamp
                    yx = _click_to_image_yx(val, Himg, Wimg)
                    if yx is not None:
                        wip.append(yx)
                        st.rerun()

            c1, c2, c3 = st.columns(3)
            if c1.button(f"✓ Finish pilus ({len(wip)} pts)", disabled=len(wip) < 2,
                         width="stretch"):
                store.setdefault(lf, []).append(
                    ManualPilus(frame=int(lf), points=[list(p) for p in wip]))
                wips[lf] = []
                st.rerun()
            if c2.button("↶ Undo point", disabled=not wip, width="stretch"):
                wip.pop()
                st.rerun()
            if c3.button("🗑 Clear frame", disabled=not (wip or store.get(lf)),
                         width="stretch"):
                wips[lf] = []
                store[lf] = []
                st.rerun()

            added = [mp for v in store.values() for mp in v]
            st.write(f"**{len(store.get(lf, []))}** finished pilus(i) on this frame · "
                     f"**{len(wip)}** point(s) in progress · "
                     f"**{len(added)}** pili across all frames.")

            movie_path = result["meta"].get("path")
            roi = result["meta"].get("roi")
            dsamp = int(result["meta"].get("downsample", 1) or 1)
            # coords are in THIS (possibly cropped/downsampled) view; record how
            # to map them back to the full movie so labels aren't misattributed.
            view_meta = {"roi": list(roi) if roi else None, "downsample": dsamp,
                         "view_shape_yx": [int(fluor.shape[1]), int(fluor.shape[2])]}
            stem = Path(movie_path).stem if movie_path and movie_path != "<array>" else "labels"
            outdir = st.text_input("Save training bundle to folder", value=f"training/{stem}")
            a, b = st.columns(2)
            if a.button("💾 Save labels for training", type="primary"):
                ann = annotations_from_art(art)          # detections as labels
                ann.manual_pili += added                 # + your added traces
                ann.movie = movie_path
                ann.meta.update(view_meta)
                meta = save_training_bundle(
                    outdir, stack=fluor, annotations=ann, cfg=cfg, movie_path=movie_path,
                    cell_labels=np.stack(art["per_frame_cell_labels"]))
                st.success(f"Saved {len(meta.get('labeled_frames') or [])} frames, "
                           f"{len(ann.manual_pili)} pili → {outdir}")
            b.download_button(
                "⬇ Download my traces (JSON)",
                json.dumps(annotations_to_dict(
                    Annotations(manual_pili=added, movie=movie_path,
                                meta=view_meta)), indent=2),
                "my_traces.json", "application/json")

    with t_fig:
        try:
            from pilitrack import figures
            p = figures.plot_distributions(res["pilus"], tempfile.mktemp(suffix=".png"))
            st.image(p, width="stretch")
            p2 = figures.plot_length_traces(art["tracks"], cfg, art["n_frames"],
                                            tempfile.mktemp(suffix=".png"))
            st.image(p2, width="stretch")
        except Exception as exc:
            st.info(f"Figures need matplotlib ({exc}).")

    with t_data:
        st.dataframe(res["pilus"], width="stretch", height=240)
        ts = pilus_length_timeseries(art["tracks"], cfg, art["n_frames"])
        events = phase_table(art["tracks"], cfg, art["n_frames"])
        st.caption("**Per-event kinetics** (each extend / pause / retract phase, "
                   "with dwell time and velocity):")
        st.dataframe(events, width="stretch", height=200)
        dl = st.columns(5)
        dl[0].download_button("pili.csv", res["pilus"].to_csv(index=False),
                              "pili.csv", "text/csv")
        dl[1].download_button("cells.csv", res["cell"].to_csv(index=False),
                              "cells.csv", "text/csv")
        dl[2].download_button("length_over_time.csv", ts.to_csv(index=False),
                              "pilus_length_over_time.csv", "text/csv")
        dl[3].download_button("events.csv", events.to_csv(index=False),
                              "events.csv", "text/csv")
        from pilitrack import provenance
        # use the path actually analyzed (from meta), NOT the live sidebar box —
        # which may have been edited since, or be empty for an uploaded file.
        analyzed_path = result["meta"].get("path") or path
        man = provenance.build_manifest(input_path=analyzed_path, cfg=cfg,
                                        meta=result["meta"],
                                        results_summary=res["population"], qc=qc)
        dl[4].download_button("manifest.json", json.dumps(man, indent=2, default=str),
                              "manifest.json", "application/json")


if __name__ == "__main__":
    main()
