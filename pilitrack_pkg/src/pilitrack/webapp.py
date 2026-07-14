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
                               pilus_length_timeseries)
from pilitrack.pipeline import detect_and_link, summarize
from pilitrack.qc import qc_metrics

DEFAULT_MOVIE = str(Path(__file__).resolve().parents[3] / "Labelled data" / "trial01007.nd2")


def analyze_for_web(path, *, detect_threshold=None, fast=True, frames=None,
                    roi=None, pili_channel=None, cell_channel=None,
                    model=None) -> dict:
    """Load + analyze a movie for the web UI. Returns everything the page renders
    (fluor stack, cfg, detect_and_link art, summary, qc, meta). ``model`` (a
    trained detector or path) replaces the ridge filter when given."""
    if fast and roi is None:
        _, _, m0 = load_movie(path, frames=slice(0, 1))
        H, W = m0["shape_yx"]
        c = 256
        y0, x0 = max(0, H // 2 - c), max(0, W // 2 - c)
        roi = (y0, y0 + 2 * c, x0, x0 + 2 * c)
        if frames is None:
            frames = slice(0, 15)
    fluor, cell, meta = load_movie(path, pili_channel=pili_channel,
                                   cell_channel=cell_channel, frames=frames, roi=roi)
    overrides = {"detect_threshold": detect_threshold or DEFAULT_DETECT_THRESHOLD}
    cfg, detection = build_config(meta, overrides=overrides)
    seg, det = _backends(meta["single_channel"], detection)
    prob = None
    if model is not None:
        from pilitrack import ml
        prob = ml.predict_prob_stack(ml.resolve_model(model), fluor)
    art = detect_and_link(fluor, fluor if meta["single_channel"] else cell, cfg,
                          segment_fn=seg,
                          detect_fn=(None if prob is not None else det),
                          pilus_prob_stack=prob)
    res = summarize(art["tracks"], art["per_frame_cell_labels"], cfg, art["n_frames"])
    qc = qc_metrics(fluor, art, res, cfg)
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


def _canvas_paths_to_manual(objects, scale_y, scale_x, frame):
    """Drawable-canvas freedraw objects -> ManualPilus list in image coords."""
    from pilitrack.annotate import ManualPilus
    out = []
    for obj in (objects or []):
        if obj.get("type") != "path":
            continue
        pts = []
        for cmd in obj.get("path", []):
            if len(cmd) >= 3:                      # ["M"/"Q"/"L", ..., x, y]
                x, y = float(cmd[-2]), float(cmd[-1])
                pts.append([round(y * scale_y, 1), round(x * scale_x, 1)])
        if len(pts) >= 2:
            out.append(ManualPilus(frame=int(frame), points=pts))
    return out


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
        thr = st.slider("Detection threshold", 0.15, 0.60, 0.30, 0.05,
                        help="Higher = stricter (less noise, may miss faint pili)")
        fast = st.checkbox("Fast preview (center crop + first frames)", value=True,
                          help="Uncheck for the whole movie — slower")
        go = st.button("Analyze", type="primary", use_container_width=True)

    if go:
        src = path
        if up is not None:
            suffix = "." + up.name.split(".")[-1]
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(up.getbuffer())
            tmp.close()
            src = tmp.name
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
                    src, detect_threshold=thr, fast=fast, model=model)
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

    t_over, t_label, t_fig, t_data = st.tabs(
        ["Frames", "Label (draw)", "Distributions", "Data & downloads"])

    with t_over:
        T = art["n_frames"]
        fr = st.slider("Frame", 0, T - 1, 0) if T > 1 else 0
        rgb = overlay_rgb(fluor[fr], art["per_frame_cell_labels"][fr],
                          art["per_frame_filaments"][fr])
        st.image(rgb, caption=f"Frame {fr} · green = cells, cyan = outline, magenta = pili",
                 use_container_width=True)

    with t_label:
        try:
            from streamlit_drawable_canvas import st_canvas
        except Exception:
            st.info("In-browser drawing needs `pip install streamlit-drawable-canvas`. "
                    "(Or use the desktop app `pilitrack-gui` for full correction.)")
        else:
            from PIL import Image
            from pilitrack.annotate import (annotations_from_art, annotations_to_dict,
                                            Annotations)
            from pilitrack.dataset import save_training_bundle
            store = st.session_state.setdefault("drawn_pili", {})
            T = art["n_frames"]
            lf = st.slider("Frame to label", 0, T - 1, 0, key="label_frame") if T > 1 else 0
            bg = overlay_rgb(fluor[lf], art["per_frame_cell_labels"][lf],
                             art["per_frame_filaments"][lf])
            Himg, Wimg = bg.shape[:2]
            disp_w = 512
            disp_h = int(disp_w * Himg / Wimg)
            st.caption("Magenta = already detected. Click-drag along any pilus the "
                       "detector MISSED. Detections + your traces are saved as labels.")
            canvas = st_canvas(
                background_image=Image.fromarray(bg).resize((disp_w, disp_h)),
                drawing_mode="freedraw", stroke_width=2, stroke_color="#ffe600",
                height=disp_h, width=disp_w, key=f"cv_{lf}")
            if canvas.json_data is not None:
                store[lf] = _canvas_paths_to_manual(
                    canvas.json_data.get("objects"), Himg / disp_h, Wimg / disp_w, lf)
            added = [mp for v in store.values() for mp in v]
            st.write(f"**{len(store.get(lf, []))}** trace(s) on this frame · "
                     f"**{len(added)}** across all frames you've labelled.")

            movie_path = result["meta"].get("path")
            stem = Path(movie_path).stem if movie_path and movie_path != "<array>" else "labels"
            outdir = st.text_input("Save training bundle to folder", value=f"training/{stem}")
            a, b = st.columns(2)
            if a.button("💾 Save labels for training", type="primary"):
                ann = annotations_from_art(art)          # detections as labels
                ann.manual_pili += added                 # + your added traces
                ann.movie = movie_path
                meta = save_training_bundle(
                    outdir, stack=fluor, annotations=ann, cfg=cfg, movie_path=movie_path,
                    cell_labels=np.stack(art["per_frame_cell_labels"]))
                st.success(f"Saved {len(meta.get('labeled_frames') or [])} frames, "
                           f"{len(ann.manual_pili)} pili → {outdir}")
            b.download_button(
                "⬇ Download my traces (JSON)",
                json.dumps(annotations_to_dict(
                    Annotations(manual_pili=added, movie=movie_path)), indent=2),
                "my_traces.json", "application/json")

    with t_fig:
        try:
            from pilitrack import figures
            p = figures.plot_distributions(res["pilus"], tempfile.mktemp(suffix=".png"))
            st.image(p, use_container_width=True)
            p2 = figures.plot_length_traces(art["tracks"], cfg, art["n_frames"],
                                            tempfile.mktemp(suffix=".png"))
            st.image(p2, use_container_width=True)
        except Exception as exc:
            st.info(f"Figures need matplotlib ({exc}).")

    with t_data:
        st.dataframe(res["pilus"], use_container_width=True, height=280)
        ts = pilus_length_timeseries(art["tracks"], cfg, art["n_frames"])
        dl = st.columns(4)
        dl[0].download_button("pili.csv", res["pilus"].to_csv(index=False),
                              "pili.csv", "text/csv")
        dl[1].download_button("cells.csv", res["cell"].to_csv(index=False),
                              "cells.csv", "text/csv")
        dl[2].download_button("length_over_time.csv", ts.to_csv(index=False),
                              "pilus_length_over_time.csv", "text/csv")
        from pilitrack import provenance
        man = provenance.build_manifest(input_path=path, cfg=cfg, meta=result["meta"],
                                        results_summary=res["population"], qc=qc)
        dl[3].download_button("manifest.json", json.dumps(man, indent=2, default=str),
                              "manifest.json", "application/json")


if __name__ == "__main__":
    main()
