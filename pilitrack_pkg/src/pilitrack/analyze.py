"""High-level, reproducible analysis of one movie file — the single code path
shared by the interactive runner and the batch runner.

``analyze_file`` loads any supported movie, resolves single- vs dual-channel
handling, derives (or loads) the config, runs detect -> summarize, computes QC,
and writes a self-describing result folder: ``pili.csv``, ``cells.csv``, QC
overlay PNGs, and a ``manifest.json`` capturing the input hash, software
versions, and every parameter, so any result can be reproduced exactly.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import AcquisitionConfig
from .io import (load_movie, load_array, config_from_meta, describe_config,
                 save_qc_overlays)
from .pipeline import detect_and_link, summarize
from .singlechannel import make_cell_segmenter, make_pili_detector
from . import provenance
from .qc import qc_metrics

# Detector-only params (NOT AcquisitionConfig fields) passed to the single-
# channel backends. `detect_threshold` and `min_pilus_length_nm` are config
# fields (the detector reads cfg.detect_threshold), so they are handled as config
# overrides, not here.
DEFAULT_DETECTION = dict(
    tophat_radius_px=6.0,
    open_radius_px=6.0,
    min_cell_area_px=80,
    norm_percentile=99.5,
)
# Runner default detection threshold — tuned for real labelled-pilus TIRF SNR
# (the AcquisitionConfig default of 0.20 over-detects on real images).
DEFAULT_DETECT_THRESHOLD = 0.30


def build_config(meta, *, config_file=None, cfg=None, overrides=None):
    """Resolve the AcquisitionConfig + detection params for a movie.

    Precedence: an explicit ``cfg`` or a saved ``config_file`` wins; otherwise
    the config is derived from the movie metadata and ``overrides`` applied.
    Returns ``(cfg, detection_params)``.
    """
    overrides = dict(overrides or {})
    detection = dict(DEFAULT_DETECTION)
    if cfg is not None:
        return cfg, detection
    if config_file is not None:
        loaded_cfg, loaded_det = provenance.load_config(config_file)
        detection.update(loaded_det or {})
        return loaded_cfg, detection
    # split overrides into config fields vs detector-only params. Whitelist the
    # config keys against the AcquisitionConfig dataclass so a loader-only key
    # (e.g. channel_names) or a typo doesn't reach AcquisitionConfig(**) and
    # raise TypeError — such keys are simply not config overrides.
    import dataclasses
    det_keys = set(DEFAULT_DETECTION)
    cfg_keys = {f.name for f in dataclasses.fields(AcquisitionConfig)}
    cfg_keys.add("reference_pixel_size_nm")           # accepted by config_from_meta
    cfg_over = {k: v for k, v in overrides.items()
                if k not in det_keys and k in cfg_keys}
    detection.update({k: v for k, v in overrides.items() if k in det_keys})
    # detect_threshold is a config field the detector reads; apply the runner
    # default unless the caller overrode it.
    cfg_over.setdefault("detect_threshold", DEFAULT_DETECT_THRESHOLD)
    cfg = config_from_meta(meta, **cfg_over)
    return cfg, detection


def _backends(single_channel: bool, detection: dict):
    """(segment_fn, detect_fn) for single- vs dual-channel movies."""
    segment_fn = make_cell_segmenter(
        open_radius_px=detection["open_radius_px"],
        min_cell_area_px=detection["min_cell_area_px"])
    detect_fn = make_pili_detector(
        tophat_radius_px=detection["tophat_radius_px"],
        norm_percentile=detection["norm_percentile"],
        open_radius_px=detection["open_radius_px"],
        # dual channel: pili channel has no cell bodies, so no interior masking
        exclude_cell_interior=single_channel)
    return segment_fn, detect_fn


def analyze_file(
    path,
    *,
    out=None,
    config_file=None,
    cfg=None,
    pili_channel=None,
    cell_channel=None,
    z="max",
    position: int = 0,
    frames=None,
    roi=None,
    overrides=None,
    array=None,
    array_axes="TYX",
    model=None,
    qc_frames: int = 3,
    save_overlays: bool = True,
    hash_max_bytes: int | None = None,
    timestamp: str | None = None,
    verbose: bool = True,
) -> dict:
    """Analyze one movie (file or in-memory ``array``) end to end.

    Returns ``{cfg, meta, art, res, qc, detection, outputs, manifest}``. If
    ``out`` is given, writes CSVs, QC overlays and ``manifest.json`` there.
    """
    if array is not None:
        fluor, cell, meta = load_array(
            array, axes=array_axes, pili_channel=pili_channel,
            cell_channel=cell_channel, z=z, position=position,
            frames=frames, roi=roi,
            **{k: v for k, v in (overrides or {}).items()
               if k in ("pixel_size_nm", "dt_s", "channel_names")})
    else:
        fluor, cell, meta = load_movie(
            path, pili_channel=pili_channel, cell_channel=cell_channel,
            z=z, position=position, frames=frames, roi=roi,
            as_dask=bool(roi) and array is None and str(path).lower().endswith(".nd2"))

    cfg, detection = build_config(meta, config_file=config_file, cfg=cfg,
                                  overrides=overrides)
    if verbose:
        print(describe_config(cfg, meta))
        print(f"  mode                  = "
              f"{'single-channel' if meta['single_channel'] else 'dual-channel'}")

    segment_fn, detect_fn = _backends(meta["single_channel"], detection)
    cell_stack = fluor if meta["single_channel"] else cell

    # a trained ML detector (path or model) replaces the ridge filter by feeding
    # its probability map into the pipeline's pilus_prob_stack seam.
    prob = None
    if model is not None:
        from . import ml
        if verbose:
            print("  detector              = trained ML model")
        prob = ml.predict_prob_stack(ml.resolve_model(model), fluor,
                                     pixel_size_nm=cfg.pixel_size_nm)
    detection["detector"] = "ml" if prob is not None else "ridge"

    art = detect_and_link(fluor, cell_stack, cfg, segment_fn=segment_fn,
                          detect_fn=(None if prob is not None else detect_fn),
                          pilus_prob_stack=prob)
    res = summarize(art["tracks"], art["per_frame_cell_labels"], cfg, art["n_frames"])
    qc = qc_metrics(fluor, art, res, cfg)

    outputs: list[str] = []
    manifest = None
    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        res["pilus"].to_csv(out / "pili.csv", index=False)
        res["cell"].to_csv(out / "cells.csv", index=False)
        # length of each individual pilus over time (the kymograph, as data)
        pilus_length_timeseries(art["tracks"], cfg, art["n_frames"]).to_csv(
            out / "pilus_length_over_time.csv", index=False)
        outputs += [str(out / "pili.csv"), str(out / "cells.csv"),
                    str(out / "pilus_length_over_time.csv")]
        if save_overlays:
            T = art["n_frames"]
            idx = sorted(set(np.linspace(0, T - 1, max(1, qc_frames)).astype(int).tolist()))
            outputs += save_qc_overlays(fluor, art, cfg, idx, out / "qc")
        provenance.save_config(cfg, out / "config.json", detection=detection)
        outputs.append(str(out / "config.json"))
        manifest = provenance.build_manifest(
            input_path=(path if array is None else "<array>"),
            cfg=cfg, meta=meta, detection=detection, roi=roi, frames=frames,
            position=position, results_summary=res["population"], qc=qc,
            outputs=outputs, hash_max_bytes=hash_max_bytes, timestamp=timestamp)
        provenance.write_manifest(manifest, out / "manifest.json")
        outputs.append(str(out / "manifest.json"))

    if verbose:
        _print_report(res, qc)
    return {"cfg": cfg, "meta": meta, "art": art, "res": res, "qc": qc,
            "detection": detection, "outputs": outputs, "manifest": manifest}


def _print_report(res, qc):
    pop = res["population"]
    cells = res["cell"]
    pil = cells[cells["piliated"]] if not cells.empty else cells
    mean_pili = float(pil["n_pili"].mean()) if not pil.empty else float("nan")
    # the five measurements the lab asks a T4P tool for
    print("\n=== measurements ===")
    print(f"piliated cells             : {pop['n_piliated_cells']} of "
          f"{pop['n_cells']}"
          + (f"  ({pop['percent_piliated']:.0f}%)"
             if pop['percent_piliated'] == pop['percent_piliated'] else ""))
    print(f"pili per piliated cell     : {_fmt(mean_pili)} (mean)")
    print(f"individual pilus length    : {_fmt(qc['median_max_length_nm'])} nm (median max)")
    print(f"extension velocity / pilus : {_fmt(qc['median_extension_velocity_nm_s'])} nm/s (median)")
    print(f"retraction velocity / pilus: {_fmt(qc['median_retraction_velocity_nm_s'])} nm/s (median)")
    print(f"(from {qc['n_qualified_pili']} qualified pili; "
          f"per-pilus rows in pili.csv, length(t) in pilus_length_over_time.csv)")
    if pop["n_cells"] == 0:
        print("note: no cells segmented — pili-only mode "
              "(per-pilus length/velocity still valid; % piliated unavailable)")
    if qc["flags"]:
        print("\n[QC FLAGS]")
        for f in qc["flags"]:
            print("  ! " + f)
    else:
        print("\n[QC] clean, no flags.")


def _fmt(v):
    return "n/a" if (v is None or (isinstance(v, float) and v != v)) else f"{v:.0f}"


def pilus_length_timeseries(tracks, cfg, n_frames) -> "pd.DataFrame":
    """Length of each individual pilus over time — long format, one row per
    (pilus, frame): ``track_id, cell_id, frame, time_s, length_nm``. This is the
    kymograph as data (the ``length of individual pili`` measurement)."""
    rows = []
    for tr in tracks:
        series = tr.length_series(n_frames) * cfg.pixel_size_nm
        for f, L in enumerate(series):
            if np.isfinite(L):
                rows.append({"track_id": tr.track_id, "cell_id": tr.cell_id,
                             "frame": int(f), "time_s": round(f * cfg.dt_s, 4),
                             "length_nm": round(float(L), 2)})
    return pd.DataFrame(rows, columns=["track_id", "cell_id", "frame",
                                       "time_s", "length_nm"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze one pili movie (ND2/TIFF/OME-TIFF/CZI).")
    p.add_argument("path", help="movie file (.nd2/.tif/.tiff/.czi)")
    p.add_argument("--out", default="pilitrack_out")
    p.add_argument("--config", default=None, help="saved config.json/.yaml to reuse")
    # channels / dims
    p.add_argument("--pili-channel", type=int, default=None)
    p.add_argument("--cell-channel", type=int, default=None)
    p.add_argument("--z", default="max", help="'max','mean', or an integer plane")
    p.add_argument("--position", type=int, default=0)
    p.add_argument("--roi", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"), default=None)
    p.add_argument("--frames", type=int, nargs=2, metavar=("START", "STOP"), default=None)
    p.add_argument("--fast", action="store_true", help="central 512 ROI + first 12 frames")
    # physical overrides
    p.add_argument("--dt", type=float, default=None, dest="dt_s")
    p.add_argument("--pixel-size-nm", type=float, default=None, dest="pixel_size_nm")
    p.add_argument("--min-length", type=float, default=None, dest="min_pilus_length_nm")
    # detection
    p.add_argument("--detect-threshold", type=float, default=None)
    p.add_argument("--tophat", type=float, default=None, dest="tophat_radius_px")
    p.add_argument("--open-radius", type=float, default=None, dest="open_radius_px")
    p.add_argument("--min-cell-area", type=int, default=None, dest="min_cell_area_px")
    p.add_argument("--model", default=None,
                   help="trained detector .joblib, or 'bootstrap' for the "
                        "no-labels synthetic model")
    p.add_argument("--qc-frames", type=int, default=3)
    p.add_argument("--no-viewer", action="store_true")
    return p


def resolve_model_arg(model):
    """CLI/GUI ``--model`` value -> a model dict (or None). ``'bootstrap'`` trains
    (or reloads) the synthetic no-labels model; anything else is a path."""
    if not model:
        return None
    if str(model).lower() == "bootstrap":
        from . import ml
        return ml.bootstrap_synthetic_model()
    from . import ml
    return ml.load_model(model)


def _overrides_from_args(args) -> dict:
    keys = ("dt_s", "pixel_size_nm", "min_pilus_length_nm", "detect_threshold",
            "tophat_radius_px", "open_radius_px", "min_cell_area_px")
    ov = {k: getattr(args, k) for k in keys if getattr(args, k, None) is not None}
    if args.detect_threshold is None and args.config is None:
        ov["detect_threshold"] = DEFAULT_DETECT_THRESHOLD
    return ov


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    z = args.z
    if isinstance(z, str) and z.lstrip("-").isdigit():
        z = int(z)
    frames = slice(*args.frames) if args.frames else None
    roi = tuple(args.roi) if args.roi else None
    if args.fast and roi is None:
        # resolve central 512 crop after a quick metadata peek
        _, _, m0 = load_movie(args.path, frames=slice(0, 1))
        H, W = m0["shape_yx"]
        c = 256
        y0, x0 = max(0, H // 2 - c), max(0, W // 2 - c)
        roi = (y0, y0 + 2 * c, x0, x0 + 2 * c)
        if frames is None:
            frames = slice(0, 12)

    result = analyze_file(
        args.path, out=args.out, config_file=args.config,
        pili_channel=args.pili_channel, cell_channel=args.cell_channel,
        z=z, position=args.position, frames=frames, roi=roi,
        overrides=_overrides_from_args(args), model=resolve_model_arg(args.model),
        qc_frames=args.qc_frames)
    print(f"\nWrote results to {args.out}")

    if not args.no_viewer:
        try:
            from .viewer import launch_viewer
            cfg = result["cfg"]
            meta = result["meta"]
            fluor = None  # re-load lazily only if the user wants the window
            print("\nOpening napari viewer (close to finish) ...")
            fluor, cell, _ = load_movie(args.path, pili_channel=args.pili_channel,
                                        cell_channel=args.cell_channel, z=z,
                                        position=args.position, frames=frames, roi=roi)
            seg, det = _backends(meta["single_channel"], result["detection"])
            launch_viewer(fluor, fluor if meta["single_channel"] else cell, cfg,
                          segment_fn=seg, detect_fn=det)
        except Exception as exc:  # napari/display optional
            print(f"(viewer unavailable: {exc}; pass --no-viewer to silence)")
    return result


if __name__ == "__main__":
    main()
