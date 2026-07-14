"""The pilitrack desktop app — an easy, GUI-first entry point.

Open a movie (file dialog or path), it runs the analysis with sensible defaults,
and drops you straight into the interactive window: a live results readout, and
the tools to review, **hand-label** missed pili, edit cells, cull false tracks,
and export CSVs / figures — no coding required. Runs on a laptop.

    pilitrack-gui                         # file dialog
    pilitrack-gui movie.nd2 --fast        # open a file, center crop for speed
    python -m pilitrack movie.ome.tif

Everything analytical lives in the tested library; this is a thin, friendly shell
over ``pilitrack.viewer.launch_annotator``.
"""
from __future__ import annotations

import argparse

from .io import load_movie, describe_config
from .analyze import build_config, _backends, DEFAULT_DETECT_THRESHOLD


def _pick_file():
    from qtpy.QtWidgets import QApplication, QFileDialog
    _ = QApplication.instance() or QApplication([])
    path, _sel = QFileDialog.getOpenFileName(
        None, "Open a pili movie", "",
        "Movies (*.nd2 *.tif *.tiff *.czi);;All files (*)")
    return path or None


def launch_app(path=None, *, fast: bool = False, roi=None, frames=None,
               pili_channel=None, cell_channel=None, config_file=None,
               detect_threshold=None, run: bool = True):
    """Open a movie and launch the interactive analysis/annotation window.

    Returns the napari ``Viewer``. With ``run=True`` (default) it also starts the
    Qt event loop, so this call blocks until the window is closed — that's what
    makes it a standalone app; pass ``run=False`` to drive it from a script.
    """
    import napari
    from .viewer import launch_annotator

    if path is None:
        path = _pick_file()
        if not path:
            print("No file chosen.")
            return None

    if fast and roi is None:
        _, _, m0 = load_movie(path, frames=slice(0, 1))
        H, W = m0["shape_yx"]
        c = 256
        y0, x0 = max(0, H // 2 - c), max(0, W // 2 - c)
        roi = (y0, y0 + 2 * c, x0, x0 + 2 * c)
        if frames is None:
            frames = slice(0, 20)

    print(f"Loading {path} ...", flush=True)
    fluor, cell, meta = load_movie(path, pili_channel=pili_channel,
                                   cell_channel=cell_channel, roi=roi, frames=frames)
    overrides = ({} if config_file else
                 {"detect_threshold": detect_threshold or DEFAULT_DETECT_THRESHOLD})
    cfg, detection = build_config(meta, config_file=config_file, overrides=overrides)
    print(describe_config(cfg, meta), flush=True)
    print(f"  mode: {'single-channel' if meta['single_channel'] else 'dual-channel'}"
          "  — analyzing (this can take a moment on a full frame) ...", flush=True)

    seg, det = _backends(meta["single_channel"], detection)
    viewer = launch_annotator(fluor, fluor if meta["single_channel"] else cell,
                              cfg, segment_fn=seg, detect_fn=det, movie_path=path)
    print("Window open. Trace missed pili, edit cells, then Recompute / Export.",
          flush=True)
    if run:
        napari.run()
    return viewer


def main(argv=None):
    p = argparse.ArgumentParser(
        description="pilitrack GUI — open a movie, analyze, hand-label, export.")
    p.add_argument("path", nargs="?", default=None,
                   help="movie file (.nd2/.tif/.czi); omit for a file dialog")
    p.add_argument("--fast", action="store_true",
                   help="center 512 crop + first 20 frames (laptop-light)")
    p.add_argument("--roi", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                   default=None)
    p.add_argument("--frames", type=int, nargs=2, metavar=("START", "STOP"),
                   default=None)
    p.add_argument("--pili-channel", type=int, default=None)
    p.add_argument("--cell-channel", type=int, default=None)
    p.add_argument("--config", default=None, help="reuse a saved config.json")
    p.add_argument("--detect-threshold", type=float, default=None)
    args = p.parse_args(argv)

    frames = slice(*args.frames) if args.frames else None
    roi = tuple(args.roi) if args.roi else None
    launch_app(args.path, fast=args.fast, roi=roi, frames=frames,
               pili_channel=args.pili_channel, cell_channel=args.cell_channel,
               config_file=args.config, detect_threshold=args.detect_threshold)


if __name__ == "__main__":
    main()
