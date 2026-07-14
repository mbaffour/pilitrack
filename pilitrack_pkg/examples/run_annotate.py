"""Open the hand-labeling GUI on a movie (trace missed pili, edit cells, fix tracks).

    python examples/run_annotate.py "Labelled data/trial01007.nd2" --fast
    python examples/run_annotate.py movie.nd2 --roi 600 1100 600 1100
    python examples/run_annotate.py movie.nd2 --load annotations.json   # resume

Runs on a laptop. For big movies, work on a crop (--fast or --roi) so napari
stays light. In the window: draw along a missed pilus in "manual pili (draw)",
paint the "cells (editable)" layer to fix cells, then hit Recompute; Save writes
annotations.json (+ edited cell labels) you can reload later.

Needs the viewer extra:  pip install "pilitrack[viewer]"
"""
import argparse

from pilitrack.io import load_movie, describe_config
from pilitrack.analyze import build_config, _backends, DEFAULT_DETECT_THRESHOLD
from pilitrack.annotate import load_annotations


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("path")
    p.add_argument("--roi", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"), default=None)
    p.add_argument("--frames", type=int, nargs=2, metavar=("START", "STOP"), default=None)
    p.add_argument("--fast", action="store_true", help="central 512 ROI (laptop-light)")
    p.add_argument("--pili-channel", type=int, default=None)
    p.add_argument("--cell-channel", type=int, default=None)
    p.add_argument("--config", default=None)
    p.add_argument("--detect-threshold", type=float, default=None)
    p.add_argument("--load", default=None, help="annotations.json to preload")
    args = p.parse_args()

    frames = slice(*args.frames) if args.frames else None
    roi = tuple(args.roi) if args.roi else None
    if args.fast and roi is None:
        _, _, m0 = load_movie(args.path, frames=slice(0, 1))
        H, W = m0["shape_yx"]
        c = 256
        y0, x0 = max(0, H // 2 - c), max(0, W // 2 - c)
        roi = (y0, y0 + 2 * c, x0, x0 + 2 * c)

    fluor, cell, meta = load_movie(args.path, pili_channel=args.pili_channel,
                                   cell_channel=args.cell_channel,
                                   frames=frames, roi=roi)
    overrides = {} if args.config else {
        "detect_threshold": args.detect_threshold or DEFAULT_DETECT_THRESHOLD}
    cfg, detection = build_config(meta, config_file=args.config, overrides=overrides)
    print(describe_config(cfg, meta))

    seg, det = _backends(meta["single_channel"], detection)
    annotations = None
    if args.load:
        annotations, _cells = load_annotations(args.load)

    from pilitrack.viewer import launch_annotator
    print("\nOpening annotator … draw missed pili, paint cells, Recompute, Save.")
    launch_annotator(fluor, fluor if meta["single_channel"] else cell, cfg,
                     segment_fn=seg, detect_fn=det, annotations=annotations,
                     movie_path=args.path)


if __name__ == "__main__":
    main()
