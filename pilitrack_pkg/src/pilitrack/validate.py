"""Validate automated detection against hand-labeled ground truth.

For a methods paper you have to state, in numbers, how well the automation
matches a human: detection precision/recall/F1, and how closely the *measured*
lengths agree with hand-traced ones (bias + Bland-Altman limits of agreement).
This module computes exactly that, given a movie's ``detect_and_link`` result and
a set of **fully** hand-labeled frames (an :class:`~pilitrack.annotate.Annotations`
whose ``manual_pili`` are the ground truth on the listed ``frames``).

Matching is by skeleton overlap within a pixel tolerance (a truth pilus counts
as detected if enough of its centerline lies within ``tol_px`` of an automated
skeleton), computed with a distance transform so it scales to full frames.
Display-free and unit-tested.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .annotate import rasterize_polyline, polyline_length_px


def _label_image(coords_list, shape) -> np.ndarray:
    lab = np.zeros(shape, dtype=np.int32)
    for i, c in enumerate(coords_list, start=1):
        c = np.asarray(c)
        if c.size:
            lab[c[:, 0], c[:, 1]] = i
    return lab


def _nearest(label_img):
    """(distance-to-nearest-labeled-pixel, nearest-label) over the whole image."""
    if label_img.max() == 0:
        return (np.full(label_img.shape, np.inf, float),
                np.zeros(label_img.shape, np.int32))
    dist, (iy, ix) = ndi.distance_transform_edt(
        label_img == 0, return_distances=True, return_indices=True)
    return dist, label_img[iy, ix]


def detection_metrics(art: dict, truth, cfg, frames=None,
                      tol_px: float = 3.0, overlap_frac: float = 0.4) -> dict:
    """Precision / recall / F1 of automated pili vs hand-labeled pili.

    ``frames`` must be **fully** labeled in ``truth`` (every pilus traced); only
    those frames are scored. A truth pilus is a true positive if at least
    ``overlap_frac`` of its centerline lies within ``tol_px`` of some automated
    skeleton; an automated filament with too little overlap with any truth is a
    false positive. Returns counts, rates, and matched (truth, auto) pairs.
    """
    shape = art["shape"]
    truth_by_frame: dict = {}
    for mp in truth.manual_pili:
        truth_by_frame.setdefault(int(mp.frame), []).append(mp)
    if frames is None:
        frames = sorted(truth_by_frame)
    frames = list(frames)

    tp = fp = fn = 0
    matched = []
    for t in frames:
        autos = art["per_frame_filaments"][t]
        auto_coords = [np.asarray(f.coords) for f in autos]
        truths = truth_by_frame.get(t, [])
        truth_coords = [rasterize_polyline(mp.points, shape) for mp in truths]

        auto_dist, auto_near = _nearest(_label_image(auto_coords, shape))
        truth_dist, _ = _nearest(_label_image(truth_coords, shape))

        # recall: each truth pilus detected or missed
        for i, tc in enumerate(truth_coords):
            if len(tc) == 0:
                fn += 1
                continue
            within = auto_dist[tc[:, 0], tc[:, 1]] <= tol_px
            if float(np.mean(within)) >= overlap_frac:
                tp += 1
                labs = auto_near[tc[within, 0], tc[within, 1]]
                labs = labs[labs > 0]
                if labs.size:
                    j = int(np.bincount(labs).argmax()) - 1
                    matched.append((truths[i], autos[j]))
            else:
                fn += 1

        # precision: each automated filament a match or a false positive
        for ac in auto_coords:
            if len(ac) == 0:
                continue
            within = truth_dist[ac[:, 0], ac[:, 1]] <= tol_px
            if float(np.mean(within)) < overlap_frac:
                fp += 1

    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and recall == recall and (precision + recall) > 0
          else float("nan"))
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision,
            "recall": recall, "f1": f1, "frames": frames, "matched": matched}


def length_agreement(matched, cfg) -> dict:
    """Agreement between measured and hand-traced length for matched pili:
    mean abs error, bias (auto - manual), and 95% Bland-Altman limits (nm)."""
    diffs, pairs = [], []
    for truth_mp, auto_fil in matched:
        lt = polyline_length_px(truth_mp.points) * cfg.pixel_size_nm
        la = float(auto_fil.length_px) * cfg.pixel_size_nm
        diffs.append(la - lt)
        pairs.append((lt, la))
    if not diffs:
        return {"n": 0}
    d = np.asarray(diffs, float)
    return {
        "n": len(d),
        "mae_nm": float(np.mean(np.abs(d))),
        "bias_nm": float(np.mean(d)),
        "std_nm": float(np.std(d)),
        "loa_low_nm": float(np.mean(d) - 1.96 * np.std(d)),
        "loa_high_nm": float(np.mean(d) + 1.96 * np.std(d)),
        "pairs_nm": pairs,
    }


def validate(art: dict, truth, cfg, frames=None,
             tol_px: float = 3.0, overlap_frac: float = 0.4) -> dict:
    """Full validation report: detection metrics + length agreement."""
    det = detection_metrics(art, truth, cfg, frames=frames,
                            tol_px=tol_px, overlap_frac=overlap_frac)
    length = length_agreement(det["matched"], cfg)
    return {
        "detection": {k: v for k, v in det.items() if k != "matched"},
        "length_agreement": {k: v for k, v in length.items() if k != "pairs_nm"},
        "n_matched": len(det["matched"]),
        "tol_px": tol_px,
        "overlap_frac": overlap_frac,
    }


# --------------------------------------------------------------------------- #
# CLI: pilitrack-validate movie.nd2 --labels annotations.json
# --------------------------------------------------------------------------- #
def main(argv=None):
    """Score a movie's automated detection against fully hand-labeled frames.

    The labels (`--labels annotations.json`, saved from the GUI) must be a
    *complete* tracing of the frames you validate on, and made with the SAME
    `--roi`/frame origin — otherwise coordinates won't line up.
    """
    import argparse
    import json

    from .io import load_movie
    from .analyze import build_config, _backends, DEFAULT_DETECT_THRESHOLD
    from .annotate import load_annotations
    from .pipeline import detect_and_link

    p = argparse.ArgumentParser(
        description="Validate automated pili detection against hand labels.")
    p.add_argument("path", help="movie file (.nd2/.tif/.czi)")
    p.add_argument("--labels", required=True,
                   help="annotations.json with FULLY-labeled frames (ground truth)")
    p.add_argument("--frames", type=int, nargs="*", default=None,
                   help="frame indices that are fully labeled (default: those in --labels)")
    p.add_argument("--roi", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"),
                   default=None, help="ROI the labels were drawn on (must match)")
    p.add_argument("--config", default=None, help="config.json used for that movie")
    p.add_argument("--detect-threshold", type=float, default=None)
    p.add_argument("--tol-px", type=float, default=3.0)
    p.add_argument("--out", default="validation.json")
    args = p.parse_args(argv)

    truth, _cells = load_annotations(args.labels)
    frames = (args.frames if args.frames
              else sorted({int(mp.frame) for mp in truth.manual_pili}))
    if not frames:
        raise SystemExit("No labeled frames found in --labels.")

    roi = tuple(args.roi) if args.roi else None
    fluor, cell, meta = load_movie(args.path, roi=roi, frames=slice(0, max(frames) + 1))
    overrides = ({} if args.config else
                 {"detect_threshold": args.detect_threshold or DEFAULT_DETECT_THRESHOLD})
    cfg, detection = build_config(meta, config_file=args.config, overrides=overrides)
    seg, det = _backends(meta["single_channel"], detection)
    art = detect_and_link(fluor, fluor if meta["single_channel"] else cell, cfg,
                          segment_fn=seg, detect_fn=det)

    report = validate(art, truth, cfg, frames=frames, tol_px=args.tol_px)
    d, la = report["detection"], report["length_agreement"]
    print("=== validation vs hand labels ===")
    print(f"frames scored : {frames}")
    print(f"precision {d['precision']:.2f}  recall {d['recall']:.2f}  "
          f"F1 {d['f1']:.2f}   (TP {d['tp']}  FP {d['fp']}  FN {d['fn']})")
    if la.get("n"):
        print(f"length agreement (n={la['n']}): bias {la['bias_nm']:.0f} nm, "
              f"MAE {la['mae_nm']:.0f} nm, 95% limits "
              f"[{la['loa_low_nm']:.0f}, {la['loa_high_nm']:.0f}] nm")
    else:
        print("length agreement: no matched pili to compare")
    from pathlib import Path
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out}")
    return report


if __name__ == "__main__":
    main()
