"""Store hand labels as a training-ready dataset.

A drawn annotation is only useful for future model training if it is stored with
the *pixels it describes* and enough provenance to trust it. ``save_training_bundle``
writes, per labeled frame, the raw image, a rasterized **pilus mask** (the
detector's target), and the **cell mask**, plus the vector annotations and a
metadata record (movie content-hash, pixel size, frame interval, ROI, which
frames are fully labeled, software versions). ``collect_dataset`` then gathers
many such bundles from a folder into a single index for training.

Layout of one bundle::

    <name>/
      images/     frame_000.tif ...   raw labeled frames (uint16)
      pili_masks/ frame_000.tif ...   binary pilus target (uint8 0/255)
      cell_masks/ frame_000.tif ...   instance cell labels (int32) [if available]
      annotations.json                vector traces + track edits + meta
      metadata.json                   provenance for the whole bundle
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import provenance
from .annotate import Annotations, pili_mask, save_annotations


def _pilitrack_version():
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("pilitrack")
    except PackageNotFoundError:  # pragma: no cover
        return None


def label_metadata(movie_path, cfg=None, *, roi=None, frames=None,
                   channel=None, extra=None, hash_max_bytes=None) -> dict:
    """Assemble the provenance record tying a label set to its source pixels."""
    meta = {
        "movie": str(movie_path),
        "movie_fingerprint": (provenance.file_fingerprint(movie_path,
                              max_bytes=hash_max_bytes)
                              if movie_path not in (None, "<array>") else None),
        "roi": list(roi) if roi else None,
        "labeled_frames": list(frames) if frames is not None else None,
        "channel": channel,
        "software": provenance.software_versions(),
        "pilitrack_version": _pilitrack_version(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if cfg is not None:
        meta["pixel_size_nm"] = cfg.pixel_size_nm
        meta["dt_s"] = cfg.dt_s
    meta.update(extra or {})
    return meta


def save_training_bundle(out_dir, *, stack, annotations: Annotations, cfg=None,
                         movie_path=None, cell_labels=None, frames=None,
                         pilus_width_px: int = 3, hash_max_bytes: int | None = None) -> dict:
    """Write a self-contained, training-ready bundle for the labeled frames.

    ``frames`` defaults to the frames that carry at least one trace. Returns the
    metadata dict; the files are written under ``out_dir``.
    """
    import tifffile

    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "pili_masks").mkdir(parents=True, exist_ok=True)
    stack = np.asarray(stack)
    shape = stack.shape[1:]
    if frames is None:
        frames = sorted({int(mp.frame) for mp in annotations.manual_pili})
    frames = [t for t in frames if 0 <= t < stack.shape[0]]

    has_cells = cell_labels is not None
    if has_cells:
        (out / "cell_masks").mkdir(parents=True, exist_ok=True)
        cell_labels = np.asarray(cell_labels)

    for t in frames:
        tag = f"frame_{t:03d}.tif"
        tifffile.imwrite(str(out / "images" / tag),
                         np.asarray(stack[t]), photometric="minisblack")
        mask = pili_mask(annotations, t, shape, pilus_width_px).astype(np.uint8) * 255
        tifffile.imwrite(str(out / "pili_masks" / tag), mask, photometric="minisblack")
        if has_cells:
            tifffile.imwrite(str(out / "cell_masks" / tag),
                             np.asarray(cell_labels[t]).astype(np.int32),
                             photometric="minisblack")

    meta = label_metadata(movie_path, cfg, roi=annotations.meta.get("roi"),
                          frames=frames, hash_max_bytes=hash_max_bytes,
                          extra={"n_pili_traced": len(annotations.manual_pili),
                                 "pilus_width_px": pilus_width_px,
                                 "has_cell_masks": has_cells,
                                 "shape_yx": [int(shape[0]), int(shape[1])],
                                 "notes": annotations.notes})
    annotations.meta = {**annotations.meta, **{k: meta[k] for k in
                        ("movie", "pixel_size_nm", "dt_s", "roi", "labeled_frames")
                        if k in meta}}
    save_annotations(annotations, out / "annotations.json")   # vectors + meta
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def collect_dataset(root) -> "pd.DataFrame":
    """Scan ``root`` for training bundles and return one row per labeled frame
    (image path, pilus-mask path, cell-mask path, movie, frame, pixel size, ...).
    Feed this straight into a training data loader."""
    import pandas as pd

    root = Path(root)
    rows = []
    for meta_path in sorted(root.rglob("metadata.json")):
        bundle = meta_path.parent
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        for img in sorted((bundle / "images").glob("frame_*.tif")):
            tag = img.name
            pm = bundle / "pili_masks" / tag
            cm = bundle / "cell_masks" / tag
            rows.append({
                "bundle": bundle.name,
                "movie": meta.get("movie"),
                "frame": int(img.stem.split("_")[-1]),
                "image": str(img),
                "pili_mask": str(pm) if pm.exists() else None,
                "cell_mask": str(cm) if cm.exists() else None,
                "pixel_size_nm": meta.get("pixel_size_nm"),
                "dt_s": meta.get("dt_s"),
                "sha256": (meta.get("movie_fingerprint") or {}).get("sha256"),
            })
    return pd.DataFrame(rows, columns=["bundle", "movie", "frame", "image",
                                       "pili_mask", "cell_mask", "pixel_size_nm",
                                       "dt_s", "sha256"])


def dataset_summary(root) -> dict:
    """Quick counts for a training dataset folder."""
    df = collect_dataset(root)
    return {
        "n_bundles": int(df["bundle"].nunique()) if not df.empty else 0,
        "n_labeled_frames": int(len(df)),
        "n_movies": int(df["movie"].nunique()) if not df.empty else 0,
        "with_cell_masks": int(df["cell_mask"].notna().sum()) if not df.empty else 0,
    }
