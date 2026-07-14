"""Batch analysis: run one consistent config over a whole folder of movies.

Reproducible-by-construction — every movie in an experiment is analyzed with the
*same* settings (auto-derived per file, or a shared ``config.json``), each into
its own result folder, and rolled up into a single ``summary.csv`` (one row per
movie, with QC flags) and a combined ``pili_all.csv`` (every pilus, tagged by
movie) for downstream stats.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .analyze import analyze_file, DEFAULT_DETECTION
from . import provenance

SUPPORTED_SUFFIXES = (".nd2", ".tif", ".tiff", ".ome.tif", ".ome.tiff", ".czi")


def find_movies(folder, patterns=SUPPORTED_SUFFIXES, recursive: bool = False) -> list:
    """List supported movie files in ``folder`` (sorted, deterministic)."""
    folder = Path(folder)
    globber = folder.rglob if recursive else folder.glob
    found: set = set()
    for p in globber("*"):
        if p.is_file() and p.name.lower().endswith(tuple(patterns)):
            found.add(p)
    return sorted(found)


def _summary_row(name, result) -> dict:
    pop = result["res"]["population"]
    qc = result["qc"]
    cfg = result["cfg"]
    return {
        "movie": name,
        "n_cells": pop.get("n_cells"),
        "percent_piliated": pop.get("percent_piliated"),
        "n_qualified_pili": qc.get("n_qualified_pili"),
        "n_tracks": qc.get("n_tracks"),
        "median_extension_velocity_nm_s": qc.get("median_extension_velocity_nm_s"),
        "median_retraction_velocity_nm_s": qc.get("median_retraction_velocity_nm_s"),
        "median_max_length_nm": qc.get("median_max_length_nm"),
        "pixel_size_nm": cfg.pixel_size_nm,
        "dt_s": cfg.dt_s,
        "saturated_fraction": qc.get("saturated_fraction"),
        "detection_rate": qc.get("detection_rate"),
        "n_flags": len(qc.get("flags", [])),
        "flags": " | ".join(qc.get("flags", [])),
    }


def run_batch(
    folder,
    out,
    *,
    config_file=None,
    recursive: bool = False,
    save_overlays: bool = True,
    hash_max_bytes: int | None = None,
    overrides=None,
    verbose: bool = True,
    **analyze_kwargs,
) -> dict:
    """Analyze every movie in ``folder`` into ``out/<movie>/`` and roll up.

    Returns ``{summary: DataFrame, rows: list, failures: list, outputs: list}``.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    movies = find_movies(folder, recursive=recursive)
    if verbose:
        print(f"Found {len(movies)} movie(s) in {folder}")
    if config_file is None and overrides is None:
        overrides = {"detect_threshold": DEFAULT_DETECTION["detect_threshold"]}

    rows, all_pili, failures = [], [], []
    for i, path in enumerate(movies, 1):
        name = path.stem
        if verbose:
            print(f"\n[{i}/{len(movies)}] {path.name}")
        try:
            result = analyze_file(
                path, out=out / name, config_file=config_file,
                overrides=overrides, save_overlays=save_overlays,
                hash_max_bytes=hash_max_bytes, verbose=verbose, **analyze_kwargs)
        except Exception as exc:  # keep the batch going; record the failure
            print(f"  ! FAILED: {exc}")
            failures.append({"movie": name, "path": str(path), "error": repr(exc)})
            continue
        rows.append(_summary_row(name, result))
        pili = result["res"]["pilus"].copy()
        if not pili.empty:
            pili.insert(0, "movie", name)
            all_pili.append(pili)

    summary = pd.DataFrame(rows)
    outputs = []
    summary_path = out / "summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs.append(str(summary_path))
    if all_pili:
        combined = pd.concat(all_pili, ignore_index=True)
        combined.to_csv(out / "pili_all.csv", index=False)
        outputs.append(str(out / "pili_all.csv"))
    if failures:
        pd.DataFrame(failures).to_csv(out / "failures.csv", index=False)
        outputs.append(str(out / "failures.csv"))

    batch_manifest = {
        "tool": "pilitrack.batch",
        "folder": str(folder),
        "n_movies": len(movies),
        "n_succeeded": len(rows),
        "n_failed": len(failures),
        "config_file": str(config_file) if config_file else None,
        "overrides": provenance._json_safe(overrides),
        "software": provenance.software_versions(),
        "outputs": outputs,
    }
    provenance.write_manifest(batch_manifest, out / "batch_manifest.json")

    if verbose:
        print(f"\n=== batch summary ({len(rows)} ok, {len(failures)} failed) ===")
        if not summary.empty:
            cols = ["movie", "n_cells", "percent_piliated",
                    "median_extension_velocity_nm_s", "median_max_length_nm", "n_flags"]
            print(summary[cols].to_string(index=False))
        print(f"\nWrote {out/'summary.csv'}")
    return {"summary": summary, "rows": rows, "failures": failures, "outputs": outputs}


def main(argv=None):
    p = argparse.ArgumentParser(description="Batch-analyze a folder of pili movies.")
    p.add_argument("folder", help="folder of movies (.nd2/.tif/.czi)")
    p.add_argument("--out", default="pilitrack_batch")
    p.add_argument("--config", default=None, help="shared config.json/.yaml")
    p.add_argument("--recursive", action="store_true")
    p.add_argument("--no-overlays", action="store_true", help="skip QC PNGs (faster)")
    p.add_argument("--detect-threshold", type=float, default=None)
    p.add_argument("--quick-hash", action="store_true",
                   help="fingerprint only the first 8 MB of each file (faster)")
    args = p.parse_args(argv)
    overrides = None
    if args.detect_threshold is not None:
        overrides = {"detect_threshold": args.detect_threshold}
    run_batch(args.folder, args.out, config_file=args.config,
              recursive=args.recursive, save_overlays=not args.no_overlays,
              hash_max_bytes=(8 << 20) if args.quick_hash else None,
              overrides=overrides)


if __name__ == "__main__":
    main()
