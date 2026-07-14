"""Reproducibility: config I/O and a run manifest.

For results to be reproducible a lab needs to know *exactly* how each number was
produced — which file (by content, not just name), which software versions, and
every parameter. ``save_config``/``load_config`` round-trip the full analysis
configuration so a study can standardize one settings file across datasets, and
``build_manifest`` captures the complete provenance record written beside every
run's outputs.

Pure standard library (json, hashlib, platform, importlib.metadata); YAML is
used only if PyYAML happens to be installed and a ``.yaml`` path is given.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path

from .config import AcquisitionConfig

# Packages whose versions materially affect the numbers; recorded in every run.
_TRACKED_PACKAGES = (
    "pilitrack", "numpy", "scipy", "scikit-image", "pandas",
    "nd2", "tifffile", "aicsimageio", "czifile",
)


def software_versions() -> dict:
    """Version of Python and every dependency that can change the results."""
    from importlib.metadata import PackageNotFoundError, version

    out = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in _TRACKED_PACKAGES:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def file_fingerprint(path, *, algo: str = "sha256", max_bytes: int | None = None) -> dict:
    """Content fingerprint of an input file: size, mtime, and a hash.

    ``max_bytes`` hashes only the first N bytes (a fast fingerprint for very
    large movies); ``None`` hashes the whole file for a true content hash.
    """
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    h = hashlib.new(algo)
    remaining = max_bytes if max_bytes is not None else float("inf")
    with open(path, "rb") as f:
        while remaining > 0:
            chunk = f.read(min(1 << 20, int(min(remaining, 1 << 20))))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(),
        f"{algo}": h.hexdigest(),
        "hash_partial_bytes": (None if max_bytes is None else int(max_bytes)),
    }


def config_to_dict(cfg: AcquisitionConfig) -> dict:
    """Plain-JSON dict of an AcquisitionConfig (tuples become lists)."""
    d = asdict(cfg)
    d["ridge_sigmas"] = list(cfg.ridge_sigmas)
    return d


def config_from_dict(d: dict) -> AcquisitionConfig:
    """Rebuild an AcquisitionConfig from a dict, ignoring unknown/derived keys."""
    valid = {f.name for f in fields(AcquisitionConfig)}
    kw = {k: v for k, v in d.items() if k in valid}
    if "ridge_sigmas" in kw and kw["ridge_sigmas"] is not None:
        kw["ridge_sigmas"] = tuple(kw["ridge_sigmas"])
    return AcquisitionConfig(**kw)


def _is_yaml(path) -> bool:
    return str(path).lower().endswith((".yaml", ".yml"))


def save_config(cfg: AcquisitionConfig, path, *, detection: dict | None = None) -> str:
    """Write the analysis config to JSON (or YAML if the path ends .yaml and
    PyYAML is installed). ``detection`` holds the single-channel detector
    parameters (thresholds, radii) so the whole recipe travels together."""
    payload = {"acquisition": config_to_dict(cfg), "detection": detection or {}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if _is_yaml(path):
        try:
            import yaml  # optional
            path.write_text(yaml.safe_dump(payload, sort_keys=False))
            return str(path)
        except Exception:
            path = path.with_suffix(".json")
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def load_config(path) -> tuple[AcquisitionConfig, dict]:
    """Load ``(AcquisitionConfig, detection_params)`` written by ``save_config``.

    Accepts either the ``{"acquisition":..., "detection":...}`` layout or a bare
    config dict."""
    path = Path(path)
    text = path.read_text()
    if _is_yaml(path):
        import yaml
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if "acquisition" in payload:
        return config_from_dict(payload["acquisition"]), payload.get("detection", {})
    return config_from_dict(payload), {}


def build_manifest(
    *,
    input_path,
    cfg: AcquisitionConfig,
    meta: dict | None = None,
    detection: dict | None = None,
    roi=None,
    frames=None,
    position: int = 0,
    results_summary: dict | None = None,
    qc: dict | None = None,
    outputs: list | None = None,
    hash_max_bytes: int | None = None,
    timestamp: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Assemble the full provenance record for one analysis run.

    ``timestamp`` may be supplied (UTC ISO string); otherwise it is stamped now.
    ``hash_max_bytes`` limits the input hash for speed on huge movies.
    """
    if timestamp is None:
        timestamp = datetime.now(timezone.utc).isoformat()
    return {
        "tool": "pilitrack",
        "created_utc": timestamp,
        "input": file_fingerprint(input_path, max_bytes=hash_max_bytes)
        if input_path not in (None, "<array>") else {"path": str(input_path)},
        "software": software_versions(),
        "acquisition_config": config_to_dict(cfg),
        "detection_params": detection or {},
        "selection": {"roi": list(roi) if roi else None,
                      "frames": str(frames) if frames is not None else None,
                      "position": position},
        "movie_meta": _json_safe(meta) if meta else None,
        "results_summary": _json_safe(results_summary) if results_summary else None,
        "qc": _json_safe(qc) if qc else None,
        "outputs": list(outputs) if outputs else [],
        "extra": extra or {},
    }


def write_manifest(manifest: dict, path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return str(path)


def _json_safe(obj):
    """Best-effort conversion of numpy / non-serializable values to JSON types.

    Non-finite floats (NaN/Inf) are coerced to ``None`` — bare ``NaN``/``Infinity``
    tokens are invalid JSON and are rejected by strict parsers (JS ``JSON.parse``,
    R ``jsonlite``, ``jq``), which would silently break the reproducibility record
    (e.g. ``percent_piliated = NaN`` on a pili-only movie with no cells)."""
    import math
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
