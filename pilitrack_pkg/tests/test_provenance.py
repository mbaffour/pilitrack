"""Reproducibility: config round-trip, fingerprint, manifest."""
import json

import pytest

from pilitrack.config import AcquisitionConfig
from pilitrack import provenance


def test_config_dict_roundtrip():
    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=43.3,
                            ridge_sigmas=(1.5, 2.25, 3.0),
                            detect_threshold=0.3, min_pilus_length_nm=250.0)
    d = provenance.config_to_dict(cfg)
    assert isinstance(d["ridge_sigmas"], list)
    back = provenance.config_from_dict(d)
    assert back == cfg
    assert isinstance(back.ridge_sigmas, tuple)


def test_config_from_dict_ignores_derived_keys():
    d = {"dt_s": 0.5, "pixel_size_nm": 60.0, "min_pilus_length_px": 999,
         "unknown": 1}
    cfg = provenance.config_from_dict(d)
    assert cfg.dt_s == 0.5 and cfg.pixel_size_nm == 60.0


def test_save_load_config_json(tmp_path):
    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=43.3)
    det = {"detect_threshold": 0.3, "tophat_radius_px": 6.0}
    p = provenance.save_config(cfg, tmp_path / "config.json", detection=det)
    loaded_cfg, loaded_det = provenance.load_config(p)
    assert loaded_cfg == cfg
    assert loaded_det == det


def test_file_fingerprint(tmp_path):
    f = tmp_path / "data.bin"
    f.write_bytes(b"hello world" * 100)
    fp = provenance.file_fingerprint(f)
    assert fp["exists"] and fp["size_bytes"] == 1100
    assert len(fp["sha256"]) == 64
    # partial hash differs from full and records the cap
    fp2 = provenance.file_fingerprint(f, max_bytes=16)
    assert fp2["hash_partial_bytes"] == 16
    assert fp2["sha256"] != fp["sha256"]


def test_software_versions_has_core():
    v = provenance.software_versions()
    assert v["numpy"] and v["scipy"] and v["python"]
    assert "pilitrack" in v


def test_build_and_write_manifest(tmp_path):
    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=43.3)
    man = provenance.build_manifest(
        input_path="<array>", cfg=cfg, meta={"n_timepoints": 5},
        detection={"detect_threshold": 0.3},
        results_summary={"n_cells": 3, "percent_piliated": 66.6},
        qc={"flags": ["saturation: 2% ..."]},
        outputs=["a.csv"], timestamp="2026-01-01T00:00:00+00:00")
    assert man["created_utc"] == "2026-01-01T00:00:00+00:00"
    assert man["acquisition_config"]["dt_s"] == 0.4
    assert man["software"]["numpy"]
    p = provenance.write_manifest(man, tmp_path / "manifest.json")
    reloaded = json.loads(open(p).read())
    assert reloaded["results_summary"]["n_cells"] == 3


def test_json_safe_coerces_nonfinite_to_null():
    """NaN/Inf must become null so the manifest is strict-valid JSON."""
    import json
    import numpy as np
    from pilitrack.provenance import _json_safe
    safe = _json_safe({"a": float("nan"), "b": float("inf"),
                       "c": np.float64("nan"), "d": np.array([1.0, np.nan]),
                       "e": 2.0})
    assert safe["a"] is None and safe["b"] is None and safe["c"] is None
    assert safe["d"] == [1.0, None] and safe["e"] == 2.0
    json.dumps(safe, allow_nan=False)          # strict JSON: must not raise
