"""Output-side reproducibility: git commit, package tracking, output checksums,
determinism recording."""
import hashlib

import numpy as np

from pilitrack import provenance
from pilitrack.config import AcquisitionConfig


def test_software_versions_records_git_and_ml_packages():
    sv = provenance.software_versions()
    assert "git_commit" in sv                       # None off a checkout, str on one
    for pkg in ("scikit-learn", "matplotlib", "numpy", "scipy"):
        assert pkg in sv


def test_write_checksums_is_sha256sum_compatible(tmp_path):
    a = tmp_path / "a.csv"
    a.write_text("x,y\n1,2\n")
    sub = tmp_path / "qc"
    sub.mkdir()
    b = sub / "b.png"
    b.write_bytes(b"\x89PNG fake")
    provenance.write_checksums([str(a), str(b)], tmp_path / "checksums.sha256",
                               base=tmp_path)
    lines = (tmp_path / "checksums.sha256").read_text().strip().splitlines()
    assert len(lines) == 2
    for line in lines:                              # '<hex>  <relpath>'
        digest, rel = line.split("  ", 1)
        assert hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest() == digest
    # nested paths use forward slashes (portable across OSes)
    assert "qc/b.png" in (tmp_path / "checksums.sha256").read_text()


def test_manifest_carries_output_checksums_and_determinism(tmp_path):
    cfg = AcquisitionConfig(dt_s=0.4, pixel_size_nm=65.0)
    f = tmp_path / "pili.csv"
    f.write_text("id\n1\n")
    man = provenance.build_manifest(input_path="<array>", cfg=cfg,
                                    outputs=[str(f)], out_dir=tmp_path)
    assert man["output_checksums"]
    assert man["output_checksums"][0]["path"] == "pili.csv"
    assert len(man["output_checksums"][0]["sha256"]) == 64
    assert set(man["determinism"]) >= {"seed", "thread_env"}


def test_set_deterministic_makes_numpy_repeatable():
    provenance.set_deterministic(123)
    a = np.random.rand(6)
    provenance.set_deterministic(123)
    b = np.random.rand(6)
    assert np.allclose(a, b)
