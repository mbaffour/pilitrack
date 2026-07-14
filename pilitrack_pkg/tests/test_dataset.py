"""Training-ready storage: pilus masks, bundles, and dataset collection."""
import json
import os

import numpy as np
import pytest

from pilitrack.config import AcquisitionConfig
from pilitrack.annotate import (Annotations, ManualPilus, pili_mask,
                                save_annotations, load_annotations)
from pilitrack.dataset import save_training_bundle, collect_dataset, dataset_summary


def test_pili_mask_rasterizes_and_dilates():
    ann = Annotations(manual_pili=[ManualPilus(0, [[10, 5], [10, 20]])])
    m = pili_mask(ann, 0, (30, 30), width_px=3)
    assert m.dtype == bool and m[10, 12]
    assert m.sum() > 16                      # dilated beyond the 16-px centerline


def test_pili_mask_is_frame_specific():
    ann = Annotations(manual_pili=[ManualPilus(2, [[5, 5], [5, 15]])])
    assert pili_mask(ann, 0, (20, 20)).sum() == 0
    assert pili_mask(ann, 2, (20, 20)).sum() > 0


def test_save_training_bundle(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    rng = np.random.default_rng(0)
    stack = (rng.random((4, 40, 40)) * 1000).astype(np.uint16)
    cells = np.zeros((4, 40, 40), np.int32)
    cells[:, 5:10, 5:10] = 1
    ann = Annotations(manual_pili=[ManualPilus(0, [[20, 5], [20, 30]]),
                                   ManualPilus(2, [[10, 5], [10, 25]])])
    cfg = AcquisitionConfig(pixel_size_nm=43.3, dt_s=0.4)
    meta = save_training_bundle(tmp_path / "m1", stack=stack, annotations=ann,
                                cfg=cfg, movie_path="m.nd2", cell_labels=cells)
    b = tmp_path / "m1"
    assert (b / "images" / "frame_000.tif").exists()
    assert (b / "pili_masks" / "frame_000.tif").exists()
    assert (b / "cell_masks" / "frame_002.tif").exists()
    assert (b / "annotations.json").exists()
    md = json.loads((b / "metadata.json").read_text())
    assert md["pixel_size_nm"] == 43.3 and md["labeled_frames"] == [0, 2]
    assert md["has_cell_masks"] is True
    # image and mask round-trip
    assert tifffile.imread(str(b / "images" / "frame_000.tif")).shape == (40, 40)
    assert tifffile.imread(str(b / "pili_masks" / "frame_000.tif")).max() == 255


def test_collect_dataset_indexes_frames(tmp_path):
    pytest.importorskip("tifffile")
    rng = np.random.default_rng(1)
    stack = (rng.random((3, 30, 30)) * 500).astype(np.uint16)
    for i in range(2):
        ann = Annotations(manual_pili=[ManualPilus(0, [[10, 5], [10, 20]])])
        save_training_bundle(tmp_path / f"b{i}", stack=stack, annotations=ann,
                             movie_path=f"m{i}.tif")
    df = collect_dataset(tmp_path)
    assert len(df) == 2 and set(df["bundle"]) == {"b0", "b1"}
    assert df["image"].map(os.path.exists).all()
    assert df["pili_mask"].map(os.path.exists).all()
    s = dataset_summary(tmp_path)
    assert s["n_bundles"] == 2 and s["n_labeled_frames"] == 2


def test_annotations_meta_roundtrip(tmp_path):
    ann = Annotations(manual_pili=[], meta={"pixel_size_nm": 43.3, "roi": [0, 100, 0, 100]})
    p = save_annotations(ann, tmp_path / "a.json")
    loaded, _ = load_annotations(p)
    assert loaded.meta["pixel_size_nm"] == 43.3
    assert loaded.meta["roi"] == [0, 100, 0, 100]
