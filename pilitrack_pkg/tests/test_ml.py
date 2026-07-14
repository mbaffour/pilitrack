"""Trainable ML detector (scikit-learn) — skipped if sklearn absent."""
import numpy as np
import pytest

pytest.importorskip("sklearn")

from pilitrack.ml import (feature_stack, train_pilus_detector, predict_prob,
                          predict_prob_stack, save_model, load_model,
                          train_from_dataset)


def _frame(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((40, 40), 30.0)
    img[20, 5:35] += 200.0                       # a bright horizontal filament
    img = img + rng.normal(0, 5, (40, 40))
    mask = np.zeros((40, 40), bool)
    mask[19:22, 5:35] = True
    return img.astype(np.float32), mask


def test_feature_stack_shape():
    F = feature_stack(np.zeros((20, 20)))
    assert F.shape[:2] == (20, 20) and F.shape[2] >= 10


def test_train_predict_higher_on_filament():
    frames = [_frame(i) for i in range(3)]
    model = train_pilus_detector([i for i, _ in frames], [m for _, m in frames],
                                 n_estimators=30)
    img, _ = _frame(9)
    p = predict_prob(model, img)
    assert p.shape == (40, 40) and p.min() >= 0 and p.max() <= 1
    assert p[20, 20] > p[5, 5]                   # on the filament vs off it


def test_predict_prob_stack():
    frames = [_frame(i) for i in range(2)]
    model = train_pilus_detector([i for i, _ in frames], [m for _, m in frames],
                                 n_estimators=20)
    stack = np.stack([_frame(5)[0], _frame(6)[0]])
    P = predict_prob_stack(model, stack)
    assert P.shape == (2, 40, 40)


def test_save_load_roundtrip(tmp_path):
    frames = [_frame(i) for i in range(2)]
    model = train_pilus_detector([i for i, _ in frames], [m for _, m in frames],
                                 n_estimators=10)
    p = save_model(model, tmp_path / "m.joblib")
    m2 = load_model(p)
    img, _ = _frame(3)
    assert np.allclose(predict_prob(model, img), predict_prob(m2, img))


def test_train_from_dataset(tmp_path):
    pytest.importorskip("tifffile")
    from pilitrack.dataset import save_training_bundle
    from pilitrack.annotate import Annotations, ManualPilus
    stack = np.stack([_frame(i)[0] for i in range(3)]).astype(np.uint16)
    ann = Annotations(manual_pili=[ManualPilus(t, [[20, 5], [20, 34]]) for t in range(3)])
    save_training_bundle(tmp_path / "b", stack=stack, annotations=ann, movie_path="m.tif")
    model = train_from_dataset(tmp_path, n_estimators=15)
    assert model["target"] == "pili" and model["n_frames"] == 3


def test_train_raises_without_positives():
    img = np.zeros((20, 20), np.float32)
    with pytest.raises(ValueError):
        train_pilus_detector([img], [np.zeros((20, 20), bool)])
