"""A trainable pilus detector — few-shot, CPU-friendly, drop-in.

The built-in ridge filter is fixed; on faint or cluttered real data a *learned*
detector does better, and it improves as the lab sends more labels. This is that
detector, in the ilastik spirit but in-package: a bank of multiscale image
features + a random-forest pixel classifier that outputs a per-pixel pilus
**probability map**, which feeds the pipeline's existing ``pilus_prob_stack``
seam. It trains on the ``(image, pilus_mask)`` pairs that
``dataset.save_training_bundle`` writes — a handful of hand-labelled frames is
enough, and it runs on a CPU.

    from pilitrack.ml import train_from_dataset, predict_prob_stack
    model = train_from_dataset("training/")          # learns from the bundles
    prob = predict_prob_stack(model, fluor_stack)    # (T, Y, X) in [0, 1]
    res = analyze_movie(fluor, cells, cfg, pilus_prob_stack=prob)

``scikit-learn`` is an optional dependency (``pip install -e '.[ml]'``); it is
imported lazily so the rest of the package installs without it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- #
# Features — designed for thin bright ridges (pili / vessels / filaments)
# --------------------------------------------------------------------------- #
DEFAULT_SIGMAS = (1.0, 2.0, 4.0)


def feature_names(sigmas=DEFAULT_SIGMAS) -> list:
    names = ["raw"]
    for s in sigmas:
        names += [f"gauss_{s}", f"grad_{s}", f"log_{s}",
                  f"hess_lo_{s}", f"hess_hi_{s}"]
    names.append("sato")
    return names


def feature_stack(img, sigmas=DEFAULT_SIGMAS) -> np.ndarray:
    """``(H, W, F)`` per-pixel feature stack: raw, and per-scale Gaussian,
    gradient magnitude, Laplacian-of-Gaussian, the two Hessian eigenvalues
    (ridge strength), plus a multiscale Sato ridge response."""
    from scipy import ndimage as ndi
    from skimage.feature import hessian_matrix, hessian_matrix_eigvals
    from skimage.filters import gaussian, sato

    img = np.asarray(img, dtype=np.float32)
    # normalize per image so a trained model transfers across intensity scales
    lo, hi = np.percentile(img, [1, 99.5])
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

    feats = [img]
    for s in sigmas:
        feats.append(gaussian(img, s))
        feats.append(ndi.gaussian_gradient_magnitude(img, s))
        feats.append(ndi.gaussian_laplace(img, s))
        try:
            H = hessian_matrix(img, sigma=s, use_gaussian_derivatives=True)
        except TypeError:  # pragma: no cover - older skimage
            H = hessian_matrix(img, sigma=s)
        lo_e, hi_e = hessian_matrix_eigvals(H)
        feats.append(lo_e)
        feats.append(hi_e)
    feats.append(sato(img, sigmas=sigmas, black_ridges=False))
    return np.stack(feats, axis=-1).astype(np.float32)


# --------------------------------------------------------------------------- #
# Train / predict
# --------------------------------------------------------------------------- #
def train_pilus_detector(images, masks, *, sigmas=DEFAULT_SIGMAS,
                         n_estimators: int = 150, neg_per_pos: int = 3,
                         max_pixels: int = 300_000, seed: int = 0):
    """Train a random-forest pixel classifier on ``(image, mask)`` pairs.

    ``masks`` are boolean/0-255 pilus masks. Negative pixels are subsampled to
    ``neg_per_pos`` per positive to balance classes; the total training set is
    capped at ``max_pixels``. Returns a model dict (classifier + feature config).
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception as exc:  # pragma: no cover
        raise ImportError("The ML detector needs scikit-learn "
                          "(`pip install -e '.[ml]'`).") from exc

    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for img, mask in zip(images, masks):
        F = feature_stack(img, sigmas)
        m = np.asarray(mask) > 0
        pos = np.argwhere(m)
        if pos.shape[0] == 0:
            continue
        neg = np.argwhere(~m)
        n_neg = min(neg.shape[0], pos.shape[0] * neg_per_pos)
        neg = neg[rng.choice(neg.shape[0], n_neg, replace=False)]
        idx = np.vstack([pos, neg])
        X_parts.append(F[idx[:, 0], idx[:, 1]])
        y_parts.append(np.r_[np.ones(pos.shape[0]), np.zeros(n_neg)])
    if not X_parts:
        raise ValueError("no positive pixels in any mask — nothing to train on")
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    if X.shape[0] > max_pixels:
        sel = rng.choice(X.shape[0], max_pixels, replace=False)
        X, y = X[sel], y[sel]

    clf = RandomForestClassifier(
        n_estimators=n_estimators, n_jobs=-1, random_state=seed,
        class_weight="balanced", min_samples_leaf=2)
    clf.fit(X, y)
    return {"clf": clf, "sigmas": tuple(sigmas),
            "feature_names": feature_names(sigmas), "n_train": int(X.shape[0])}


def predict_prob(model, image) -> np.ndarray:
    """Per-pixel pilus probability ``(H, W)`` in [0, 1]."""
    F = feature_stack(image, model["sigmas"])
    H, W, Fd = F.shape
    p = model["clf"].predict_proba(F.reshape(-1, Fd))[:, 1]
    return p.reshape(H, W).astype(np.float32)


def predict_prob_stack(model, stack) -> np.ndarray:
    """Probability map for a whole ``(T, Y, X)`` stack -> feeds
    ``analyze_movie(..., pilus_prob_stack=...)``."""
    stack = np.asarray(stack)
    return np.stack([predict_prob(model, stack[t]) for t in range(stack.shape[0])])


# --------------------------------------------------------------------------- #
# Train straight from a labelled dataset folder, and persistence
# --------------------------------------------------------------------------- #
def train_from_dataset(root, *, target="pili", **kw):
    """Train on all bundles under ``root`` (written by ``dataset.save_training_bundle``).

    ``target='pili'`` learns the pilus detector from ``pili_masks/``; ``'cell'``
    learns a cell classifier from ``cell_masks/`` (boolean foreground)."""
    import tifffile
    from .dataset import collect_dataset

    df = collect_dataset(root)
    col = "pili_mask" if target == "pili" else "cell_mask"
    df = df[df[col].notna()]
    if df.empty:
        raise ValueError(f"no {col} files found under {root}")
    images = [tifffile.imread(p) for p in df["image"]]
    masks = [tifffile.imread(p) > 0 for p in df[col]]
    model = train_pilus_detector(images, masks, **kw)
    model["target"] = target
    model["n_frames"] = int(len(df))
    return model


def save_model(model, path) -> str:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, str(path))
    return str(path)


def load_model(path):
    import joblib
    return joblib.load(str(path))
