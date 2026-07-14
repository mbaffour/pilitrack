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


def _scaled_sigmas(model, pixel_size_nm):
    """Scale the model's feature sigmas so they probe the same *physical* scale
    on a movie whose pixel size differs from training. The feature COUNT is
    unchanged (only the sigma values), so the classifier input still lines up."""
    base = tuple(model["sigmas"])
    train_px = model.get("pixel_size_nm")
    if not train_px or not pixel_size_nm or pixel_size_nm <= 0:
        return base
    factor = float(train_px) / float(pixel_size_nm)
    if abs(factor - 1.0) < 0.05:
        return base
    return tuple(max(0.5, s * factor) for s in base)


def predict_prob(model, image, pixel_size_nm=None) -> np.ndarray:
    """Per-pixel pilus probability ``(H, W)`` in [0, 1]. Pass ``pixel_size_nm``
    to rescale the features to the training scale when it differs."""
    F = feature_stack(image, _scaled_sigmas(model, pixel_size_nm))
    H, W, Fd = F.shape
    clf = model["clf"]
    proba = clf.predict_proba(F.reshape(-1, Fd))
    # select the positive-class column robustly: a forest trained on a single
    # class returns a 1-column proba, so a hardcoded [:, 1] would IndexError.
    classes = list(getattr(clf, "classes_", []))
    if 1 in classes:
        p = proba[:, classes.index(1)]
    else:
        p = np.zeros(proba.shape[0], dtype=np.float32)
    return p.reshape(H, W).astype(np.float32)


def predict_prob_stack(model, stack, pixel_size_nm=None) -> np.ndarray:
    """Probability map for a whole ``(T, Y, X)`` stack -> feeds
    ``analyze_movie(..., pilus_prob_stack=...)``."""
    stack = np.asarray(stack)
    return np.stack([predict_prob(model, stack[t], pixel_size_nm)
                     for t in range(stack.shape[0])])


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
    # Record the training pixel size so predict_prob can rescale features for a
    # movie at a different scale. The features use fixed pixel sigmas, so mixing
    # bundles of very different pixel sizes blurs the model — warn if so.
    pxs = [float(v) for v in df.get("pixel_size_nm", []) if v and float(v) > 0]
    if pxs:
        model["pixel_size_nm"] = float(np.median(pxs))
        if max(pxs) / min(pxs) > 1.1:
            import warnings
            warnings.warn(
                f"training bundles span pixel sizes {min(pxs):.1f}-{max(pxs):.1f} "
                "nm/px; features are computed at fixed pixel sigmas, so mixing "
                "scales may blur the model — prefer one pixel size per model.")
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


def resolve_model(model):
    """Accept a model dict or a path to a saved model; return the model dict."""
    if model is None or isinstance(model, dict):
        return model
    return load_model(model)


# --------------------------------------------------------------------------- #
# Bootstrap: a usable detector with NO hand labels, trained on synthetic pili.
# A starting point that already beats the fixed ridge filter on faint pili;
# retrain on real labels as they arrive.
# --------------------------------------------------------------------------- #
def _synthetic_frame(rng, H=140, W=140, n_pili=7, n_cells=3):
    from scipy.ndimage import gaussian_filter, binary_dilation
    img = np.full((H, W), 30.0)
    mask = np.zeros((H, W), bool)
    yy, xx = np.ogrid[:H, :W]
    for _ in range(n_cells):
        cy, cx = rng.uniform(20, H - 20), rng.uniform(20, W - 20)
        img[((yy - cy) ** 2 + (xx - cx) ** 2) <= rng.uniform(5, 9) ** 2] += rng.uniform(1800, 4200)
    for _ in range(n_pili):
        y0, x0 = rng.uniform(12, H - 12), rng.uniform(12, W - 12)
        ang, L = rng.uniform(0, 2 * np.pi), rng.uniform(12, 34)
        ts = np.linspace(0, L, int(L * 3))
        ys = np.clip((y0 + ts * np.sin(ang)).astype(int), 0, H - 1)
        xs = np.clip((x0 + ts * np.cos(ang)).astype(int), 0, W - 1)
        img[ys, xs] += rng.uniform(35, 130)          # varied faintness
        mask[ys, xs] = True
    img = gaussian_filter(img, 1.0)
    img = rng.poisson(np.clip(img, 0, None)) + rng.normal(0, 8, (H, W))
    return np.clip(img, 0, None).astype(np.float32), binary_dilation(mask)


def synthetic_training_data(n_frames: int = 16, seed: int = 0):
    """(images, masks) of synthetic bright filaments on noisy cell backgrounds."""
    rng = np.random.default_rng(seed)
    pairs = [_synthetic_frame(rng) for _ in range(n_frames)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


def _cache_dir() -> Path:
    d = Path.home() / ".pilitrack" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def bootstrap_synthetic_model(*, n_frames: int = 16, n_estimators: int = 120,
                              seed: int = 0, cache: bool = True):
    """A ready-to-use pilus detector trained on synthetic data — **no hand
    labels required**. Cached to ``~/.pilitrack/models`` so it trains once
    (a few seconds) and reloads instantly. Retrain on real labels via
    ``train_from_dataset`` when you have them."""
    path = _cache_dir() / f"bootstrap_rf_n{n_frames}_e{n_estimators}_s{seed}.joblib"
    if cache and path.exists():
        return load_model(path)
    imgs, masks = synthetic_training_data(n_frames, seed=seed)
    model = train_pilus_detector(imgs, masks, n_estimators=n_estimators, seed=seed)
    model["source"] = "synthetic-bootstrap"
    if cache:
        save_model(model, path)
    return model
