"""Omnipose backend for cell-body segmentation.

Omnipose does morphology-independent *instance* segmentation and separates
touching rod-shaped bacteria, which the per-cell metrics depend on. Install:

    pip install omnipose        # pulls torch; a GPU helps but CPU works

Note: Omnipose is mid-refactor upstream (splitting into ``omnipose`` +
``omnitools``); the model-loading import path may shift. The call below uses the
stable published API. Verify against https://omnipose.readthedocs.io for your
installed version, and use the bacterial phase model ``bact_phase_omni`` for
Pseudomonas imaged in phase contrast.
"""
from __future__ import annotations

import numpy as np

from ..config import AcquisitionConfig


def _load_model(cfg: AcquisitionConfig, gpu: bool):
    try:
        from cellpose_omni import models  # stable published entry point
    except Exception as exc:  # pragma: no cover - depends on external install
        raise ImportError(
            "Omnipose not available. `pip install omnipose`. If your version "
            "moved the API, import the model class from the current package "
            "(see omnipose.readthedocs.io)."
        ) from exc
    return models.CellposeModel(gpu=gpu, model_type=cfg.omnipose_model)


def make_omnipose_segmenter(cfg: AcquisitionConfig, gpu: bool = False,
                            diameter=None, mask_threshold: float = -1.0):
    """Return a ``segment_fn(frame, cfg) -> label image`` for analyze_movie.

    The model is loaded once and closed over, so it is not re-instantiated for
    every frame.
    """
    model = _load_model(cfg, gpu)

    def segment_fn(frame: np.ndarray, _cfg: AcquisitionConfig) -> np.ndarray:
        masks, _flows, _styles = model.eval(
            frame, channels=[0, 0], omni=True,
            diameter=diameter, mask_threshold=mask_threshold,
        )
        return np.asarray(masks, dtype=int)

    return segment_fn


def segment_stack_omnipose(cell_stack: np.ndarray, cfg: AcquisitionConfig,
                           gpu: bool = False, **kw) -> np.ndarray:
    """Segment a whole (T, H, W) stack -> (T, H, W) integer label stack.

    Convenient when you would rather run Omnipose once up front and pass the
    result to ``analyze_movie(..., cell_label_stack=...)``.
    """
    seg = make_omnipose_segmenter(cfg, gpu=gpu, **kw)
    return np.stack([seg(cell_stack[t], cfg) for t in range(cell_stack.shape[0])])
