"""Cellpose backend for cell-body instance segmentation.

Cellpose does deep-learning *instance* segmentation and separates touching
cells, which is what accurate per-cell counts and % piliated need — the honest
weak point of the built-in morphological mask. Cellpose is actively maintained
(Omnipose, the other option, is mid-refactor upstream), so it is the recommended
route for dense fields. A GPU helps but CPU works for a handful of frames.

    pip install cellpose        # pulls torch

Generalist models: ``cyto3`` (Cellpose 3) or ``cpsam`` (Cellpose-SAM, v4). For
fluorescent labelled cells the generalist model on the single channel works;
tune ``diameter`` to the cell size in pixels (``None`` lets Cellpose estimate).
"""
from __future__ import annotations

import numpy as np

from ..config import AcquisitionConfig


def _load_model(model_type: str, gpu: bool):
    try:
        from cellpose import models  # published entry point
    except Exception as exc:  # pragma: no cover - external optional install
        raise ImportError(
            "Cellpose not available. `pip install cellpose` (pulls torch). "
            "See cellpose.readthedocs.io for the model list."
        ) from exc
    # Cellpose 3.x: CellposeModel(gpu=, model_type=); 4.x (SAM) drops model_type.
    try:
        return models.CellposeModel(gpu=gpu, model_type=model_type)
    except TypeError:  # pragma: no cover - version-dependent signature
        return models.CellposeModel(gpu=gpu)


def make_cellpose_segmenter(model_type: str = "cyto3", gpu: bool = False,
                            diameter=None, flow_threshold: float = 0.4,
                            cellprob_threshold: float = 0.0, channels=(0, 0)):
    """Return a ``segment_fn(frame, cfg) -> int label image`` for the pipeline.

    The model is loaded once and closed over (not re-instantiated per frame).
    Pass the returned function as ``analyze_movie(..., segment_fn=...)`` or
    materialize a stack with :func:`segment_stack_cellpose`.
    """
    model = _load_model(model_type, gpu)

    def segment_fn(frame: np.ndarray, _cfg: AcquisitionConfig) -> np.ndarray:
        out = model.eval(np.asarray(frame), diameter=diameter, channels=list(channels),
                         flow_threshold=flow_threshold,
                         cellprob_threshold=cellprob_threshold)
        masks = out[0]  # (masks, flows, styles[, diams]) across versions
        return np.asarray(masks, dtype=int)

    return segment_fn


def segment_stack_cellpose(cell_stack: np.ndarray, cfg: AcquisitionConfig,
                           gpu: bool = False, **kw) -> np.ndarray:
    """Segment a whole ``(T, H, W)`` stack -> ``(T, H, W)`` integer label stack,
    for ``analyze_movie(..., cell_label_stack=...)``."""
    seg = make_cellpose_segmenter(gpu=gpu, **kw)
    return np.stack([seg(cell_stack[t], cfg) for t in range(cell_stack.shape[0])])
