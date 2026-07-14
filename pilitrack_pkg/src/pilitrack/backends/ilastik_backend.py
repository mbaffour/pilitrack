"""ilastik backend for pilus detection via Pixel Classification.

Workflow (ilastik >= 1.4):
  1. In the ilastik GUI, Pixel Classification workflow: load a representative
     fluorescence movie, add two labels (pilus / background), paint a few brush
     strokes on thin filaments vs background, let the random forest train live,
     save the project as ``pili.ilp``.
  2. Batch-predict headless on all movies to export probability maps (this
     module shells out to ``run_ilastik.sh``).
  3. Load the pilus-probability channel and pass the stack to
     ``analyze_movie(..., pilus_prob_stack=...)``.

A trained classifier on a feature stack handles faint, low-SNR filaments far
better than a fixed ridge filter, and moves the detection tuning from code
constants into painted examples.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ..config import AcquisitionConfig


def run_ilastik_headless(
    ilastik_executable: str,
    project_ilp: str,
    input_paths: list[str],
    output_dir: str,
    export_source: str = "Probabilities",
    output_format: str = "hdf5",
) -> list[str]:
    """Run ilastik Pixel Classification headless and return output file paths.

    ``ilastik_executable`` is the ``run_ilastik.sh`` / ``ilastik.exe`` shipped
    with the ilastik download. One call can batch many inputs.
    """
    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    fmt = "{dataset_dir}/{nickname}_probs.h5" if output_format == "hdf5" \
        else "{dataset_dir}/{nickname}_probs.tiff"
    cmd = [
        ilastik_executable, "--headless",
        f"--project={project_ilp}",
        f"--export_source={export_source}",
        f"--output_format={output_format}",
        f"--output_filename_format={outdir}/{{nickname}}_probs."
        + ("h5" if output_format == "hdf5" else "tiff"),
        *input_paths,
    ]
    subprocess.run(cmd, check=True)
    suffix = "_probs.h5" if output_format == "hdf5" else "_probs.tiff"
    return [str(outdir / (Path(p).stem + suffix)) for p in input_paths]


def load_probability_h5(path: str, dataset: str = "exported_data") -> np.ndarray:
    """Load an ilastik HDF5 probability export as (T, H, W, C) or (H, W, C)."""
    import h5py  # optional dependency
    with h5py.File(path, "r") as f:
        return np.asarray(f[dataset])


def pilus_channel_stack(prob: np.ndarray, cfg: AcquisitionConfig) -> np.ndarray:
    """Extract the pilus-probability channel as a (T, H, W) stack in [0, 1].

    ilastik writes probabilities in the last axis; the channel index is set by
    the label order in your project (``cfg.pilus_prob_channel``).
    """
    prob = np.asarray(prob, dtype=float)
    c = cfg.pilus_prob_channel
    if prob.ndim == 4:            # (T, H, W, C)
        return prob[..., c]
    if prob.ndim == 3 and prob.shape[-1] <= 8:   # (H, W, C) single frame
        return prob[..., c][None, ...]
    raise ValueError(f"Unexpected probability array shape {prob.shape}")
