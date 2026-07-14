"""pilitrack: quantify type IV pili dynamics from labelled time-lapse movies."""
from .config import AcquisitionConfig
from .kinetics import segment_trace, summarize_pilus, Phase
from .pipeline import analyze_movie, detect_and_link, summarize
from .io import load_movie, load_array, config_from_meta, config_from_nd2
from .analyze import analyze_file, pilus_length_timeseries
from .batch import run_batch
from .annotate import Annotations, ManualPilus, apply_annotations, pili_mask
from .dataset import save_training_bundle, collect_dataset
from .viewer import launch_viewer, launch_annotator
from . import (synth, io, singlechannel, qc, provenance, annotate, viewer,
               validate, figures, stats, dataset)

__all__ = [
    "AcquisitionConfig",
    "segment_trace",
    "summarize_pilus",
    "Phase",
    "analyze_movie",
    "detect_and_link",
    "summarize",
    # real-data entry points
    "load_movie",
    "load_array",
    "config_from_meta",
    "config_from_nd2",
    "analyze_file",
    "pilus_length_timeseries",
    "run_batch",
    # hand-labeling
    "Annotations",
    "ManualPilus",
    "apply_annotations",
    "pili_mask",
    "save_training_bundle",
    "collect_dataset",
    "launch_viewer",
    "launch_annotator",
    # modules
    "synth",
    "io",
    "singlechannel",
    "qc",
    "provenance",
    "annotate",
    "viewer",
    "validate",
    "figures",
    "stats",
    "dataset",
]
__version__ = "0.1.0"
