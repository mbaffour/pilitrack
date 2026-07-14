"""Acquisition and analysis configuration.

All physical parameters live here so the rest of the pipeline is unit-aware.
Set ``dt_s`` and ``pixel_size_nm`` to match your microscope once known; the
defaults below are reasonable placeholders for P. aeruginosa T4P imaging
(~1 s frame interval, 65 nm/px is typical for a 100x objective + sCMOS).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcquisitionConfig:
    # --- microscope / movie ---
    dt_s: float = 1.0                 # seconds between frames
    pixel_size_nm: float = 65.0       # nm per pixel
    window_s: float = 30.0            # analysis window length

    # --- pilus detection ---
    ridge_sigmas: tuple = (1.0, 1.5, 2.0)   # scales for the tubeness filter (px)
    detect_threshold: float = 0.20    # relative threshold on the ridge response
    pilus_prob_threshold: float = 0.50   # threshold on an ilastik probability map
    pilus_prob_channel: int = 1          # which class channel is "pilus"
    omnipose_model: str = "bact_phase_omni"  # bacterial phase-contrast model
    min_pilus_length_nm: float = 300.0  # ignore filaments shorter than this
    max_pilus_length_nm: float = 10000.0  # reject components longer than this (artifacts)
    max_branch_endpoints: int = 4       # reject branchy (non-filament) skeletons
    base_search_radius_px: float = 6.0  # how close a filament base must sit to a cell

    # --- temporal linking ---
    max_base_jump_px: float = 5.0     # max frame-to-frame move of a pilus base
    max_gap_frames: int = 1           # bridge this many missed frames
    min_piliation_frames: int = 3     # a cell must show a pilus this many frames
    linker: str = "greedy"            # "greedy" or "lap" (optimal Hungarian)

    # --- kinetics segmentation ---
    smoothing_window: int = 3         # median window on length(t) before slopes
    min_phase_frames: int = 2         # a phase must persist this many frames
    velocity_sign_eps_nm_s: float = 20.0  # |v| below this counts as a dwell

    @property
    def min_pilus_length_px(self) -> float:
        return self.min_pilus_length_nm / self.pixel_size_nm

    @property
    def n_frames_window(self) -> int:
        return int(round(self.window_s / self.dt_s))
