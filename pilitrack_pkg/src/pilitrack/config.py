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
    hysteresis_low_frac: float = 1.0  # double-threshold detection: keep ridge
                                      # pixels down to frac*detect_threshold when
                                      # connected to a strong core, recovering
                                      # faint distal ends. Opt-in (set e.g. 0.5)
                                      # for faint-pilus data; 1.0 = single cut,
                                      # the safe default (won't merge pili into
                                      # nearby cell-edge ridge on artifact frames)
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
    # Continuity terms added to the base-distance assignment cost so a pilus keeps
    # its identity through a crossing: a candidate emerging at a very different
    # angle, or whose length jumped faster than physically possible, is penalized.
    # Both only re-rank matches already allowed by max_base_jump_px (0 disables).
    link_orientation_weight_px: float = 2.0   # cost of a full direction reversal/2
    link_length_weight: float = 1.0           # cost per px of impossible dL
    max_velocity_nm_s: float = 2000.0  # above published T4P retraction (~0.6-2 um/s)

    # --- kinetics segmentation ---
    smoothing_window: int = 3         # median window on length(t) before slopes
    min_phase_frames: int = 2         # a phase must persist this many frames
    velocity_sign_eps_nm_s: float = 20.0  # |v| below this counts as a dwell

    def __post_init__(self):
        # dt_s and pixel_size_nm are divisors throughout (velocities, px<->nm,
        # savgol delta). A zero/negative value would surface as an opaque
        # ZeroDivisionError deep in the kinetics; fail early and clearly instead.
        if not (self.dt_s > 0):
            raise ValueError(f"dt_s must be > 0 (got {self.dt_s}); pass --dt or "
                             "set the frame interval.")
        if not (self.pixel_size_nm > 0):
            raise ValueError(f"pixel_size_nm must be > 0 (got {self.pixel_size_nm}).")

    @property
    def min_pilus_length_px(self) -> float:
        return self.min_pilus_length_nm / self.pixel_size_nm

    @property
    def n_frames_window(self) -> int:
        return int(round(self.window_s / self.dt_s))
