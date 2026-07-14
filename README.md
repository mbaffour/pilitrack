# pilitrack

Quantify **type IV pili dynamics** in *Pseudomonas* (or any T4P system) from
labelled two-channel time-lapse microscopy.

> **New here / setting up a lab computer?** See **[SETUP.md](SETUP.md)** — install
> Miniconda, double-click **Install pilitrack**, then **Start pilitrack (browser)**.
> **Where is everything?** See **[MAP.md](MAP.md)**.

Assumes pili are **fluorescently labelled** — e.g. a PilA cysteine knock-in
(PilA-A86C in *P. aeruginosa*) tagged with a thiol-reactive maleimide dye such
as Alexa488-mal — imaged alongside a cell-body channel (phase or membrane
label). Pili are ~6 nm wide, below the diffraction limit, so unlabelled cells
cannot give per-pilus length or velocity.

## What it measures (per 30 s window, configurable)

| Requirement | Where |
|---|---|
| Detect cells | `detect.segment_cells` |
| Detect cells with pili / % piliated | `pipeline.analyze_movie` → `population` |
| Number of pili per cell | `cell` DataFrame → `n_pili` |
| Max pilus length | `pilus`/`cell` DataFrames → `max_length_nm` |
| Length of individual pili over time | per-track `length(t)` |
| Extension & retraction velocity | `kinetics.summarize_pilus` |

## Pipeline

```
two-channel stack
  ├─ segment_cells      (Otsu default; swap in Omnipose for real rods)
  ├─ detect_pili        (Sato tubeness → threshold → skeletonize)
  ├─ extract_filaments  (geodesic skeleton length; reject branchy blobs)
  ├─ associate_to_cells (base within N px of a cell → assign)
  ├─ link_tracks        (base-anchored greedy linking → length(t) per pilus)
  └─ summarize_pilus    (Savitzky-Golay derivative → extension/dwell/retraction
                         phases → per-phase velocities)
```

## Install & run

```bash
pip install -e .            # core (numpy, scipy, scikit-image, pandas)
pip install -e ".[dev]"     # + pytest
pip install -e ".[io,qc]"   # + nd2/tifffile reader and matplotlib QC (for real movies)
pytest                      # 62 tests
python examples/run_synthetic.py
```

## Configuration

`AcquisitionConfig` holds every physical parameter. **Set these to your rig**
before real data — the analysis is unit-aware and results depend on them:

```python
from pilitrack import AcquisitionConfig, analyze_movie
cfg = AcquisitionConfig(dt_s=1.0, pixel_size_nm=65.0, window_s=30.0,
                        min_pilus_length_nm=300.0)
res = analyze_movie(fluor_stack, cell_stack, cfg)
res["population"], res["cell"], res["pilus"]
```

## Running on real data (any format)

`pilitrack.io.load_movie` loads a time-lapse from **ND2 (Nikon), TIFF /
OME-TIFF, or CZI (Zeiss)** — or `load_array` wraps an in-memory array — into the
`(T, Y, X)` stacks the pipeline consumes, resolving channels, projecting Z, and
selecting a position. `config_from_meta` reads pixel size and frame interval
from the file's own metadata and rescales the pixel-unit params to your rig, so
the same recipe behaves consistently across microscopes:

```python
from pilitrack.io import load_movie, config_from_meta
fluor, cell, meta = load_movie("trial01007.nd2")   # cell is None if single-channel
cfg = config_from_meta(meta)
```

When cell body and pili share one channel (labelled-pilus TIRF),
`pilitrack.singlechannel` serves both pipeline roles from that channel:
dynamic-range-robust cell masking (background-flatten + winsorize + Otsu, so
faint *and* saturated cells segment) and white-top-hat + ridge pilus detection.

```bash
python examples/run_movie.py "Labelled data/trial01007.nd2" --fast   # one movie, tune on a crop
python examples/run_movie.py m.nd2 --config results/config.json       # reuse exact settings
python examples/run_batch.py  "Labelled data/" --out batch_results    # a whole folder
```

Every run writes `pili.csv`, `cells.csv`, QC overlays, a reusable `config.json`,
and a reproducibility `manifest.json` (input SHA-256 + all software versions +
every parameter). Automatic QC **flags** questionable movies (saturation,
defocus, no cells, sparse detection, out-of-range kinetics) rather than
averaging them in. Per-pilus kinetics are the robust output; per-cell counts /
% piliated depend on segmentation quality — use Omnipose or a real cell channel
for dense fields (below).

## Detection backends (Omnipose + ilastik)

The pipeline runs on built-in defaults (Otsu cells, Sato-ridge pili) but accepts
pluggable backends, in priority order — precomputed stack > callable > default:

```python
analyze_movie(fluor_stack, cell_stack, cfg,
              segment_fn=..., detect_fn=...,          # per-frame callables
              cell_label_stack=..., pilus_prob_stack=...)  # precomputed batches
```

**Omnipose → cells.** Morphology-independent *instance* segmentation that
separates touching rods (what the per-cell counts need). Use the bacterial phase
model `bact_phase_omni`:

```python
from pilitrack.backends.omnipose_backend import make_omnipose_segmenter
seg = make_omnipose_segmenter(cfg, gpu=False)   # model loaded once
res = analyze_movie(fluor, cells, cfg, segment_fn=seg)
# or batch up front:  cell_label_stack=segment_stack_omnipose(cells, cfg)
```

`pip install omnipose` (pulls torch). Omnipose is mid-refactor upstream
(splitting into `omnipose` + `omnitools`); the model-import path may shift, so
check omnipose.readthedocs.io for your version.

**ilastik → pili.** Pixel Classification turns a few brush strokes (pilus vs
background) into a trained per-pixel probability map — far better on faint,
low-SNR filaments than a fixed ridge filter, and it moves detection tuning out
of code constants into painted examples. Train `pili.ilp` in the GUI, then:

```python
from pilitrack.backends.ilastik_backend import (
    run_ilastik_headless, load_probability_h5, pilus_channel_stack)
outs = run_ilastik_headless("run_ilastik.sh", "pili.ilp",
                            ["movie1.tiff"], "out/")
prob = pilus_channel_stack(load_probability_h5(outs[0]), cfg)
res = analyze_movie(fluor, cells, cfg, pilus_prob_stack=prob)
```

ilastik (>=1.4.2) is a standalone app from ilastik.org, not a pip package; this
backend shells out to its headless CLI and reads the HDF5 export (`pip install
h5py`). Set `cfg.pilus_prob_channel` to match your label order and
`cfg.pilus_prob_threshold` for the cut.

You can mix them: Omnipose for cells, ilastik for pili, defaults for neither.

## GUI: review, curate, and hand-label (napari)

Automated detection always misses faint or crossing pili and mis-segments
touching cells, so there's an interactive step to **correct** it — and the fixes
flow through the same measurement path, so the numbers include them:

```bash
python examples/run_annotate.py "Labelled data/trial01007.nd2" --fast   # laptop-light crop
python examples/run_annotate.py movie.nd2 --load annotations.json         # resume
```

In the window: **trace a missed pilus** (draw its centerline → it becomes a
measured track), **paint cells** to fix segmentation, **remove** false-positive
tracks, and **Save/Load annotations** (`annotations.json` + an edited cell-label
TIFF) so work persists, reproduces, and doubles as training data. Hit
*Recompute* to fold edits into the kinetics. Runs on a laptop; work on a crop
(`--fast`/`--roi`) for large movies. The lighter `launch_viewer` (review + cull
only) is still available. Install `pip install "pilitrack[viewer]"`; the
annotation *logic* (`pilitrack.annotate`) is display-free and unit-tested.

Scriptable without a display, too — the viewer is thin over tested helpers:

```python
from pilitrack.pipeline import detect_and_link, summarize
art = detect_and_link(fluor_stack, cell_stack, cfg)
res = summarize(art["tracks"], art["per_frame_cell_labels"], cfg,
                art["n_frames"], keep_ids={0, 2, 5})   # keep only these tracks
```

## Validation status

- **Kinetics core** (`kinetics.py`) is unit-tested against synthetic traces
  with known velocities and recovers extension/retraction rates within ~15%,
  including under 40 nm localization noise. This is the load-bearing math.
- **Backend seams** are unit-tested with mock callables and precomputed stacks,
  so Omnipose/ilastik drop in without touching the pipeline. The wrappers
  themselves are written against each tool's documented API and run where those
  tools are installed.
- **GUI / hand-labeling** — the annotation *logic* (traced polyline → measured
  filament, folding manual pili + cell edits + track removals into the summary,
  save/load) is unit-tested display-free, and the napari path-shape format is
  verified against the real library; the window itself runs on a laptop with a
  display.
- **Built-in image front-end** runs end-to-end on synthetic movies and recovers
  per-pilus velocities near ground truth. Its ridge-detection threshold,
  `min_pilus_length_nm`, and `min_piliation_frames` are sensitive to real image
  SNR and should be tuned on your data — this is exactly what swapping in an
  ilastik probability map is meant to reduce.

## Notes on method

Length is the **geodesic path** along the pilus skeleton, not the endpoint
chord, because pili flex by Brownian motion (persistence length ~5 µm) and a
chord underestimates true length. Only components with a simple-path topology
are accepted; branchy skeletons (overlapping pili, debris) are rejected rather
than mismeasured. Tip-tracking relative to a fixed base is an alternative
front-end worth prototyping for high-density fields.
