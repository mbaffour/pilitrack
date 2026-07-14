# pilitrack

Quantify **type IV pili dynamics** in *Pseudomonas* (or any T4P system) from
labelled two-channel time-lapse microscopy.

Assumes pili are **fluorescently labelled** — e.g. a PilA cysteine knock-in
(PilA-A86C in *P. aeruginosa*) tagged with a thiol-reactive maleimide dye such
as Alexa488-mal — imaged alongside a cell-body channel (phase or membrane
label). Pili are ~6 nm wide, below the diffraction limit, so unlabelled cells
cannot give per-pilus length or velocity.

## What it measures (per analysis window, configurable)

The five quantities a T4P lab wants, and where each lands in the output:

| Measurement | Output |
|---|---|
| **Number of piliated cells** | `population["n_piliated_cells"]` (+ `percent_piliated`) |
| **Number of pili per cell** | `cells.csv` → `n_pili` |
| **Length of individual pili** | `pili.csv` → `max_length_nm`; full length(t) trace in `pilus_length_over_time.csv` (`track_id, cell_id, frame, time_s, length_nm`) |
| **Extension velocity of individual pili** | `pili.csv` → `mean_extension_velocity_nm_s` |
| **Retraction velocity of individual pili** | `pili.csv` → `mean_retraction_velocity_nm_s` |

Every run writes these as CSVs (plus a reproducibility `manifest.json`); the GUI
readout and the terminal report print them directly.

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
selecting a position/scene. `config_from_meta` reads the acquisition config from
the file's own metadata (pixel size, frame interval) so you never hand-transcribe
it:

```python
from pilitrack.io import load_movie, config_from_meta
fluor, cell, meta = load_movie("trial01007.nd2")   # cell is None if single-channel
cfg = config_from_meta(meta)                        # dt_s / pixel_size_nm from the file
```

`config_from_meta` **rescales the pixel-unit params** (`ridge_sigmas`,
`base_search_radius_px`, `max_base_jump_px`) to your pixel size instead of the
65 nm/px the defaults were tuned at, and floors `velocity_sign_eps_nm_s` at half
a pixel-step per frame so pixel quantization isn't mistaken for real
extension/retraction — the same recipe therefore behaves consistently across
rigs. Any keyword overrides a computed value.

**Single- vs dual-channel.** If the movie has a separate cell-body channel
(brightfield/phase/membrane), it is used directly for segmentation. If cell body
and pili share **one** channel (labelled-pilus TIRF — surface pilin is labelled
too), `pilitrack.singlechannel` serves both roles from that one channel by
separating them morphologically: cells via a **dynamic-range-robust mask**
(background-flatten + winsorize + Otsu, so faint *and* saturated cells are both
segmented — a single global Otsu catches only the brightest); pili via a white
top-hat (removes the cell body, flattens illumination) + Sato ridge, normalized
by a high percentile with cell interiors masked out.

**One movie** — writes `pili.csv`, `cells.csv`, QC overlays, a `config.json` you
can reuse, and a reproducibility `manifest.json`:

```bash
python examples/run_movie.py "Labelled data/trial01007.nd2" --fast    # tune on a crop
python examples/run_movie.py movie.ome.tif --out results --no-viewer
python examples/run_movie.py m.nd2 --config results/config.json        # reuse exact settings
```

**A whole folder** — one consistent config over every movie, rolled up into
`summary.csv` (one row per movie, with QC flags) and `pili_all.csv`:

```bash
python examples/run_batch.py "Labelled data/" --out batch_results
```

**Reproducibility & QC.** Every run records a `manifest.json` — the input file's
SHA-256, the versions of Python and every dependency, and the full config +
detection parameters — so any number can be reproduced exactly. Automatic QC
metrics **flag** questionable movies (saturation, defocus, no cells segmented,
sparse detection, out-of-range kinetics) instead of silently averaging them in.

`--detect-threshold` (default 0.30, tuned for real TIRF SNR) trades background
noise against faint-pilus sensitivity. The **per-pilus kinetics** (length,
extension/retraction velocity) are the robust, load-bearing output; **per-cell
counts and % piliated depend on segmentation quality** — for dense fields or
touching cells, plug in Omnipose (instance segmentation) or a real cell channel
(below). The heavier upgrade for very faint filaments is a trained ilastik
probability map.

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

## GUI: review, curate, and **hand-label** (napari)

The easiest way in — one command opens a movie, runs the analysis, and drops you
into the interactive window with a live results readout and every tool
(install `pip install "pilitrack[viewer]"` first):

```bash
pilitrack-gui                              # file dialog -> open any movie
pilitrack-gui "Labelled data/trial01007.nd2" --fast    # laptop-light center crop
python -m pilitrack movie.ome.tif          # equivalent
```

Detection failure modes for pili are visual — a missed faint filament, two
crossing pili linked into one track, a break mid-retraction — so there's a
**human review + correction** step, and the fixes flow through the *same*
measurement path so the reported numbers include them:

```
detect_and_link(...)  ->  [review + trace missed + edit cells + cull]  ->  summarize
```

```python
from pilitrack.viewer import launch_annotator     # or drive it from a script
launch_annotator(fluor_stack, cell_stack, cfg, movie_path="movie.nd2")
```

In the window:
- **Trace a missed pilus** — select *manual pili (draw)*, pick the path tool, and
  click along the filament; it becomes a real, measured track (length +
  extension/retraction velocity).
- **Edit cells** — paint the *cells (editable)* layer to add a missed cell, erase
  a false one, or draw a 0-valued split line between two merged cells.
- **Fix tracks** — type false-positive track IDs to remove.
- **Save / Load annotations** — writes `annotations.json` (+ an edited
  cell-label TIFF) so work persists, results reproduce, and the labels double as
  training data for better auto-detection.

Hit **Recompute** after any edit to fold it into the kinetics; **Export CSVs**
writes the corrected tables. Runs on a laptop — for large movies work on a crop
(`--fast`/`--roi`). The lighter `launch_viewer` (review + cull only) is still
available. Install with `pip install "pilitrack[viewer]"` (napari + magicgui + a
Qt backend); the annotation *logic* (`pilitrack.annotate`) is display-free and
unit-tested, so the science is verified even though the window needs a screen.

## Validation against ground truth & figures (for publication)

Hand-label a few frames **fully** in the annotator, then score the automation
against them — detection precision/recall/F1 and how closely measured lengths
match hand-traced ones (bias + Bland-Altman limits):

```python
from pilitrack.annotate import load_annotations
from pilitrack.validate import validate
truth, _ = load_annotations("val_frames.json")     # fully-labeled frames = ground truth
report = validate(art, truth, cfg, frames=[0, 10, 20])
report["detection"]   # {precision, recall, f1, tp, fp, fn}
report["length_agreement"]  # {mae_nm, bias_nm, loa_low_nm, loa_high_nm}
```

Render the canonical figures (each pilus's length over time — the kymograph
equivalent — and the velocity/length distributions):

```python
from pilitrack.figures import make_report_figures, plot_single_kymograph
make_report_figures(res, art, cfg, "figs/")       # length_traces.png, distributions.png
plot_single_kymograph(art["tracks"][0], cfg, art["n_frames"], "figs/kymo0.png")
```

## More knobs for accuracy

- **Instance cell segmentation** (touching cells → exact per-cell counts):
  `pilitrack.backends.cellpose_backend.make_cellpose_segmenter()` returns a
  `segment_fn` for `analyze_movie` (`pip install cellpose`; a GPU helps).
- **Optimal tracking** for crossing/dense pili:
  `AcquisitionConfig(linker="lap")` swaps greedy base-matching for a globally
  optimal Hungarian assignment.
- **Publication statistics**: `pilitrack.stats.kinetics_table(pilus_df,
  level="per_cell")` averages within each cell first (pili aren't independent)
  and reports bootstrap 95% CIs; `per_cell_table`/`combine_pili` aggregate across
  cells and movies.

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
- **Curation viewer** — its data-prep (skeleton label stack, tracks array,
  culled re-summary) is unit-tested; the napari window itself runs on a machine
  with a display.
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
