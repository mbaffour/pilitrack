# Project map — where everything is

## Top level  (`Pili Track/`)

```
Pili Track/
├── Install pilitrack.bat / .command      one-click install (run once)
├── Start pilitrack (browser).bat/.command   open the browser app
├── Start pilitrack (desktop).bat            open the napari hand-labeling app
├── SETUP.md            from-zero setup for a lab computer
├── MAP.md              this file
├── README.md / SOUL.md project overview + design story
├── environment.yml     conda env  ·  requirements.txt  pip install
├── .gitignore          excludes raw *.nd2 movies, caches, archives
├── Labelled data/      raw movies (e.g. trial01007.nd2) — NOT in git (large)
└── pilitrack_pkg/      the Python package (below)
```

## The package  (`pilitrack_pkg/`)

```
pilitrack_pkg/
├── pyproject.toml      deps, extras ([all], [io], [web], [viewer], [cellpose]…),
│                       and the console commands
├── examples/           runnable scripts (run_movie, run_batch, run_annotate,
│                       run_validate, run_synthetic)
├── tests/              101 tests (pytest)
└── src/pilitrack/      the code (below)
```

### `src/pilitrack/` — one line each

| Module | What it does |
|---|---|
| `config.py` | `AcquisitionConfig` — every physical/tuning parameter |
| `io.py` | load ND2 / TIFF / OME-TIFF / CZI / arrays → `(T,Y,X)`; auto-config from metadata |
| `singlechannel.py` | serve both pipeline roles from one labelled-pilus channel |
| `detect.py` · `measure.py` · `track.py` · `kinetics.py` | detect → measure → link → velocities (the validated core) |
| `pipeline.py` | `detect_and_link` → `summarize` |
| `analyze.py` | one reproducible run (load→detect→summarize→QC→export); `pilus_length_timeseries` |
| `batch.py` | run one config over a folder of movies |
| `qc.py` | quality metrics + flags (saturation, impossible outputs, …) |
| `provenance.py` | config save/load + run manifest (input hash, versions, params) |
| `annotate.py` | hand-labeling core: traced pili → measured tracks; **`pili_mask`**; save/load |
| **`dataset.py`** | **training storage: `save_training_bundle`, `collect_dataset`** |
| `validate.py` | score automation vs hand labels (F1 + length agreement) |
| `figures.py` · `stats.py` | kymographs/distributions · hierarchical bootstrap stats |
| `viewer.py` | napari GUI (review, trace, edit cells, **Save for training**) |
| `app.py` · `web_launch.py` · `webapp.py` | desktop launcher · web launcher · browser app |
| `backends/` | Cellpose / Omnipose (cells) · ilastik (pili) — optional |

## The commands (after install)

| Command / double-click | Opens |
|---|---|
| `pilitrack-web` · **Start pilitrack (browser)** | browser app: pick movie → analyze → download |
| `pilitrack-gui` · **Start pilitrack (desktop)** | napari: trace missed pili, edit cells, save labels |
| `pilitrack-batch <folder>` | analyze every movie in a folder |
| `pilitrack-validate <movie> --labels <json>` | detection F1 + length agreement vs hand labels |

## Where results and labels go

Nothing is written unless you ask (a runner `--out`, a GUI Save button). Then:

```
<your out folder>/                 # analysis of one movie
├── pili.csv                       per-pilus: length, ext/ret velocity, cell
├── cells.csv                      per-cell: n_pili, % piliated
├── pilus_length_over_time.csv     length(t) of each pilus
├── qc/…                           overlay PNGs
├── config.json                    reusable settings
└── manifest.json                  input SHA-256 + versions + params (reproduce it)

annotations.json (+ *_cells.tif)   # hand labels saved from a GUI
```

## Where **training data** goes  (the important new bit)

The GUI's **Save for training** (or `dataset.save_training_bundle`) writes one
self-contained, ML-ready **bundle per movie** — image + mask pairs plus full
provenance:

```
<training folder>/
└── trial01007/                    # one bundle per labelled movie
    ├── images/      frame_000.tif …   raw labelled frames (uint16)
    ├── pili_masks/  frame_000.tif …   pixel target for a pilus detector (0/255)
    ├── cell_masks/  frame_000.tif …   instance cell labels
    ├── annotations.json               vector traces + track edits + meta
    └── metadata.json                  movie hash, pixel size, dt, ROI,
                                        labelled frames, software versions
```

Collect many bundles into a training set with one call:

```python
from pilitrack.dataset import collect_dataset, dataset_summary
df = collect_dataset("training folder/")   # one row per labelled frame:
                                           # image, pili_mask, cell_mask, movie, frame, pixel_size…
dataset_summary("training folder/")        # {n_bundles, n_labeled_frames, n_movies, …}
```

That `df` feeds straight into a data loader to train a pilus detector or cell
segmenter — each row is an `(image, mask)` pair with the metadata to keep splits
honest (e.g. never put two frames of the same movie across train/test).
