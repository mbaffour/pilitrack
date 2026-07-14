# pilitrack — the soul of the project

*Quantifying the reach and retreat of type IV pili, one honest measurement at a time.*

---

## Why this exists

Type IV pili are how *Pseudomonas* touches the world. They extend, grip, and
retract — driving twitching motility, surface sensing, and the earliest
decisions a cell makes about where to live. To understand that behaviour you
have to *measure* it: how long the pili get, how many a cell makes, how fast
they push out and reel back in, and how much of a population is piliated at all.

The biology community can already *see* these pili — the field solved that with
a beautiful trick: knock a cysteine into the major pilin (PilA-A86C in
*P. aeruginosa*) and label it with a maleimide dye, and suddenly a sub-6-nm
filament, far below the diffraction limit, lights up in a fluorescence movie.
What the community mostly *doesn't* have is a way to quantify those movies
without a human tracing pilus tips by hand, frame by frame, kymograph by
kymograph. The measurements exist; the automation doesn't. That gap is the
reason for this project.

pilitrack is an attempt to turn a labelled time-lapse into numbers you can
trust — automatically, reproducibly, and with the seams left open so better
tools can slot in as they arrive.

## What we're trying to achieve

A single pipeline that takes a two-channel movie (cell bodies + labelled pili)
and returns, over any analysis window:

- which objects are cells, and which cells carry pili;
- the number of distinct pili per cell;
- each pilus's length over time, and its maximum;
- extension and retraction velocities, phase by phase;
- the percentage of the population that is piliated.

Not a black box. A pipeline whose load-bearing math is checked against known
answers, whose weak points are named out loud, and whose detection front-end can
be swapped for the best available model without rewriting the science.

## The principles that shaped it

These weren't decided up front — they emerged as we built, and they're worth
keeping.

1. **Validate the thing that matters against ground truth.** Velocity extraction
   is the measurement people will cite, so it is tested directly against
   synthetic traces with *known* extension and retraction rates — including
   under localization noise. When the first finite-difference version
   underestimated velocity, we didn't loosen the test; we fixed the method
   (Savitzky-Golay derivative, transition frames trimmed before fitting).

2. **Be honest about what's tuned vs. what's true.** The kinetics core is
   locked to ground truth. The image front-end — segmentation, ridge detection,
   thresholds — is genuinely sensitive to real image SNR, and the docs say so
   plainly rather than pretending a synthetic demo is proof.

3. **Separate detection from curation from summarizing.** Pili fail *visually*
   (a missed faint filament, two crossing pili linked as one, a break
   mid-retraction). So the pipeline is three stages with a human review step in
   the middle — detect and link, cull the spurious tracks, then summarize only
   what survived.

4. **Leave the seams open.** The built-in detectors are defaults, not
   commitments. Omnipose (for instance-segmenting touching rods) and ilastik
   (for a trained pilus-probability map) drop in through pluggable backends
   without touching the pipeline — because the right tool for detection will
   keep changing, and the science shouldn't have to.

5. **Ground truth is a first-class citizen.** The synthetic movie generator
   isn't a toy — it's the fixture that lets us tune thresholds against known
   answers before trusting a single real number, and it's what a methods paper
   would lean on to prove the tool works.

## What exists today

A proper Python package (`src` layout, `pyproject`, 14 passing tests), organized
as a clean chain:

```
config        physical parameters (frame interval, pixel size, thresholds)
synth         synthetic traces + rendered movies with ground truth
detect        cell segmentation + pilus ridge detection (+ probability-map path)
measure       geodesic skeleton length + pilus-to-cell association
track         base-anchored linking into per-pilus length(t) trajectories
kinetics      Savitzky-Golay phase segmentation -> extension/dwell/retraction
pipeline      detect_and_link -> summarize (three separable stages)
io            load ND2 / TIFF / OME-TIFF / CZI / arrays -> (T,Y,X) + auto-config
singlechannel serve both pipeline roles from one labelled-pilus channel
analyze       one reproducible code path: load -> detect -> summarize -> QC -> manifest
batch         run one config over a whole folder -> per-movie + combined summary
qc            automatic quality metrics + flags (saturation, defocus, sparse, ...)
provenance    config save/load + run manifest (input hash, versions, params)
annotate      hand-labeling core: traced pili -> measured tracks; save/load labels
validate      score automation vs hand labels: detection F1 + length agreement
figures       publication plots: length-over-time (kymograph) + distributions
stats         hierarchical aggregation + bootstrap CIs for publication tables
app           the GUI-first entry point: open -> analyze -> hand-label -> export
backends/     Omnipose + Cellpose (cells) and ilastik (pili) wrappers, guarded
viewer        napari GUI (review + trace missed pili + edit cells) + tested helpers
```

Length is measured as the geodesic path along the pilus skeleton, not the
endpoint chord, because pili flex by Brownian motion and a straight line
underestimates the truth. Branchy, non-filament skeletons are rejected rather
than mismeasured. Piliation is judged from *tracked* pili that persist and reach
a real length, so a one-frame noise ridge never counts as a pilus.

## What's validated, and what isn't

- **Validated:** the kinetics core recovers known extension/retraction
  velocities within ~15%, including under 40 nm noise; the backend seams are
  tested with mock callables and precomputed stacks; the viewer's data-prep and
  the culling-changes-the-summary logic are tested.
- **Written but run on your machine:** the Omnipose and ilastik wrappers (heavy
  deps / GUI app) and the napari window itself (needs a display).
- **Measured on a real movie:** the pipeline now runs end-to-end on a real
  labelled-pilus TIRF ND2 (single 488 nm channel, 1952x1952x70, 43.3 nm/px) and
  recovers per-pilus kinetics squarely in the published *P. aeruginosa* T4P
  range (extension ~0.3-0.4 um/s, retraction ~0.3-0.5 um/s, lengths sub-um to a
  few um). Detection thresholds, `min_pilus_length_nm`, and the % piliated metric
  are still the tunable, SNR-sensitive parts.
- **Cell segmentation is the honest weak point on a single channel.** With a
  10-100x brightness spread and diffuse background label, a global threshold
  can't cleanly separate all cells; the `robust` mask (flatten + winsorize +
  Otsu) recovers faint *and* bright cells, but touching cells still merge.
  Accurate per-cell counts want Omnipose or a real cell/phase channel.

## Where it's going

- **Done:** a multi-format loader (ND2 / TIFF / OME-TIFF / CZI / arrays) with
  metadata-driven config, single- and dual-channel handling, batch processing,
  automatic QC flags, and a reproducibility manifest (input hash + software
  versions + every parameter) — the pipeline now points at real datasets, and
  the same recipe travels across movies and rigs.
- **Done:** a GUI-first tool (`pilitrack-gui`) — open a movie, it analyzes, and
  you review, **trace missed pili**, paint to fix cells, cull false positives,
  and save/reload annotations (which double as training data). Display-free
  labeling logic, unit-tested; the window runs on a laptop.
- **Done:** the publication scaffolding — `validate` (detection F1 + length
  agreement vs hand labels), `figures` (kymographs + distributions), `stats`
  (hierarchical CIs), a `Cellpose` instance-segmentation backend for touching
  cells, and an optimal (`linker="lap"`) tracker for crossing pili.
- **Next (needs data, not code):** collect a fully hand-labeled validation set
  and report F1 + length agreement; benchmark against manual kymograph tracing.
- **Next (code):** a trained pilus detector (U-Net/ilastik) to cut full-field
  false positives at the source; deconvolution + per-measurement uncertainty.
- And, on the horizon, the methods paper — the synthetic ground-truth harness
  is already the backbone of the validation study it would need.

---

A pilus is only interesting because it moves — extends, grips, lets go. The
whole point of this project is to catch that motion in numbers without a human
tracing every frame by hand. We've now measured a real movie, and the velocities
land where the biology says they should. The hard part was never the seeing;
it's the trusting — and every run now carries its own provenance so the numbers
can be checked, reproduced, and believed.
