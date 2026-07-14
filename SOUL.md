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
config      physical parameters (frame interval, pixel size, thresholds)
synth       synthetic traces + rendered movies with ground truth
detect      cell segmentation + pilus ridge detection (+ probability-map path)
measure     geodesic skeleton length + pilus-to-cell association
track       base-anchored linking into per-pilus length(t) trajectories
kinetics    Savitzky-Golay phase segmentation -> extension/dwell/retraction
pipeline    detect_and_link -> summarize (three separable stages)
backends/   Omnipose (cells) and ilastik (pili) wrappers, guarded imports
viewer      napari curation window + tested, display-free data-prep helpers
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
- **Needs real data to settle:** detection thresholds, `min_pilus_length_nm`,
  and `min_piliation_frames` — the % piliated metric especially. Swapping in an
  ilastik probability map is the intended way to reduce that sensitivity.

## Where it's going

- A loader (`tifffile`, ND2, CZI) so the pipeline points at real movies, not
  just arrays — the last thing standing between this and a real dataset.
- Click-to-select culling in napari, and manual *repair* of tracks (split a
  merged track, join a broken one) once we know the common failure modes.
- Calibration of detection against a hand-counted field of real data.
- And, on the horizon, the methods paper — the synthetic ground-truth harness
  is already the backbone of the validation study it would need.

---

A pilus is only interesting because it moves — extends, grips, lets go. The
whole point of this project is to catch that motion in numbers without a human
tracing every frame by hand. We haven't measured a real movie yet. But the hard
part isn't the seeing anymore; it's the trusting — and trusting is what we built
first.
