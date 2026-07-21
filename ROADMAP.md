# pilitrack roadmap

Prioritized from a multi-agent research pass (2026-07) grounded in the codebase.
Ranked by value / effort; the lab's needs come first (reproducible, easy to run,
accurate pili numbers). Items are checked off as they land.

## Themes
- **Surface computation we already do.** Phases (kinetics), per-cell nested stats
  (stats), and provenance hashes are computed then under-shown. Cheapest wins are
  exports/figures over existing results, not new algorithms.
- **Close the reproducibility chain on the output side.** Inputs were hashed;
  outputs, environment, and git commit were not.
- **Global-vs-local tracking.** The linker resolves identity greedily on base
  distance alone; crossings/dropouts/drift are all closed by continuity terms.
- **Ease-of-run for bench biologists is a requirement, not polish.**
- **Keep the core install light** — heavy accelerators as optional extras.

## Done
- [x] **#5 Output checksums** — every results folder gets `checksums.sha256`
  (`sha256sum -c`-compatible) + `output_checksums` in the manifest.
- [x] **#7 git commit + package tracking + CITATION.cff** — manifest records the
  pilitrack git SHA (`-dirty` when the tree is modified) and scikit-learn /
  matplotlib / etc.; `CITATION.cff` at the repo root.
- [x] **#20 Determinism recording** — manifest records RNG seed, native
  thread caps, and loaded BLAS/OpenMP pools; `set_deterministic()` helper.
- [x] Crossing-filament resolution (research #3), organism-aware QC, per-frame
  readout, sub-pixel length — landed earlier this cycle.

## Next (high value / low effort)
- [ ] **#1 events.csv** — export the extend/pause/retract phases already computed
  in `kinetics` (dwell time, per-event velocity, run length, processivity).
- [ ] **#2 Hysteresis (double-threshold) detection** — recover faint distal ends
  / blinking pili via `skimage.filters.apply_hysteresis_threshold`.
- [ ] **#3 ruptures PELT phase segmentation** — principled extend/pause/retract
  boundaries instead of one velocity threshold (optional `[stats]` extra).
- [ ] **#4 Orientation + length-continuity linking cost** — keep pilus identity
  through crossings.
- [ ] **#6 SuperPlots** — honest per-cell hierarchical figure over existing stats.
- [ ] **#9 Downcast label images** to a minimal int dtype (memory).
- [ ] **#10 Progress bars + per-frame error isolation + memory preflight.**

## Later (higher effort)
- [ ] **#14 laptrack gap-closing + merge/split**, **#15 motion-aware cost**,
  **#16 sub-pixel base anchoring** (tracking).
- [ ] **#17 denoising pre-filter**, **#36 pretrained DL detector (bioimage.io +
  onnxruntime)** (detection).
- [ ] **#19 one-page PDF report**, **#18 mixed-effects WT-vs-mutant**, **#13
  one-click launcher** (biology/UX).
- [ ] **#21/#22 joblib parallel + disk cache**, **#27 OME-Zarr**, **#28 pixi
  lockfile/container** (performance/reproducibility).
- [ ] **#23 traccuracy tracking metrics**, **#24 realistic-PSF SNR sweeps**
  (validation).
