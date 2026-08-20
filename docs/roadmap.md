# Roadmap

Where the project is going, track by track, with acceptance criteria. Status values:
`planned` → `in progress` → `done` (with the landing PR). Superseded plans are rewritten,
not appended; **finished work leaves this file** for
[`CHANGELOG.md`](../CHANGELOG.md)'s `[Unreleased]` section, which is where the "what
landed, with which numbers" record lives. Background and rationale live in
[`system-design.md`](system-design.md); deferred items and known debt in
[`backlog.md`](backlog.md).

## Completed foundations

| Milestone | PR |
|---|---|
| 12 → 3 crate consolidation, CI unblocked, docs realigned | #13 |
| Dependency bumps, licensing, real CI (6 jobs), 4 algorithm bug fixes, 198 tests | #15 |
| Shape-based object detection replacing the chamfer matcher; validated 256/256 on canend | #18 |
| B-track closure, extractor split, serde model persistence | #20 |
| Detection performance: lazy tiled direction fields, 7.8 → 3.46 ms full-360° | #21, #23 |
| Visual diagnostics + corrmatch external validation | #22 |
| Track A — v0.2 substrate reset (`Pixel`, generic pyramid, six defects); B1 `fit`; B2 `measure` | #24 |
| Track D — v0.3 API reset (visibility, configs, result honesty, model format 3, `core` split, `shape` → `lsd`) | #25 |
| Docs realigned to the v0.3 surface + the deterministic illustration system | #26 |
| Track D.1 — vm-python parity (config mirror, dtype dispatch, `.pyi` + `py.typed`) | #27 |
| Visual Metrology Lab MVP (FastAPI backend + browser workbench) | #28 |
| Track C1 — accuracy regression suite, first rows and pinned envelopes | #29 |
| Lab typed API client + `LabBackend` transport abstraction | #30 |
| Track B4 — `warp::Map` (+ the C1 scale-sweep row) | #31 |
| Track B4.1 — rectify: canonical-pose crops (`matching` + `warp`) | #32 |
| Track B5 — `metric`: the pixel → millimetre calibration bridge | #33 |
| Track B6 — `corr`: corrmatch wrapper and inter-frame `displacement` | #34 |
| Track B7 — mosaic: bird's-eye composite of N calibrated cameras | #35 |
| Track C4 + B2.1 — Tauri desktop shell, contract fixtures, `measure::diagnostics::layout` | #36 |
| Track B8 — `scale`: estimate-then-verify scale invariance | #37 |

## Where this is going

The library can **find** a part, **rectify** it, **measure** it with calipers, **fit**
primitives robustly, and report the result **in millimetres** through a calibration. The
whole chain runs end to end on real data:

```
acquire → rectify → locate (matching) → fixture (pose) → measure (calipers)
        → fit primitives (robust) → gauge in millimetres → pass / fail
```

What is left is filling in the gaps around that spine: one missing module (`filter`), the
accuracy record that makes the numbers trustworthy, and the workbench that makes the whole
thing usable by a human.

---

## Track B — the measurement spine — `in progress — B1/B2/B2.1/B4/B4.1/B5/B6/B7/B8 done, B3 remaining`

Each module ships with its own bench, doc example, Python parity, and an accuracy entry in
Track C1.

### B3 — `filter`: the absent workhorse — `planned`

Separable and recursive (Deriche / van Vliet) Gaussian, sliding-window box mean, an
O(1)-per-radius histogram **median**, grayscale erode/dilate/open/close/tophat (van
Herk–Gil-Werman). `edge/conv1d.rs` folds in here. Feeds the pyramid pre-smooth and
illumination correction. Scope is deliberately median only, not the full rank-order family
— see [`backlog.md`](backlog.md).

**Accept:** each filter matches a naive reference bit-for-bit on random fixtures; median is
O(1) in radius, measured rather than asserted.

`warp` does **not** block on this: `apply`'s per-pixel gather is a plain resample, and
`pyr::Pyramid` already covers decimation (see system-design's `warp` entry).

---

## Track C — credibility and infrastructure — `in progress`

### C1 — accuracy regression suite  ← the differentiator — `in progress — envelopes pinned, coverage and doc table pending`

Performance is measured and recorded; **accuracy is not**, and for a metrology library the
accuracy numbers *are* the product. Track A found three separate systematic biases that
every existing test passed straight through.

| Operator | Sweep | Report | Status |
|---|---|---|---|
| `Edge2DDetector`, `Edge1DDetector` | edge angle 0–90°, blur σ 0.5–3, noise 0–5 LSB | bias, σ (px) | done (#29) |
| `MeasureHandle::pos` | same, plus caliper width | bias, σ | done (#29) |
| `fit_circle` | point count, arc extent, noise, outlier fraction | radius bias, centre σ | done (#29) |
| `ShapeMatcher` | sub-pixel translation / rotation / scale sweep | pose bias, σ | done (#29, #31, #37) |
| rectify, `corr::displacement` | pose jitter; exact fractional shifts | intensity bias/σ; px bias/σ | done (#32, #34) |
| `fit_ellipse` | point count, arc extent, noise, outlier fraction | axis bias, centre σ | **open** |
| `LaserExtractor` | stripe width, saturation, tilt | centre bias, σ | **open** |

Every landed row is gated inside a recorded envelope. Still to do: the two open rows above,
and `docs/accuracy.md` — a generated table (written by an example, the way the performance
table is) alongside the performance numbers. No open-source Rust CV crate publishes this.

### C2 — blob features — `planned`

`ComponentStats` is bbox + centroid + count. Add second-order moments → orientation and
elongation, plus convex hull, min-area rect, circularity, rectangularity. Cheap on top of
the existing CCL and needed for blob-based inspection.

### C3 — bindings and CI — `in progress`

Python dtype dispatch is **done** (Track D.1). Still open: generate the vm-python config
conversions instead of hand-mirroring them (now spread across `src/config/*.rs`, ~750
lines but each file under the 600-line soft cap); a Python binding for `laser` and for
`segment::watershed`/region-growing (see [`backlog.md`](backlog.md)); `cargo publish
--dry-run` in CI; miri over **all** unsafe, not just `laser::`.

---

## Track E — the lab as a workbench — `in progress`

The desktop shell landed in #36 as a transport experiment: the same frontend, talking to
Rust instead of HTTP. This track turns it into something a person can actually work in —
open a capture, see what a model learned, correct it, and run it over the whole set.

- **Desktop transport rewrite.** Image tiers (`thumb` 256 / `preview` 1024 / `full`) are
  PNG-encoded once into `{app_cache}/tiers/{sha256}/{tier}.png` and served to the webview
  as paths through Tauri's asset protocol, replacing the per-request re-encode and the
  bytes-over-IPC hop. Keying on pixel-content `sha256` (not image id) is what makes the
  cache survive re-opening a file. `images_open_paths` / `images_scan_dir` register images
  **by path** — no pixels cross the IPC boundary. Every wrapper that touches pixels is
  `async` over `spawn_blocking`, so find / encode / scan no longer freeze the window.
- **Curated teaching.** `ShapeModelBuilder::build_with_mask` (closing the backlog's
  arbitrary-region model ROI item) and `ShapeModelConfig::reference_angle`, which rotates
  the model frame onto a caller-chosen canonical orientation. Lab side: `teach_preview`
  runs exactly the extraction `ShapeModelBuilder` runs and links it into pickable contours;
  `mask_for_contours` turns the picked subset back into the inclusion mask — the two halves
  agreeing by recomputing, not by caching (invariant 12).
- **Model geometry accessors** — `ShapeModel::{model_geometry, reference_geometry,
  reference_points, reference_angle}`, so an overlay can draw what a model actually learned,
  in either frame. Model format 4 → **5** (`reference_angle`), backward-loading.
- **Batch find** — one model over a whole capture, per-frame failures reported on that frame
  and the run continuing, `lab://batch` progress events. Reading a model one frame at a time
  hides exactly the tail that decides whether it is usable.
- **Workspace-based frontend IA** — routes under Library / Recognize (teach, find, verify) /
  Gauge (measure, align) / Camera (motion, mosaic), shared state in `LabProvider`, frame in
  `AppShell`, replacing the single tab-switching `App.tsx`.

**Accept:** contract-fixture parity still green on both shells; the desktop build opens a
folder of real frames, teaches a masked model from picked contours, and batch-finds across
the set without blocking the window.

---

## Later milestones

- **v0.2.0 publish**: `vm-primitives` + `vision-metrology` to crates.io (+ wheels) once
  Track B lands. Gate: `cargo publish --dry-run` both crates, README/docs.rs review.
- **Direct vision-calibration dependency**: when tiny-solver/faer move to nalgebra 0.35 and
  calibration-rs rebases; replaces the `metric` mirror types.
- **Shared substrate across the vision workspaces**: `box-image-pyramid`, `corrmatch` and
  `chess-corners-rs` each carry their own `ImageView`. Publishing this crate's `core` as the
  common substrate would end that duplication — worth doing only after v0.2 settles the API.
- See [`backlog.md`](backlog.md) for unscheduled items.
