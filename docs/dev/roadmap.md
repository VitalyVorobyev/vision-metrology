# Roadmap

Open work, in priority order, with acceptance criteria. Status values: `planned`,
`in progress`. When an item lands it leaves this file: user-visible changes go to
[`CHANGELOG.md`](../../CHANGELOG.md), and decisions to an ADR. Known debt that is not
scheduled lives in [`backlog.md`](backlog.md). Each item links its GitHub issue. The
milestones on GitHub mirror the sections below.

## Where the library stands

The measurement chain runs end to end on real data: rectify → locate → fixture → calipers
→ robust fit → millimetres → pass/fail ([system design](system-design.md)). Beads and
stripes are tracked along a prior curve and measured station by station
([ADR-0018](adr/0018-tracked-curves.md)). Masked translation registration is a
general-purpose `corr` operator.

The program below turns this into what tracked-curve metrology needs next: one curve model
from many registered observations, measured on native pixels. A downstream application
prototype has validated each step on private data and reimplemented it locally. Its
generic findings are issues [#73]–[#78]. The goal is that such applications stand on
these operators instead of copies of them.

## How the work is grounded

**Three tiers of evidence.** Every operator reports the tiers that apply to it.

| Tier | Data | Where it is reported | Rules |
|---|---|---|---|
| Synthetic | Generators with exact truth: the ribbon scenes, phase sweeps, the sequence benchmark ([#86]) | `tests/accuracy/*`, envelopes at about 1.5× the measured value; [`performance.md`](../performance.md) | In CI; deterministic (invariant 12) |
| Public real | DamSegment; planar-tracking and viewpoint benchmarks; human-annotated paired edges ([#85]) | [`performance.md`](../performance.md), with the dataset, split and bootstrap intervals | Licence checked and recorded in the [dataset registry](../../tools/datasets/README.md); data never committed; settings frozen on dev before the test split runs |
| Private | Application sequences under opaque labels: development A–D and the early part of H; held out E, F and the late part of H; raw multi-strip G; calibrated scans P1–P4 | Aggregates only, under those labels | Adapters in the git-ignored `tools/private/` write the neutral `sequence.json`; held-out data runs once with frozen settings; `check-invariants.py` guards against leaks |

Public reports never depend on private data. A claim that only private data supports is
labelled as such.

**An operator's lifecycle.**
1. An issue with the use case and a synthetic acceptance case.
2. A Python prototype in `tools/` when the algorithm is uncertain, with a decision gate.
3. An ADR.
4. Rust, Python parity (invariant 15), accuracy rows, a bench and a public report.
5. The downstream consumer adopts it in place of its copy, and its own private gate is no
   worse. Until then, a new operator is marked experimental.

**What stays downstream:** acquisition, recipes, tolerances and defect decisions, tool and
dispenser models, view selection for a laid curve ([#75]), pairing with blank runs by
external stamps, reviewer interfaces and label stores. Calibration solvers and target
detection belong in the calibration crates ([ADR-0003](adr/0003-metric-calibration-bridge.md)).

The motion models are not expanded further. Registration must not depend on the tracked
object: callers mask it out, and a test checks that an object moving relative to its
substrate is flagged, not absorbed as motion.

---

## v0.2.0: consumable

### R: first release, `in progress` ([#79])

- CI packages both library crates (`cargo publish --dry-run`) and installs and tests each
  built wheel on Linux x86_64 and aarch64, macOS and Windows.
- The public Rust and Python surface is reviewed before the release fixes it, in four
  phases, each approved on its own. It follows registration calibration, which settles
  `TranslationConfig`.
- CHANGELOG cut, versions 0.2.0, install sections point at crates.io and PyPI.
- crates.io's first publish is manual (trusted publishing needs an existing crate); later
  releases publish from CI over OIDC, PyPI over a trusted publisher. No stored tokens.

**Accept:** `vm-primitives` and `vision-metrology` 0.2.0 on crates.io with docs.rs builds;
the `vision-metrology` 0.2.0 wheel on PyPI imports and runs the README quick start on all
four targets.

### A nalgebra range for downstream crates, `planned` ([#80])

Crates that also depend on calibration-rs are held at nalgebra 0.34 by its solvers, so
today they cannot depend on these crates at all. Declare `>=0.34, <0.36`, with a CI job
that pins the lowest version, and a smoke crate that resolves both dependencies to one
nalgebra.

**Accept:** both versions green in clippy, tests and docs; invariant 13 and
[ADR-0002](adr/0002-dependency-and-toolchain-policy.md) state the range and how it is
tested. If the differences between versions make this impractical, ADR-0002 records it,
and consumers use the wheel and versioned artifacts.

### Edge fidelity: phase-swept accuracy and a locking diagnostic, `planned` ([#77])

A sub-pixel locator can be unbiased at phases 0, ¼, ½ and ¾ and biased between them, and
on real data it shows as a non-uniform histogram of fractional positions.
- Sweep 20 phases × 4 angles × 3 blurs for the strip and caliper rows, and report the
  worst bias and a locking index: the max/min ratio of a 10-bin histogram of fractional
  positions under noise.
- A ground-truth-free histogram of fractional edge coordinates, over a `sequence.json`
  manifest.

**Accept:** worst bias and locking index published for every locator; the private
development sequences' histograms reported as aggregates.

### Registration calibration, `planned` ([#81])

The similarity design is in [registration-stage2.md](registration-stage2.md).
- Diagnose first. Synthetic: a true translation or similarity on DoG-band texture at
  320×97 with realistic noise, measuring the current false-reject rate. Private: on
  development sequences A and B, disable the regional check and compare `region_error`
  with reciprocal closure and anchor corrections.
- Replace the fixed-px regional threshold with a covariance-normalized test.
  Translation: χ² on pairwise regional shift differences, from the summed regional
  covariances plus a model-bias floor in px². Similarity: compare each regional fit with
  the global one at that region's own centroid and extent, without extrapolation.
  Config: `region_check: Option<RegionCheck { max_chi2, bias_floor_px }>`.
- Convergence: freeze the observation set at the start of refinement, fix the Huber scale
  after the first iteration, and test convergence on the motion step only. Reproduce the
  failure first with a synthetic overlap/mask fixture.
- Give the similarity initializer a search-only path.
- Public real data: HPatches, registered with its split by sequence
  ([dataset registry](../../tools/datasets/README.md)). Its illumination sequences, with
  a known identity homography, for false rejects on real texture; its viewpoint pairs for
  rejection of an observable rotation or affine.

**Accept:** synthetic false rejects ≤ 1% at the stated noise; an observable 0.01 rad
rotation (translation) and a mild affine (similarity) are still rejected; the public
numbers published with the split and the settings frozen; before/after coverage on the
development sequences, with reciprocal-closure p95 no worse; the held-out sequences
reported with the settings frozen; benches updated.

---

## v0.3: native boundary evidence

### Continuous derivative-of-Gaussian profiles, `planned` ([#82])

A public, allocation-free sampler of profiles along curve normals. It evaluates
`R(p) = Σ_q I(q) ∇G_σ(p − q)·n` at continuous points, honours a validity mask, and tapers
the kernel by `(1 − d²/r²)²` to zero at r = 4σ, so the response is C¹. It brings forward
`filter`'s separable Gaussian and its derivatives. `MeasureStrip` uses it, or its locking
is bounded and documented.

**Accept:** locking index ≤ 1.2 over the phase sweep; worst bias reported per σ; an
allocation gate, a bench and Python parity.

### Shared DP core and robust smoother, `planned` ([#83])

- An exact min-path over layered states with banded quadratic transitions, separable
  where the state allows, max-marginals per layer, and soft pins (a hard constraint can
  make a step-bounded path infeasible).
- A robust Whittaker smoother, `Σ wᵢ ρ(uᵢ − ûᵢ) + (ℓ/h)⁴ Σ (Δ²u)²`, solved with a banded
  LDLᵀ in O(n) under IRLS. It removes the bin quantization a path leaves, and ADR-0018's
  regularized solve shares it.

**Accept:** path and max-marginals equal brute force on random small instances; the
smoother matches a dense solve and benches O(n); the bead accuracy rows do not change.

### Coupled two-boundary extraction, `in progress` ([#76], [#84])

A global choice of both walls along a curve instead of `pairing::choose`'s choice at each
station alone. The Python prototype in `tools/bead_eval` met its decision gate on
DamSegment: the dense variant locks 91.0% of calls against the tracker's 85.3%, with 7.1%
wrong at equal coverage against 13.0%. Its max-marginals flag a competing parallel line,
but on cracks most wrong locks are not close calls. The design is ADR-0019, proposed.
- A dense path over (centre, width) bins on the shared core, with costs from the
  continuous profiles, in ADR-0018's length-scale parameters so the result does not
  depend on station spacing.
- States for both walls, **left only and right only**, and absent: where one side leaves
  the image or a mask, the visible wall is still measured.
- Walls refined on native pixels with the caliper; max-marginal margins per station and
  for the whole call; typed absence.
- A `BeadTracker` pairing mode, with `explain_bead` parity through the existing probes.

**Accept:** the Rust results reproduce the Python DamSegment numbers within their
bootstrap intervals; #73's synthetic case within 0.25 px on the observable walls, with no
wall reported where every sample is occluded; accuracy rows for boundary bias and σ, the
2 px distractor and the parallel line, which the whole-call margin flags; a bench at 300
stations within the tracker's budget; Python parity; the 600-line cap; ADR-0019 accepted.

### Bead ends, `planned` ([#78])

With a prior that runs onto a bead's end, the end station can pair noise: in 1 trial of
270 the curve's end swung by about 7 px. Tangential extension is not observable with
normal searches, so the end is measured as a polar rim from the walls' last points: rays
from the chord's centre, evidence along each ray from the continuous profiles, and a path
on the shared core pinned to the half chord at both ends. Weak evidence keeps the chord
and reports the end as unmeasured.

**Accept:** half-disc, flat, skewed, image-clipped and distractor-adjacent ends with rim
bias and σ; the 1-in-270 swing does not occur with measured ends; Python parity and a
bench.

### Public two-boundary benchmark, `planned` ([#85])

Paired boundary positions annotated by several observers, such as a vessel-width
database, or OCT layer annotations as the fallback. Edge and width error against the
observer mean, beside the inter-observer spread, for the tracker and the coupled mode.

**Accept:** licence recorded; dev/test split with the settings frozen on dev; numbers with
bootstrap intervals in `performance.md`.

---

## v0.4: many observations, one model

### Planar rig substrate pose, `planned` ([#74])

Several views of one planar substrate under a fixed rig, each with a plane→image
homography and a masked translation per frame pair, give one rigid SE(2) pose per frame.
Each view's translation is lifted at the centroid of its measured region, so rotation
comes from the lever arms between views. Start robustly from the cross-view median, then
IRLS. A step with one view holds rotation at the prior and says `single_view`; a step
with none is bridged at constant velocity with growing uncertainty and says `bridged`.
Also: a smoother over the shift series, and the path-length ratio as a separate scale
diagnostic. Its home is settled in an ADR.

**Accept:** #74's synthetic cases (a bridged step, a gross view down-weighted, a
single-view step held at the prior); provenance per step.

### Synthetic sequence benchmark, `planned` ([#86])

A deterministic generator writing `sequence.json` with truth: a curved ribbon with
asymmetric, variable-width walls on a moving textured substrate, one to three affine
views with a handover, camera-fixed highlights, occlusions, a parallel line, a
distractor also present in the blank, missing observations, an actual gap, injected
registration error, and a bead that moves relative to its substrate. No public dataset
has many registered observations of one curve, so this is the public stand-in.

**Accept:** deterministic and in CI; truth per station, per frame and per view; the
consuming operators' numbers in `performance.md`.

### Multi-observation fusion, `planned` ([#73])

- Stations in the model's own coordinates. Each observation is sampled in its native
  frame through the caller's map, as soft-saturated continuous profiles, weighted by
  visibility, registration uncertainty and whether the edge's support lies on the
  material side.
- **Accumulated evidence chooses, native edges place.** The coupled extraction runs on
  the accumulated evidence to choose the edges; each observation's own native edges then
  set positions through a robust consensus. A per-observation alignment is a nuisance
  parameter, reported for QA.
- Provenance: every model station traces to its observations, accepted and rejected.
- Each wall point rests on a native edge, on other observations, or on neither (bridged),
  and the result reports that.
- Soft pins, a straightened view (stations × offset) as an output for review, typed
  disconnected support across camera breaks, and rejection of fusion the registration
  does not support.
- Extension past the seed's end is the same extraction on new stations.

**Accept:** walls recovered where a single frame fails; the bead-moved case stays a
mismatch; degradation under injected registration error reported; the parallel line
flagged; on private sequences, the single-frame tracker, the single-frame extraction and
the fused extraction compared as aggregates.

### Cross-section annotation, `planned` ([#87])

A two-edge sub-pixel annotation mode in the `registration_eval` viewer, on magnified
interpolated profiles. Each section is annotated twice, blind, for the annotator's σ, and
stored in the manifest's `annotations`. A protocol covers how to choose and split
sections. Annotations of private data stay with it. The target is 50–100 sections per
sequence on development and held-out sequences.

**Accept:** annotations round-trip through the manifest, with tests on a synthetic
manifest; wall bias and σ reported against them, with the annotator's σ beside them.

---

## v0.5: online and metric

### Measurement plan, `planned` ([#89])

Compile a curve model into native caliper poses and search regions, with density set by
curvature and uncertainty, and execute it per frame given a pose. No panoramas.

**Accept:** runs on original incoming images, keeps irregular reference geometry, and
meets the latency budget its ADR states, in a bench; accuracy rows on the sequence
benchmark.

### Metric composition and scale diagnostics, `planned` ([#90])

Compose tracked-curve and fused results with `metric` while localization stays in native
coordinates. A common plane-height error cancels in frame-to-frame motion, so cross-view
agreement does not check scale; the path-length ratio against an external path does.

**Accept:** calibrated synthetic fixtures give consistent image and metric results; a
synthetic plane-height error is caught by the path ratio and not by cross-view agreement.
One reference supplies nominal geometry, never statistical tolerances.

### Appearance-assisted localization, `planned`, conditional ([#92])

Only if fusion leaves a measurable wrong-lock residue. Fit `a·A_s(u − δ) + b` per station
against the fused profiles, and compare with the gradient calipers.

**Accept:** fewer wrong locks on held-out sequences and adversarial synthetic gradients,
at a reported coverage and cost; mismatch stays explicit; the reference is fixed.

### Planar pose graph, conditional ([#91])

Not scheduled. It starts only if chained poses drift beyond a threshold on synthetic loop
sequences and a consumer still reports layout drift after adopting rig poses and fusion.
Then a backlog item and an update to
[ADR-0016](adr/0016-scope-what-we-do-not-build.md) come first.

---

## Interleaved tracks

These run between the milestones. `filter`'s Gaussian part moves forward into the
continuous profiles, and among the bindings `Edge1DDetector` and the level crossings come
first, because a consumer needs them.

### C1: accuracy coverage, `in progress`

The suite (`tests/accuracy.rs`) covers edges, the strip, rect, arc and radial calipers,
circle fitting, shape matching (translation, rotation, scale), rectified crops and
displacement and masked translation. Open rows:

| Operator | Sweep | Report |
|---|---|---|
| `fit_ellipse` | point count, arc extent, noise, outlier fraction | axis bias, centre σ |
| `LaserExtractor` | stripe width, saturation, tilt | centre bias, σ |

**Accept:** each row has an envelope pinned at about 1.5× the measured value and appears in
`docs/performance.md`.

### C2: blob features, `planned`

`ComponentStats` has label, pixel count, bounding box and centroid. Add second-order moments
(orientation, elongation), convex hull, minimum-area rectangle, circularity and
rectangularity.

**Accept:** analytic shapes (ellipse, rotated rectangle) recover their parameters within a
pinned tolerance; Python parity.

### B3: `filter`, `planned`

- Separable and recursive (Deriche / van Vliet) Gaussian, sliding-window box mean, an
  O(1)-per-radius histogram median, and grayscale erode/dilate/open/close/top-hat
  (van Herk–Gil-Werman). `edge/conv1d.rs` folds in here.
- It feeds the pyramid pre-smooth and illumination correction. The rank-filter scope stops
  at median ([ADR-0016](adr/0016-scope-what-we-do-not-build.md)).

**Accept:** each filter matches a naive reference bit-for-bit on seeded fixtures; median
cost is measured constant in radius.

### Lab catch-up, `planned`

- Move to `@vitavision/stage2d` 0.10 and `@vitavision/ui` 0.11. stage2d's `PointSet`,
  `ShapeEditor`, `HeatmapLayer` and cross-layer hit-test replace what the lab draws or
  hit-tests itself.
- Close our side of lab-ui issues [lab-ui#76], [lab-ui#77], [lab-ui#78] and [lab-ui#84].
- A Tauri `mosaic` command with a contract fixture. The desktop build has none and reports
  Mosaic as unavailable. Port `lab/backend/src/vm_lab/routers/mosaic.py`: grid auto-fit,
  nearest-centre priority, the `source_id` map, PNG encoding.

**Accept:** the lab on the latest packages with no local copy of a component they provide;
Mosaic works on the desktop.

[lab-ui#76]: https://github.com/VitalyVorobyev/lab-ui/issues/76
[lab-ui#77]: https://github.com/VitalyVorobyev/lab-ui/issues/77
[lab-ui#78]: https://github.com/VitalyVorobyev/lab-ui/issues/78
[lab-ui#84]: https://github.com/VitalyVorobyev/lab-ui/issues/84

### C3: bindings and CI, `planned`

- Generate the vm-python config conversions instead of mirroring them by hand
  (`src/config/*.rs`, about 1,850 lines).
- Python bindings for `laser`, `segment::watershed` / region growing and
  `build_graph_from_edgels`.
- A weekly miri job over every `unsafe` block.

---

## Later

- **Direct `calibration-rs` dependency** once its solver chain shares our nalgebra, through
  the range above or a move to 0.35. It replaces the `metric` mirror types
  ([ADR-0003](adr/0003-metric-calibration-bridge.md)).
- **A shared raster substrate.** Several sibling crates carry their own `ImageView`.
  Publishing `core::raster` as a common crate would end that, and is worth doing after the
  first release settles the API ([ADR-0004](adr/0004-core-raster-and-geometry.md)).

[#73]: https://github.com/VitalyVorobyev/vision-metrology/issues/73
[#74]: https://github.com/VitalyVorobyev/vision-metrology/issues/74
[#75]: https://github.com/VitalyVorobyev/vision-metrology/issues/75
[#76]: https://github.com/VitalyVorobyev/vision-metrology/pull/76
[#77]: https://github.com/VitalyVorobyev/vision-metrology/issues/77
[#78]: https://github.com/VitalyVorobyev/vision-metrology/issues/78
[#79]: https://github.com/VitalyVorobyev/vision-metrology/issues/79
[#80]: https://github.com/VitalyVorobyev/vision-metrology/issues/80
[#81]: https://github.com/VitalyVorobyev/vision-metrology/issues/81
[#82]: https://github.com/VitalyVorobyev/vision-metrology/issues/82
[#83]: https://github.com/VitalyVorobyev/vision-metrology/issues/83
[#84]: https://github.com/VitalyVorobyev/vision-metrology/issues/84
[#85]: https://github.com/VitalyVorobyev/vision-metrology/issues/85
[#86]: https://github.com/VitalyVorobyev/vision-metrology/issues/86
[#87]: https://github.com/VitalyVorobyev/vision-metrology/issues/87
[#89]: https://github.com/VitalyVorobyev/vision-metrology/issues/89
[#90]: https://github.com/VitalyVorobyev/vision-metrology/issues/90
[#91]: https://github.com/VitalyVorobyev/vision-metrology/issues/91
[#92]: https://github.com/VitalyVorobyev/vision-metrology/issues/92
