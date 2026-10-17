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

- Done: CI packages both library crates on every PR (`release-crates.yml`), and builds
  each abi3 wheel, installs it into a clean venv and runs the README quick starts on Linux
  x86_64 and aarch64, macOS and Windows (`python-wheels.yml`). The wheel version comes from
  `Cargo.toml`. Both publish jobs exist, over OIDC, gated to version tags on this
  repository.
- The public Rust and Python surface is reviewed before the release fixes it, in four
  phases, each approved on its own. Done: phase 1 (scaffolding and dead surface) and
  phase 2 (honest structure: private model levels and stored-model types, the model
  config's `tuning`, explicit preludes and facade root). Left: phase 3 (`#[non_exhaustive]`,
  common traits, the registration status and region-check shapes; registration
  calibration has settled `TranslationConfig` with `region_check: Option<RegionCheck>`)
  and phase 4 (the Python surface).
- CHANGELOG cut, versions 0.2.0, install sections point at crates.io and PyPI.
- crates.io's first publish is manual (trusted publishing needs an existing crate). Then
  configure the trusted publishers (crates.io: both crates, `release-crates.yml`,
  environment `crates-io`; PyPI: `python-wheels.yml`, environment `pypi`) and set the
  repository variable `CRATES_IO_TRUSTED_PUBLISHING=true`. No stored tokens.

**Accept:** `vm-primitives` and `vision-metrology` 0.2.0 on crates.io with docs.rs builds;
the `vision-metrology` 0.2.0 wheel on PyPI imports and runs the README quick start on all
four targets.

---

## v0.3: native boundary evidence

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

### Public two-boundary benchmark, `in progress`: awaiting annotation ([#85])

The existing paired-edge datasets are research-only or carry no licence. We therefore
annotate cross-sections ourselves on CC BY 4.0 images, HRF vessels and DamSegment cracks,
and publish the annotations under CC BY 4.0. The stations are frozen: 120 per dataset,
36 dev and 84 test, by image. The manifests, the evaluation driver and its synthetic
end-to-end test are in place
([`tools/bead_eval`](../../tools/bead_eval/README.md#hand-annotated-cross-sections)).
Remaining work:
- a person annotates both datasets in two blind passes, about 4 hours;
- the settings are frozen on dev, including HRF's provisional thresholds;
- test runs once;
- the wall and width error go into `performance.md`, for the tracker and the coupled mode,
  against the annotation mean and beside the annotator's σ.

**Accept:** licence recorded; dev/test split with the settings frozen on dev; numbers with
bootstrap intervals in `performance.md`.

---

## v0.4: many observations, one model

### Synthetic sequence benchmark, `in progress` ([#86])

A deterministic generator writing `sequence.json` with truth: a curved ribbon with
asymmetric, variable-width walls on a moving textured substrate, one to three affine
views with a handover, camera-fixed highlights, occlusions, a parallel line, a
distractor also present in the blank, missing observations, an actual gap, injected
registration error, and a bead that moves relative to its substrate. No public dataset
has many registered observations of one curve, so this is the public stand-in.

The generator is available in [`tools/sequence_bench/`](../../tools/sequence_bench/README.md),
deterministic and in CI, with presets for #73, #74 and #78. The rig poses (#74) report on
it; what remains is the fusion's and the bead ends' numbers, which they report as they land.

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

These run between the milestones. Among the bindings `Edge1DDetector` and the level
crossings come first, because a consumer needs them.

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

- **Direct `calibration-rs` dependency.** It replaces the `metric` mirror types
  ([ADR-0003](adr/0003-metric-calibration-bridge.md)). The nalgebra range already lets
  the two share one nalgebra ([ADR-0002](adr/0002-dependency-and-toolchain-policy.md)),
  but until calibration-rs's solver chain moves to 0.35 our own lock would sit on 0.34.
- **A shared raster substrate.** Several sibling crates carry their own `ImageView`.
  Publishing `core::raster` as a common crate would end that, and is worth doing after the
  first release settles the API ([ADR-0004](adr/0004-core-raster-and-geometry.md)).

[#73]: https://github.com/VitalyVorobyev/vision-metrology/issues/73
[#75]: https://github.com/VitalyVorobyev/vision-metrology/issues/75
[#78]: https://github.com/VitalyVorobyev/vision-metrology/issues/78
[#79]: https://github.com/VitalyVorobyev/vision-metrology/issues/79
[#85]: https://github.com/VitalyVorobyev/vision-metrology/issues/85
[#86]: https://github.com/VitalyVorobyev/vision-metrology/issues/86
[#89]: https://github.com/VitalyVorobyev/vision-metrology/issues/89
[#90]: https://github.com/VitalyVorobyev/vision-metrology/issues/90
[#91]: https://github.com/VitalyVorobyev/vision-metrology/issues/91
[#92]: https://github.com/VitalyVorobyev/vision-metrology/issues/92
