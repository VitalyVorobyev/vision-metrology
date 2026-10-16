# Roadmap

Open work, in priority order, with acceptance criteria. Status values: `planned`,
`in progress`. When an item lands it leaves this file: user-visible changes go to
[`CHANGELOG.md`](../../CHANGELOG.md), and decisions to an ADR. Known debt that is not
scheduled lives in [`backlog.md`](backlog.md).

## Where the library stands

The measurement chain runs end to end on real data: rectify → locate → fixture → calipers
→ robust fit → millimetres → pass/fail ([system design](system-design.md)). Beads and
stripes are tracked along a prior curve and measured station by station
([ADR-0018](adr/0018-tracked-curves.md)). The work ahead is the first release, whose API
review covers the bead tracker, then the remaining gaps: accuracy coverage, blob
features, `filter`, the lab's package catch-up and bindings.
Masked translation registration is available as a general-purpose `corr` operator;
reference learning remains a separate sequence of algorithmic work below.

---

## R: first release (v0.2.0), `in progress`

- CI packages both library crates (`cargo publish --dry-run`) and installs and tests each
  built wheel on Linux x86_64 and aarch64, macOS and Windows.
- The public Rust and Python surface is reviewed before the release fixes it.
- CHANGELOG cut, versions 0.2.0, install sections point at crates.io and PyPI.
- crates.io's first publish is manual (trusted publishing needs an existing crate); later
  releases publish from CI over OIDC, PyPI over a trusted publisher. No stored tokens.

**Accept:** `vm-primitives` and `vision-metrology` 0.2.0 on crates.io with docs.rs builds;
the `vision-metrology` 0.2.0 wheel on PyPI imports and runs the README quick start on all
four targets.

## C1: accuracy coverage, `in progress`

The suite (`tests/accuracy.rs`) covers edges, the strip, rect, arc and radial calipers,
circle fitting, shape matching (translation, rotation, scale), rectified crops and
displacement and masked translation. Open rows:

| Operator | Sweep | Report |
|---|---|---|
| `fit_ellipse` | point count, arc extent, noise, outlier fraction | axis bias, centre σ |
| `LaserExtractor` | stripe width, saturation, tilt | centre bias, σ |

**Accept:** each row has an envelope pinned at about 1.5× the measured value and appears in
`docs/performance.md`.

## C2: blob features, `planned`

`ComponentStats` has label, pixel count, bounding box and centroid. Add second-order moments
(orientation, elongation), convex hull, minimum-area rectangle, circularity and
rectangularity.

**Accept:** analytic shapes (ellipse, rotated rectangle) recover their parameters within a
pinned tolerance; Python parity.

## B3: `filter`, `planned`

- Separable and recursive (Deriche / van Vliet) Gaussian, sliding-window box mean, an
  O(1)-per-radius histogram median, and grayscale erode/dilate/open/close/top-hat
  (van Herk–Gil-Werman). `edge/conv1d.rs` folds in here.
- It feeds the pyramid pre-smooth and illumination correction. The rank-filter scope stops
  at median ([ADR-0016](adr/0016-scope-what-we-do-not-build.md)).

**Accept:** each filter matches a naive reference bit-for-bit on seeded fixtures; median
cost is measured constant in radius.

## Lab catch-up, `planned`

- Move to `@vitavision/stage2d` 0.10 and `@vitavision/ui` 0.11. stage2d's `PointSet`,
  `ShapeEditor`, `HeatmapLayer` and cross-layer hit-test replace what the lab draws or
  hit-tests itself.
- Close our side of lab-ui issues [#76], [#77], [#78] and [#84].
- A Tauri `mosaic` command with a contract fixture. The desktop build has none and reports
  Mosaic as unavailable. Port `lab/backend/src/vm_lab/routers/mosaic.py`: grid auto-fit,
  nearest-centre priority, the `source_id` map, PNG encoding.

**Accept:** the lab on the latest packages with no local copy of a component they provide;
Mosaic works on the desktop.

[#76]: https://github.com/VitalyVorobyev/lab-ui/issues/76
[#77]: https://github.com/VitalyVorobyev/lab-ui/issues/77
[#78]: https://github.com/VitalyVorobyev/lab-ui/issues/78
[#84]: https://github.com/VitalyVorobyev/lab-ui/issues/84

## C3: bindings and CI, `planned`

- Generate the vm-python config conversions instead of mirroring them by hand
  (`src/config/*.rs`, about 1,850 lines).
- Python bindings for `laser`, `segment::watershed` / region growing and
  `build_graph_from_edgels`.
- A weekly miri job over every `unsafe` block.

---

## Reference learning algorithms, `planned`

The goal is one geometric and appearance model of a tracked curve, such as a bead, built
from many observations, and measurements from that model. These are reusable operators,
validated with adhesive beads and other tracked curves. Acquisition, annotation stores,
recipe orchestration, tolerances and defect decisions remain application
responsibilities, and private datasets only test the operators: their adapters write a
neutral sequence manifest (`tools/registration_eval`). The motion models are not expanded
further: masked translation and the experimental local similarity are the registration
this work builds on. Registration must not depend on the tracked object: callers mask it
out, and a test checks that an object moving relative to its substrate is flagged, not
absorbed as motion. Each stage starts with a finite design, and the next begins only after
its prerequisite passes synthetic accuracy, Python parity and performance gates.

1. **Registration calibration, `planned`; before R, which freezes `TranslationConfig`.**
   The similarity design is in [registration-stage2.md](registration-stage2.md).
   - Diagnose first. Synthetic: a true translation or similarity on DoG-band texture at
     320×97 with realistic noise, measuring the current false-reject rate. Private: on
     development sequences A and B, disable the regional check and compare `region_error`
     with reciprocal closure and anchor corrections.
   - Replace the fixed-px regional threshold with a covariance-normalized test.
     Translation: χ² on pairwise regional shift differences, from the summed regional
     covariances plus a model-bias floor in px². Similarity: compare each regional fit
     with the global one at that region's own centroid and extent, without
     extrapolation. Config: `region_check: Option<RegionCheck { max_chi2, bias_floor_px }>`.
   - Convergence: freeze the observation set at the start of refinement, fix the Huber
     scale after the first iteration, and test convergence on the motion step only.
     Reproduce the failure first with a synthetic overlap/mask fixture.
   - Give the similarity initializer a search-only path.

   **Accept:** synthetic false rejects ≤ 1% at the stated noise; an observable 0.01 rad
   rotation (translation) and a mild affine (similarity) are still rejected; before/after
   coverage on the development sequences, with reciprocal-closure p95 no worse; the
   held-out sequences reported with the settings frozen; benches updated.
2. **Coupled two-boundary extraction, single observation, `planned`.** A global choice of
   both walls along a curve, instead of `pairing::choose`'s choice at each station alone;
   its max-marginals give an ambiguity signal for the whole call, which no station-level
   signal provides (backlog, silent wrong locks). Settled in its own ADR.
   - Prototype in Python in `tools/bead_eval` on DamSegment first, in two variants: a
     sparse Viterbi over the caliper's candidate edge pairs per station, plus an "absent"
     state; and a dense DP over profile costs. Measure lock rate, false locks and a
     max-marginal ambiguity AUC against today's tracker. It goes to Rust only if the
     global choice lowers wrong locks at equal coverage.
   - The Rust operator takes a reference polyline, stations from `measure/bead/curve.rs`,
     masked profiles along the normals and signed DoG edge costs. States are (left bin,
     width bin) plus an absent state with a cost; transitions bound |Δl| and |Δr| with
     quadratic smoothness and width-change penalties, in ADR-0018's length-scale
     parameters, so the result does not depend on the station spacing. It reports
     boundaries refined subpixel on native pixels with the existing caliper and `Edge1D`,
     a max-marginal margin per station and for the whole call, and typed absence. It
     plugs into `BeadTracker` as an alternative pairing mode, with `explain_bead` parity
     through the existing probes.
   - Fixtures: `examples/common/ribbon.rs`'s scenes as a shared fixture, with asymmetric
     variable-width walls, a distractor within 2 px, a continuous parallel line, gaps,
     occlusion and highlights.

   **Accept:** accuracy rows for boundary bias and σ; the backlog's 2 px distractor and
   parallel-line wrong locks are resolved, or flagged by the whole-call margin; DamSegment
   shows fewer false locks at equal lock rate; Python parity, a bench and the 600-line
   file cap.
3. **Multi-observation fusion and extension, `planned`.** An evidence stack holds, per
   station, observations (id, profile samples, validity, weight), each sampled from its
   own native frame through the caller's registered map. A robust combination of the
   per-observation costs, weighted by registration covariance and visibility, feeds stage
   2's extraction; provenance records which observations support each station. Extension
   past the seed's end is the same extraction on new stations. Fusion runs only on
   segments that registration supports. A pose chain over the reciprocal anchors stays in
   tooling. A synthetic sequence generator renders known motion, a bead with asymmetric
   walls, per-frame occlusions and highlights, injected registration error, and a bead
   that moves relative to its substrate. Private validation annotates 50–100
   cross-sections of a development sequence and a held-out one, read through the
   manifest's `annotations`, and compares the single-frame tracker, the single-frame
   extraction and the fused extraction.

   **Accept:** synthetic walls are recovered where a single frame fails; the bead-moved
   case is flagged; degradation under injected registration error is graceful and
   reported; the private comparison is reported as aggregates.
4. **Appearance-assisted localization, `planned`.** Fit a learned cross-section
   appearance `a·A_s(u−δ)+b` against the fused profiles, and compare it with the gradient
   calipers under photometric variation and distractors.

   **Accept:** held-out sequences and adversarial synthetic gradients show fewer wrong
   locks at a reported coverage/cost tradeoff; appearance mismatch remains explicit and
   the reference is fixed.
5. **Adaptive measurement placement, `planned`.** Compile curves and boundary evidence
   into local caliper poses and search regions on native images, with density controlled
   by curvature and uncertainty.

   **Accept:** measurements execute on original incoming images, preserve irregular
   reference geometry and meet a stated native latency budget without per-frame
   panoramas.
6. **Calibration composition, `planned`.** Compose with `metric` for offline reference
   geometry and reporting.

   **Accept:** calibrated synthetic fixtures give consistent image/metric results while
   online localization and measurements stay in native image coordinates. One reference
   supplies nominal geometry, never statistical tolerances.

---

## Later

- **Direct `calibration-rs` dependency** once its solver chain moves to nalgebra 0.35. It
  replaces the `metric` mirror types ([ADR-0003](adr/0003-metric-calibration-bridge.md)).
- **A shared raster substrate.** Several sibling crates carry their own `ImageView`.
  Publishing `core::raster` as a common crate would end that, and is worth doing after the
  first release settles the API ([ADR-0004](adr/0004-core-raster-and-geometry.md)).
