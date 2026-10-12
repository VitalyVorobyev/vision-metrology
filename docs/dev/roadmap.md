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

These are reusable operators, validated with adhesive beads and other tracked curves.
Acquisition, annotation stores, recipe orchestration, tolerances and defect decisions
remain application responsibilities. Each stage starts with a finite API/algorithm design;
the next stage begins only after its prerequisite passes synthetic accuracy, Python
parity and performance gates. Existing release, accuracy and filter work retains its scope.

1. **Similarity registration and anchored sequences.** Build on masked translation;
   support observable rotation/scale and keyframe consistency, with explicit rank and
   drift diagnostics. **Accept:** known synthetic similarity transforms recover within
   pinned envelopes; long-sequence anchors reduce accumulated error without inventing
   evidence in unobservable directions. Translation-only results retain their contract.
2. **Curve propagation and extension.** Build on reliable registration and existing
   prior-driven measurement. Preserve arc/material identity while new portions enter
   view; independently supported walls and genuine gaps remain distinct.
   **Accept:** synthetic sequences longer than the seed segment recover new visible
   geometry, handle disappearance and preserve the original annotated segment's identity.
3. **Multi-frame reference learning.** Pool native-image evidence in a global frame;
   learn independent boundaries and retain support, uncertainty and source provenance.
   Panoramas visualize geometry rather than supply blurred width measurements.
   **Accept:** asymmetric variable-width fixtures recover both walls without symmetry
   assumptions, including noise, occlusion and unsupported intervals.
4. **Appearance-assisted localization.** Compare learned local profiles with gradient
   measurements under photometric variation and distractors. **Accept:** held-out
   sequences and adversarial synthetic gradients show fewer wrong locks at a reported
   coverage/cost tradeoff; appearance mismatch remains explicit and the reference is fixed.
5. **Adaptive measurement placement.** Compile curves and boundary evidence into local
   caliper poses and search regions, with density controlled by curvature and uncertainty.
   **Accept:** measurements execute on original incoming images, preserve irregular
   reference geometry and meet a stated native latency budget without per-frame panoramas.
6. **Calibration and reference comparison.** Compose with `metric` for offline reference
   geometry and reporting. **Accept:** calibrated synthetic fixtures give consistent
   image/metric results while online localization and measurements stay in native image
   coordinates. One reference supplies nominal geometry, never statistical tolerances.

---

## Later

- **Direct `calibration-rs` dependency** once its solver chain moves to nalgebra 0.35. It
  replaces the `metric` mirror types ([ADR-0003](adr/0003-metric-calibration-bridge.md)).
- **A shared raster substrate.** Several sibling crates carry their own `ImageView`.
  Publishing `core::raster` as a common crate would end that, and is worth doing after the
  first release settles the API ([ADR-0004](adr/0004-core-raster-and-geometry.md)).
