# Roadmap

Open work, in priority order, with acceptance criteria. Status values: `planned`,
`in progress`. When an item lands it leaves this file: user-visible changes go to
[`CHANGELOG.md`](../../CHANGELOG.md), and decisions to an ADR. Known debt that is not
scheduled lives in [`backlog.md`](backlog.md).

## Where the library stands

The measurement chain runs end to end on real data: rectify → locate → fixture → calipers
→ robust fit → millimetres → pass/fail ([system design](system-design.md)). The work ahead
is the bead tracker, then the first release, then the remaining gaps: accuracy coverage,
blob features, `filter`, the lab's package catch-up and bindings.

---

## M10: bead tracking, `in progress`

`measure::BeadTracker` refines a prior centreline from caliper edge pairs and measures on
the refined curve ([ADR-0018](adr/0018-tracked-curves.md)), with Python parity and a user
guide. The release waits for the rest, so the API review covers it.

- `diagnostics::explain_bead`, which returns `track`'s result to the bit, with Python
  parity. The run already goes through a probe for it.
- Accuracy rows for the final centre, the final width and the tracked curve. A
  convergence-basin sweep against translation, rotation, smooth and local deformation of
  the prior.
- A bench at 100–300 poses: search length, 1 vs 3 passes, valid vs partly invalid.
- An example with an overlay, and a Python example.
- Offline tooling under `tools/`:
  - a robustness evaluation on DamSegment crack masks, whose truth is pixel-level only;
  - an evaluation of ridge-based acquisition (no prior), ending in a recorded decision.

**Accept:**
- each accuracy row has an envelope at about 1.5× the measured value;
- the basins and bench numbers are in `docs/performance.md`;
- the example runs in CI;
- the acquisition decision is recorded in ADR-0018.

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
displacement. Open rows:

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

## Later

- **Direct `calibration-rs` dependency** once its solver chain moves to nalgebra 0.35. It
  replaces the `metric` mirror types ([ADR-0003](adr/0003-metric-calibration-bridge.md)).
- **A shared raster substrate.** Several sibling crates carry their own `ImageView`.
  Publishing `core::raster` as a common crate would end that, and is worth doing after the
  first release settles the API ([ADR-0004](adr/0004-core-raster-and-geometry.md)).
