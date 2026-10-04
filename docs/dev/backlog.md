# Backlog

Known debt and unscheduled features. Each item has enough context to be picked up cold.
When an item is scheduled it moves to [`roadmap.md`](roadmap.md); when it lands it is
deleted here.

## Shape matching

- **Calibrate `Contrast::FractionOfRange` on real data.** Its conversion assumes Scharr's
  gain of 16 on an ideal, unblurred step. Under `PreSmooth::Binomial121` or camera blur,
  real edges reach only part of that gain. Sweep it against `Contrast::Raw` on more than
  one dataset before the docs recommend it.
- **Anisotropic scale.** A 5-DOF search with a different refinement Jacobian. Design it
  from scratch when a use case exists; do not bolt it onto the 4-DOF pose structs.
- **Quantized directions and a SIMD score loop.** On the cluttered fixture most of the
  search is the candidate descent, which is per-pose cost
  ([`docs/performance.md`](../performance.md)). i8 directions with i16 dot products would
  cut it 3–4×, but the scores stop being bit-comparable to f32. Do it as its own change
  with a documented tolerance policy.
- **`PreparedScene` for several models on one scene.** With lazy tiled fields, the
  shareable per-scene work is a small fraction of one search. Revisit only if a
  multi-model station measures the per-model overhead as material.
- **A `rayon` feature.** The top-level angle sweep is the natural fan-out. Results must
  stay deterministic (stable reduction order).
- **Scale search in clutter with no prior.** `find_scale_invariant` needs a segmentable
  ROI or an approximate centre. With neither, the options are a coarse scan at the top
  pyramid level only, or a small bank of models pre-resampled at K scales.
- **Point-collapse score inflation at `scale < 1`.** Pinned by a test; the rejected fixes
  and what a fix must satisfy are in [ADR-0013](adr/0013-scale-invariance.md).

## Measurement

- **Pixel-unit settings convert with the nominal `step`** for rect, arc and radial
  calipers: σ, the `SmoothThenCentral` radius and the half-contrast flank distances and
  tolerance go to samples through `Placement::sigma_spacing`, not the real spacing that
  `Caliper::spacing` reports. Switching them moves the can-end reference numbers, so it
  needs a deliberate re-baseline.
- **`EdgeSelect::Strongest` breaks ties towards the later edge.** Choosing the earlier one
  would match `StrongestInOrder`'s tie rule (earlier wins); changing it alters existing
  results on exact ties.
- **`MeasureArc` obliquity** is checked against the arc tangent, which is right for
  features crossing the arc. A mode that measures the arc's own edge would check the
  radial direction.
- **Fuzzy / expected-position scoring.** Prefer an edge near the nominal geometry over an
  equally strong one elsewhere, by scoring candidates against an expected
  position/amplitude profile before `EdgeSelect` (HALCON `fuzzy_measure_pos`).
- **A bead/stripe tool on `measure`**, tracking a contour with calipers
  ([ADR-0008](adr/0008-calipers.md) says why it is not a caliper option).
- **Variation model (golden template).** Teach a per-pixel mean/σ band from N good parts
  warped to a common pose with `warp::Map`, then flag pixels outside it (HALCON
  `create_variation_model`). No design work started.
- **Laser profile in millimetres.** Laser image → stripe centreline → 3-D profile through a
  calibrated laser plane (`LaserPlane`, `laser_line_to_profile`). Needs the laser-plane
  export from calibration-rs.

## Contour

- **Contour → primitive segmentation.** Split a `GraphEdge` polyline at curvature
  breakpoints, classify runs as lines or arcs, and fit each with `fit` (HALCON
  `segment_contours_xld`). Estimated at about 300 lines over `contour` and `fit`.

## Mosaic

- **Exposure and gain compensation across cameras.** Estimate per-camera gain and offset
  from the overlap and apply them before compositing. Needs an accuracy fixture with a
  known exposure ratio.
- **The lab's `/api/mosaic` composites on `z = 0` of the calibration's reference frame.**
  For a rig whose reference frame is a camera, that plane passes through the camera
  centre and the homography is singular. `examples/birdseye_mosaic.rs` estimates the target
  plane from the images instead. Let `MosaicRequest` carry an explicit plane and return
  the overlap ZNCC.

## Python

- **Polarity strings disagree across the lab's transports (latent).** The lab's
  `MeasureConfigIn` accepts `bright_to_dark` / `dark_to_bright` / `either`, and
  `routers/measure.py` passes the value straight to `vm.MeasureConfig`, whose constructor
  accepts only `any` / `rising` / `falling` and raises on the others. The frontend never
  sends `polarity`, so the mismatch never triggers. vm-python's setters silently fall back
  to `any`, and the Tauri command maps both spellings. Align the contract on one spelling,
  and make the setters validate.
- **`ShapeMatch.matrix()` convention** needs a worked pixel → pose → pixel example in the
  vm-python README.
- **No `Edge1DDetector` or `LevelCrossing1D` binding.** 1-D detection is reachable only
  through `Caliper`.
- **No built wheel is import-tested** on any platform; CI builds the extension from
  source.

## Testing

- **Laser extractor over u16/f32.** The generic scan loop makes the full matrix cheap.
  u16 has two tests (rows against u8, transposed columns against gather) and f32 one (rows
  against u8).

## Code health

- **Files over the size cap** (invariant 14), measured as non-blank, non-comment lines
  before the test module:
  - `crates/vision-metrology/src/contour/build.rs`
  - `crates/vm-python/src/measure_py.rs`

  Split them when a change touches them.
- **The `serde` feature implies `matching`.** Serde derives on non-matching types such as
  `CaliperTrace` therefore pull in the matcher. Split model persistence into its own
  feature.

## Lab

- **The contour inventory renders every row.** Hundreds of rows are fine. Thousands, at a
  low `min_contrast`, need windowing or a visible cap. The canvas does not share the limit:
  stage2d's `PolylineSet` draws batched paths and picks through a spatial index.
- **`ContourOut` drops free data:** per-point strengths and the junction node ids. Add them
  if a question needs them.
- **`teach_preview` has no browser counterpart and no contract fixture.** It is covered by
  the transport test and Rust unit tests only.
- **The Library's "Run across the set" shows in the browser build,** where batch find is
  desktop-only, so pressing it reports an error instead of being hidden.
- **Desktop distribution is unsigned.** A real distribution needs a signing identity and
  macOS notarization.

## Waiting on upstream

- **corrmatch scale search.** corrmatch has rotation but no scale, so the ZNCC cross-check
  in `pose_audit` is valid at scale ≈ 1 only.
- **corrmatch `u16`/`f32`.** `corr` is `u8`-only because corrmatch is
  ([ADR-0011](adr/0011-corr-delegates-to-corrmatch.md)).
