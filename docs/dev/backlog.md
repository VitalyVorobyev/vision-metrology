# Backlog

Known debt and unscheduled features. Each item has enough context to be picked up cold.
When an item is scheduled it moves to [`roadmap.md`](roadmap.md); when it lands it is
deleted here. Measured numbers live in [`performance.md`](../performance.md), not here.

## Shape matching

- **Calibrate `Contrast::FractionOfRange` on real data.** Its conversion assumes Scharr's
  gain of 16 on an ideal, unblurred step. Under `PreSmooth::Binomial121` or camera blur,
  real edges reach only part of that gain. Sweep it against `Contrast::Raw` on more than
  one dataset before the docs recommend it.
- **Anisotropic scale.** A 5-DOF search with a different refinement Jacobian. Design it
  from scratch when a use case exists; do not bolt it onto the 4-DOF pose structs.
- **Quantized directions and a SIMD score loop.** On the cluttered fixture most of the
  search is the candidate descent ([performance](../performance.md)). i8 directions with
  i16 dot products would cut it 3–4×, but the scores stop being bit-comparable to f32. Do it
  as its own change with a documented tolerance policy.
- **`PreparedScene` for several models on one scene.** The shareable per-scene work is a
  small fraction of one search. Revisit only if a multi-model station measures the
  per-model overhead as material.
- **A `rayon` feature.** The top-level angle sweep is the natural fan-out. Results must
  stay deterministic (stable reduction order).
- **Scale search in clutter with no prior.** `find_scale_invariant` needs a segmentable
  ROI or an approximate centre. With neither, the options are a coarse scan at the top
  pyramid level, or a small bank of models pre-resampled at K scales.
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
- **σ in samples is not constant along a bead stage.** A strip's endpoints are stored in
  `f32`, so its sample spacing, and σ in samples with it, differs between a bead tracker's
  stations by a few ulps. The radius can flip between neighbours (3↔4), and edge positions
  move by about 1e-4 px. A nominal-spacing option on strips, which converts σ with `step`
  as rect calipers do, would remove both.
- **A distractor within about 2 px of a bead's edge merges with it.** The edge detector
  sees one edge, so `clearance` never sees the distractor, and the width reads about
  0.4 px short ([`docs/bead.md`](../bead.md)). A fix needs a profile-shape check, such as
  edge symmetry, or a two-edge model. Coupled pairing reads the same biased wall: the merge
  happens in the derivative, before any choice between edges.
- **Silent wrong locks on a textured surface.** On DamSegment cracks, 91% of the calls that
  fail to lock still report a support of at least 0.5, and no per-call signal separates
  them (support's AUC is 0.71). A wrong lock is typically another dark line, a pit, a shadow
  or a parallel crack, measured as well as the crack itself, so no signal of the pair or of
  the fit can see it. A fix needs evidence from outside the station: the profile's contrast
  and width against the previous frame's, an ambiguity flag from a second line in the window,
  or an appearance check along the curve. It needs a design and a dataset whose truth is the
  dark line, not a mask. Numbers: [performance](../performance.md#real-data-bead-tracking-on-damsegment-cracks).
- **Coupled pairing has no mask, and wants one pass.** The tracker reads the whole image,
  so an occluder inside a wall is evidence like any other. The path handles invalid samples,
  but `track` needs a mask argument and `ProfileSampler` evidence to use it. A second
  tracking pass re-centres the window on the chosen line and locked less often on
  DamSegment's development split, so the docs recommend `passes = 1` while `BeadTuning`'s
  default stays 3 for per-station pairing. A pairing-dependent default would be a breaking
  change to decide before the release. Prerequisite of [#73](https://github.com/VitalyVorobyev/vision-metrology/issues/73).
- **`Converged` almost never fires on real data.** 99.9% of DamSegment calls stop on
  `PassLimit`, because `max|d| < tol` (0.05 px) is far below a rough crack's station noise.
  Options: an rms test, a test relative to the pass's residual, or keeping `max|d|` and
  documenting `PassLimit` as the normal outcome on rough edges. Changing what `Converged`
  means is an API-semantics decision for the user. Numbers and the candidate tests are in
  [performance](../performance.md#bead-tracking-convergence).
- **Acquisition without a prior** is not built ([ADR-0018](adr/0018-tracked-curves.md)).
  A detected path seeds the tracker as well as the truth does, so a ridge detector adds
  nothing; the open problem is choosing the bead among other lines
  ([performance](../performance.md#finding-a-bead-without-a-prior-and-choosing-both-walls)). Revisit when:
  - an inspection case has no prior source at all (no reference part, CAD or robot path, or
    taught frame), in a deployment that cannot run a Python or OpenCV detector for the first
    frame;
  - a selection rule works on real images (width, polarity and contrast gates, or a region
    that leaves one candidate per bead);
  - `filter` gains recursive Gaussian derivatives, which would make a full-image Hessian at a
    wide bead's scale cheap. Any such module is its own module, never on the prior-driven
    path, with sub-pixel centre and width from the Hessian (Steger), real scale selection
    (ADR-0016), accuracy rows and a bench.
- **`MeasureArc` obliquity** is checked against the arc tangent, which is right for
  features crossing the arc. A mode that measures the arc's own edge would check the radial
  direction.
- **Fuzzy / expected-position scoring.** Prefer an edge near the nominal geometry over an
  equally strong one elsewhere, by scoring candidates against an expected position and
  amplitude profile before `EdgeSelect` (HALCON `fuzzy_measure_pos`). Trigger: a consumer
  reports a wrong lock that nominal geometry would have prevented.
- **Variation model (golden template).** Teach a per-pixel mean and σ band from N good parts
  warped to a common pose with `warp::Map`, then flag pixels outside it (HALCON
  `create_variation_model`). No design work started. Trigger: a consumer asks for it.
- **Laser profile in millimetres.** Laser image → stripe centreline → 3-D profile through a
  calibrated laser plane (`LaserPlane`, `laser_line_to_profile`). Trigger: the laser-plane
  export from calibration-rs.

## Edges

- **The default caliper derivative locks to the pixel grid.** A bilinear profile's
  derivative has knots at the pixel boundaries, so the default strip locks in the phase
  sweep at every blur. `Derivative::Continuous` does not (ADR-0021), at several times the
  cost. Making it the default re-pins every accuracy envelope, the can-end baseline and the
  lab's contract fixtures. Decide it with the release, from `docs/performance.md`'s costs.
- **`SubpixRefine::Centroid` locks** when the derivative peak is wider than its window: the
  truncated window pulls the centroid to the peak sample. A radius of at least 3× the
  response's σ keeps the locking index near that of the parabola (bounds in
  `docs/performance.md`). A window that follows the response width, or a baseline subtracted
  before the centroid, would remove the need to choose it.

## Contour

- **Contour → primitive segmentation.** Split a `GraphEdge` polyline at curvature
  breakpoints, classify runs as lines or arcs, and fit each with `fit` (HALCON
  `segment_contours_xld`). Estimated at about 300 lines over `contour` and `fit`.

## Mosaic

- **Exposure and gain compensation across cameras.** Estimate per-camera gain and offset
  from the overlap and apply them before compositing. Needs an accuracy fixture with a known
  exposure ratio.
- **The lab's `/api/mosaic` composites on `z = 0` of the calibration's reference frame.**
  For a rig whose reference frame is a camera, that plane passes through the camera centre
  and the homography is singular. `examples/birdseye_mosaic.rs` estimates the target plane
  from the images instead. Let `MosaicRequest` carry an explicit plane and return the overlap
  ZNCC.

## Python

- **`ShapeMatch.matrix()` convention** needs a worked pixel → pose → pixel example in the
  vm-python README.
- **No `Edge1DDetector` or `LevelCrossing1D` binding.** 1-D detection is reachable only
  through `Caliper`.
- **The Python projection is a copy of the Rust one.** `metric_py.rs` `project_plane_points`
  repeats the pose, perspective divide and distortion chain of
  `metric::homography::plane_to_camera_matrix`, and has no Rust test. The lab's mosaic code
  calls it "the exact forward geometry". Add `metric::plane_to_pixel` on the Rust side, bind
  it, and delete the loop.
- **Binding helpers are copied.** `to_rect` appears five times, `parse_border` and
  `parse_polarity` twice each, and `positive` twice with different messages. `POLARITIES` is
  defined twice in `vm-python/src/config` with different meanings. Move the validators to
  `config/mod.rs` and parameterise one border parser by type.

## Testing

- **Laser extractor over u16/f32.** The generic scan loop makes the full matrix cheap. The
  u16 and f32 agreement checks are one combined test (`laser/tests.rs`); split them when the
  matrix is extended.

## Code health

Findings from the 2026-10 review. Each names the copies to unify; the fixes are breaking
where the API is named, so they belong before the release or after it, by decision.

- **Border resolution exists in five copies.** The canonical one is `core::raster`'s
  `map_index`. `filter/fir.rs` and `filter/cascade.rs` have identical `resolve` bodies,
  `filter/integer.rs` a u8 copy, and `edge/conv1d.rs` re-implements Clamp, Constant and
  Reflect101 with `clamp_index`. A border change must be made five times today.
- **Parabolic peak is computed twice** with different degenerate behaviour: `edge/edge1d.rs`
  `parabola_offset` returns 0 below `1e-12`, and `core::geom::transform`
  `parabolic_peak_offset` returns `None` at `<= 1e-12`. The shared helper is scalar 1-D maths
  and does not belong in the geometry module.
- **1-D kernels in four places:** `filter::Kernel1D::gaussian`, the DoG in
  `edge/kernels1d.rs`, an inline unnormalised exp loop in `edge/edge1d.rs`, and
  `edge::convolve_f32`, which is public next to filter's crate-internal `separable_fir_f32`.
  Its only external caller is in-crate (`laser/tests.rs`).
- **Smoothing and median helpers are duplicated.** `smooth_binomial3` is in
  `edge/gradient.rs` and `edge/edge2d.rs`, about 25 lines each. `median` is in
  `metric/rig/solve.rs` and `median_of` in `edge/level1d.rs`. `corr/plane.rs` `downsample`
  repeats `pyr`'s 2×2 mean with a mask, and defines a private `Plane` parallel to
  `Image<f32>`.
- **Hand-rolled linear solves.** `solve3` in `fit/circle.rs`, `solve_weighted` in
  `fit/line.rs` and `gauss_newton_step` in `fit/circle.rs` do work that nalgebra provides
  (ADR-0002).
- **Invariant 19 is broken.** Single-type operations carry `_u8` suffixes: the morph
  operations in `morph/mod.rs`, `morph/thin.rs`, `morph/distance.rs`, the segment functions,
  and `filter::box_mean_u8` and `binomial_u8`. `morph` has four `*3x3_binary_u8` aliases of
  `Square(1)`. `sample_bilinear_f32` takes any `Pixel`, so its suffix lies, and it takes
  `BorderMode<f32>` while `sample_nearest` takes `BorderMode<T>`. Make these generic over
  `Pixel`, delete the aliases, and give both samplers one border type.
- **Invariant 6 is not met in `filter`.** Every call copies the image and builds fresh
  scratch. The module doc defers a reusable-scratch object to the hot-path work, which this
  list now tracks; see the roadmap's B3 entry.
- **Public output enums lack `#[non_exhaustive]`.** About 30 `pub` enums, among them
  `RejectReason`, `RigStep`, `ViewOutcome`, `LevelOutcome` and `Interp`. `BeadReject` and
  `TranslationStatus` have it. Breaking, so before the release.
- **Polarity has five vocabularies:** `EdgePolarity`, `PolaritySelect`, `BeadPolarity`,
  `BlobPolarity` and `matching::Polarity`. One vocabulary with a documented mapping is a
  breaking decision.
- **`corr` names one concept two ways.** `min_score` is `f32` in `TranslationConfig`,
  `Option<f32>` in `CorrConfig` and `f32` in `DisplacementConfig`. Levels are `max_levels` and
  `max_image_levels`. Invariant 10 says Option for no floor.
- **A public kernel type exposes mutable fields.** `DoGKernel1D` in `edge/kernels1d.rs` has
  public `sigma`, `radius`, `g` and `dg`, while `rebuild` keeps the normalisation. Make them
  private with getters.
- **Public result types with lifetimes break invariant 9:** `GradientBuffers<'a>` in
  `edge/edge2d.rs` and `TiledField<'a>` in `edge/gradient.rs`. Neither is bound in vm-python
  today. Record an exception in `system-design.md`, or return owned data.
- **Dead or leaky public items:** `measure::layout_object` (used only by its test and one doc
  line), `core::raster::Image::as_contiguous_slice_mut` (no caller), and public
  `unsafe fn get_unchecked*` on `Image`, which invariant 7 does not mention.
- **Error handling is mixed.** `sample_bilinear_f32` panics on an empty image with a
  non-constant border, `edge::conv1d` asserts, and `filter` returns `Error::InvalidConfig`.
  Invariant 8 asks for `Result`.
- **`pub mod core` shadows the std crate name.** It works because `error.rs` imports from a
  submodule. Rename before 1.0.
- **Files over the size cap** (invariant 14). Split a listed file when a change touches it.
  Two examples are also over 600 lines; the checker does not scan `examples/`, so they are
  not listed.
  - `crates/vision-metrology/src/contour/build.rs`
- **The `serde` feature implies `matching`.** Serde derives on non-matching types such as
  `CaliperTrace` pull in the matcher. Split model persistence into its own feature.

## Waiting on upstream

- **corrmatch scale search.** corrmatch has rotation but no scale, so the ZNCC cross-check
  in `pose_audit` is valid at scale ≈ 1 only.
- **corrmatch `u16`/`f32`.** `corr` is `u8`-only because corrmatch is
  ([ADR-0011](adr/0011-corr-delegates-to-corrmatch.md)).

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

[#73]: https://github.com/VitalyVorobyev/vision-metrology/issues/73
