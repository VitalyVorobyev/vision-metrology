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
- **σ in samples is not constant along a bead stage.** A strip's endpoints are stored in
  `f32`, so its sample spacing, and σ in samples with it, differs between a bead tracker's
  stations by a few ulps. The caliper refills its kernel in place at most stations: 955 of
  the 1200 strips of the 300-station, 3-pass bench, about 37 ns each, 1.4% of the call.
  The radius can flip between neighbours (3↔4), and edge positions move by about 1e-4 px.
  A nominal-spacing option on strips, which converts σ with `step` as rect calipers do,
  would remove both.
- **Bead ends.** With a prior that runs onto the bead's square end, the end station can
  pair noise. In 1 trial of 270 this swung the curve's end by about 7 px, and the robust
  loss did not stop it.
  - Options: a per-end support check, a down-weighted or frozen end once its station
    loses support, or a reported end status.
  - Needs a design; tangential end extension is not observable with normal searches.
- **A distractor within about 2 px of a bead's edge merges with it.** The edge detector
  sees one edge, so `clearance` never sees the distractor, and the width reads about
  0.4 px short. `docs/bead.md` documents it. A fix would need a profile-shape check, such
  as edge symmetry, or a two-edge model.
- **Silent wrong locks on a textured surface.** On DamSegment cracks, 91% of the calls
  that fail to lock still report a support of at least 0.5. `tools/bead_eval/signals.py`
  measured every per-call signal against the lock, over 5400 calls on 300 paths:
  - the AUC is 0.71 for support, 0.68 for `longest_gap`, 0.59 for `center_rms` and 0.48
    for the median `BeadHit::confidence`;
  - `min_margin` 0.1, 0.2 or 0.3 does not help: support's AUC is 0.66–0.68 and the lock
    rate rises from 85% to 86%;
  - a support threshold that keeps 90% of the locked calls passes 65% of the others.

  A wrong lock is typically another dark line in the window, a pit, a shadow or a
  parallel crack, measured as well as the crack would be, so no signal of the pair or of
  the fit can see it. A fix needs evidence from outside the station: the profile's
  contrast and width against the previous frame's, a second line in the window as an
  ambiguity flag over the whole call, or an appearance check along the curve. Needs a
  design, and a dataset whose truth is the dark line, not a mask.
- **`Converged` almost never fires on real data.** 99.9% of DamSegment calls stop on
  `PassLimit`, because `max|d| < tol` (0.05 px) is far below a rough crack's station
  noise. With 10 passes the median rms correction levels off near 0.09 px, about a
  tenth of the residual, and the median largest near 0.33 px; 13% of those calls
  converge. At the third pass, the candidate tests hold for:
  - `rms(d) < tol`: 1% of calls; `rms(d) < 2·tol`: 13%;
  - `max|d| < residual_rms / 2`: 19%; `rms(d) < residual_rms / 10`: 3%.

  Options: an rms test, a test relative to the pass's residual, or keeping `max|d|` and
  documenting `PassLimit` as the normal outcome on rough edges. Changing what
  `Converged` means is an API-semantics decision for the user. The numbers are in
  `signals.py`'s report (`--passes 10` for the long runs).
- **Acquisition without a prior** is not built ([ADR-0018](adr/0018-tracked-curves.md)).
  The recommendation is not to build a Rust ridge module now. The evaluation
  (`tools/bead_eval/acquire_eval.py`, numbers in
  [`docs/performance.md`](../performance.md#finding-a-bead-without-a-prior)) showed that a
  detected path seeds the tracker as well as the truth does, so a ridge detector's
  sub-pixel output adds nothing; the open problem is choosing the bead among other
  lines. What would change the answer:
  - an inspection case with no prior source at all (no reference part, CAD or robot
    path, or taught frame), in a deployment that cannot run a Python or OpenCV detector
    for the first frame;
  - a selection rule that works on real images: width, polarity and contrast gates, or a
    region, that leave one candidate per bead. The evaluation is the place to try one;
  - `filter` landing with recursive Gaussian derivatives, which would make a
    full-image Hessian at a wide bead's scale cheap to build.

  If built, it should be its own module, never on the prior-driven path, with sub-pixel
  centre and width from the Hessian (Steger), scale selection over a real scale space
  (ADR-0016), a stated junction scope, accuracy rows and a bench.
- **`MeasureArc` obliquity** is checked against the arc tangent, which is right for
  features crossing the arc. A mode that measures the arc's own edge would check the
  radial direction.
- **Fuzzy / expected-position scoring.** Prefer an edge near the nominal geometry over an
  equally strong one elsewhere, by scoring candidates against an expected
  position/amplitude profile before `EdgeSelect` (HALCON `fuzzy_measure_pos`).
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

- **Files over the size cap** (invariant 14), measured as non-blank, non-`//` lines
  before the first `#[cfg(test)]`. `tools/check-invariants.py` reads this list: it fails
  on an offender missing from it and on a listed file back under the cap.
  - `crates/vision-metrology/src/contour/build.rs`

  Split a listed file when a change touches it.
- **`fit/circle.rs` hand-rolls a 3×3 solve** (`solve3`, Gaussian elimination with
  pivoting) that nalgebra provides. ADR-0002 says to use nalgebra's.
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
