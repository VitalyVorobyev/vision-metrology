# Roadmap

Where the project is going, track by track, with acceptance criteria. Status values:
`planned` → `in progress` → `done` (with the landing PR). Superseded plans are rewritten,
not appended. Background and rationale live in [`system-design.md`](system-design.md);
deferred items and known debt in [`backlog.md`](backlog.md).

## Completed foundations

| Milestone | PR |
|---|---|
| 12 → 3 crate consolidation, CI unblocked, docs realigned | #13 |
| Dependency bumps, licensing, real CI (6 jobs), 4 algorithm bug fixes, 198 tests | #15 |
| Shape-based object detection replacing the chamfer matcher; validated 256/256 on canend | #18 |
| B-track closure, extractor split, serde model persistence | #20 |
| Detection performance: lazy tiled direction fields, 7.8 → 3.46 ms full-360° | #21, #23 |
| Visual diagnostics + corrmatch external validation | #22 |

## Where this is going

The library can **find** a part and **see** its edges. It cannot yet **measure** one: there
is no caliper, no line fit, no circle fit, no general filter, no image warp, no
pixel→millimetre bridge. Tracks B and C close that, so the whole chain works end to end:

```
acquire → rectify → locate (matching) → fixture (pose) → measure (calipers)
        → fit primitives (robust) → gauge in millimetres → pass / fail
```

---

## Track A — v0.2 substrate reset — `done`

One breaking wave over the whole surface, taken while nothing is published.

**Landed:**
- **`Pixel`**, a sealed trait over `u8`/`u16`/`f32`. Every `_u8`/`_u16`/`_f32` triplet
  collapsed into one generic entry point: **40 entry points → 16**. `laser`'s private
  `ScanPixel` deleted.
- **`Pyramid`** generic over `Pixel`, `unsafe`-free (5 blocks removed, and 1.1% *faster*
  than the raw-pointer version it replaced), `downsample.rs` 489 → 189 lines. Optional
  `PreSmooth::Binomial121` closes the primitive half of R3. `level_to_base` /
  `base_to_level` are now the single implementation of invariant 2.
- **`Point2f` / `Vec2f` are nalgebra aliases**; 7 conversion functions and ~250 lines of
  hand-written operators gone. `Vec2fExt::normalized_or_zero` guards the NaN trap this
  introduced.
- **Module hygiene**: no glob re-exports, `prelude` on both crates, every domain module a
  default-on feature, CI feature-matrix job. `Error` is `#[non_exhaustive]`.
- **Six defects fixed**, three of which were silently corrupting measurements:
  - `multiscale` mapped level-*l* edgels without the `(2^l − 1)/2` term, biasing every
    fitted circle centre 0.07–0.10 px. **Module deleted** — its dedup and its `scale`
    annotation were unsound too.
  - LSD carried a second, divergent downsample and mapped endpoints back without the
    half-pixel term: −0.50 px at the default config, −0.95 px on odd widths. It now uses
    `pyr`, and its two fictional config fields (`scale`, which meant 0.5 not 0.8, and
    `sigma_scale`, never read) are replaced by honest ones.
  - `downsample2x2_mean_*_into` guarded a release-mode out-of-bounds write with a
    `debug_assert!`; the size contract is now checked and returns `Error::SizeMismatch`.
  - `Pyramid::ensure` was public and could construct 0×0 levels.
  - `[profile.release]` was untuned despite a ~5 ms target: `lto = "thin"`,
    `codegen-units = 1`, worth −2.6% / −1.9%.
  - Four stale claims in the persistent-context docs.

**Verified:** all gates green including the feature matrix and `cargo doc -D warnings`;
`shape_find_1280x1024_360deg` 3.42 → 3.36 ms and clutter 6.61 → 6.42 ms; canend set1
150/150 across three folders with ZNCC p50 0.912–0.961, identical to four decimals.

---

## Track D — v0.3 API reset — `done`

The last wave of breaking changes taken while the crates are unpublished, run as six
commits, each with the full gate set green before the next started. Nothing about an
algorithm moved; everything about how the API *states* things did. Detail and reasoning
are in [`system-design.md`](system-design.md).

1. **Visibility sweep.** The flat domain re-export block left the crate root at the
   crate root; every type had two paths, in violation of invariant 17. Removed, `prelude`
   extended to `fit` / `measure` / `segment`, and nine leaked internals pulled back:
   the laser stage functions (a `_u8`/`_u16`/`_f32` triplet that also broke invariant 19),
   `MAX_KERNEL_PTS`, the `downsample2x2_mean*` free functions, `create_shape_model`,
   `transform_point_iso`, the `edge` submodule paths, and `match_point_scores` (now only
   `matching::diagnostics::match_point_scores`).
2. **Config split, sentinels, `Contrast`.** `ShapeSearchConfig` 14 flat fields → 8 + a
   nested `ShapeSearchTuning`; `LaserExtractConfig` likewise. Every `0`/`0.0` = "auto"
   became `Option<NonZeroUsize>` / `Option<f32>` / a named enum (`Hysteresis`,
   `CenterSmoothing`) — **invariant 10 rewritten**. `min_contrast` gained a unit:
   `Contrast::{Raw, FractionOfRange}`.
3. **Result honesty.** `Caliper::measure_checked` → `measure` returning
   `Result<&[MeasureEdge], RejectReason>`, with the lossy twin deleted;
   `MetrologyModel::apply` returns one `Result` per object, in object order;
   `hits()` folded into `MetrologyResult`.
4. **One batched model-format bump** (2 → 3): opaque `save`/`load` + `to_bytes`/
   `from_bytes`, `FORMAT_VERSION` internal, `ModelPoint`/`ShapeModelLevel` read-only
   through accessors, and **backlog R3 closed** — the pyramid `PreSmooth` is stored in
   the model and the search reads it from there, so invariant 3 cannot be violated.
5. **`core` split into `raster` (zero nalgebra) and `geom`.** Module split only, both
   private, no public path changed. The audit came out clean: the raster signatures
   already took bare `f32` coordinates.
6. **`shape` → `lsd`** (module + feature), and `DirectionField`'s stateful tiling protocol
   became a `TiledField<'a>` session that owns `ensure_rect` and cannot be handed the
   wrong image.

**Verified:** all gates green, per-feature clippy by hand (which caught one stale
`#[cfg(feature = "serde")]` that `--all-features` could not); every self-asserting example
passes; `match_shape` held or improved on all six cases (360° 3.408 → 3.370 ms,
scale sweep 17.57 → 16.83 ms); canend set1 `inspect_canend` **100/100** measured
(dome mean 365.237 px σ 0.282, dark 365.696 px σ 0.307) and shape matching 150/150 found
across dome / bright / dark with dome p50 **0.998**, unchanged to three decimals.

**Deliberately not done here:** README and `docs/*.md` still describe the pre-reset
surface in places (that is the docs wave), and `vm-python` mirrors the new configs only
well enough to compile and pass its tests — full parity, `.pyi` stubs and a `Contrast`
mode selector are the Python wave (**done**, see Track D.1 below).

### D.1 — vm-python parity wave — `done`

Four commits, gates green before each: config mirror (nested `ShapeSearchTuning`, `roi`/
`angle_range`/`scale_range`, a tagged `Contrast` pyclass rather than a bare float),
`measure`/`fit_line`/`contour`/`morph` bindings (`Caliper` raising `MeasureRejected`,
`MetrologyModel` returning `MetrologyResult | MetrologyError` per object), dtype dispatch
replacing every `_u8` name (`EdgeDetector`, `LsdDetector`, `ShapeModel`/`ShapeMatcher`
all accept `uint8`/`uint16`/`float32`), and `.pyi` + `py.typed` wired into the wheel via
maturin's `python-source` mixed layout — which turned out to need a bridging `__init__.py`
(the compiled extension lands as a *submodule* of the same-named package, not merged into
it; verified by installing the actual built wheel, not just compiling). `laser` and
`segment::watershed`/region-growing got no binding this wave — recorded in
`docs/backlog.md` under Python, not silently skipped.

---

## Track B — the measurement spine — `planned`

Each module ships with its own bench, doc example, Python parity, and an accuracy entry
in Track C1.

### B1 — `fit`: robust primitive fitting — `done`
Absorbs `shape/{conic,fit_conic,ransac,fitter}`. Adds what is simply missing: **there is
no line fit and no circle fit today**, the two most common metrology measurements.

```rust
pub enum RobustLoss { None, Huber { k: f32 }, Tukey { c: f32 } }
pub struct Fit<M> { pub model: M, pub rms: f32, pub max_dev: f32,
                    pub n_used: usize, pub inliers: Vec<u32> }

pub fn fit_line(&[Point2f], &FitConfig)    -> Result<Fit<Line2f>, Error>;    // TLS + IRLS
pub fn fit_circle(&[Point2f], &FitConfig)  -> Result<Fit<Circle2f>, Error>;  // Taubin → geometric
pub fn fit_ellipse(&[Point2f], &FitConfig) -> Result<Fit<Ellipse2f>, Error>; // Fitzgibbon → geometric
```

Two invariants to adopt with it: **storage f32, accumulation f64**, and **every measurement
reports its residual**. The geometric (orthogonal-distance) refinement stage is what
separates a metrology fit from an algebraic one; the current ellipse path stops at
algebraic + RANSAC.

**Landed.** Taubin → Gauss–Newton on the true residual; `fit_line` (TLS + IRLS + RANSAC),
`fit_circle`, `fit_ellipse`, all reporting `rms` / `max_dev` / `n_used`. Two findings
recorded in system-design: Tukey needs graduated non-convexity to start from a contaminated
fit, and a gross outlier flips a TLS line's principal axis, which only RANSAC recovers.
Short arcs (30°–180°) stay within 0.05 px.

### B2 — `measure`: calipers and the metrology model — `done`
A geometric wrapper over the existing `Edge1DDetector` and `sample_bilinear_f32`.

```rust
pub struct MeasureRect { center: Point2f, angle: f32, half_len: f32, half_width: f32 }
pub struct MeasureArc  { center: Point2f, radius: f32, angle_start: f32,
                         angle_extent: f32, half_width: f32 }

impl MeasureHandle {
    pub fn pos<P: Pixel>(&mut self,   img: &ImageView<'_, P>) -> &[MeasureEdge];
    pub fn pairs<P: Pixel>(&mut self, img: &ImageView<'_, P>) -> &[MeasurePair];
}

impl MetrologyModel {
    /// `fixture` is `ShapeMatch::pose` — that is the whole point.
    pub fn apply<P: Pixel>(&mut self, img: &ImageView<'_, P>,
                           fixture: &Similarity2f) -> Vec<Result<MetrologyResult, Error>>;
}
```

For each position along the measurement axis, average `2·half_width+1` bilinear samples
perpendicular into a 1-D profile, then hand it to the existing detector. `find` → `pose` →
`apply` → fitted circle + rms → `metric` → millimetres closes the loop from "I found the
part" to "the hole is 12.03 mm ± 0.01".

**Landed.** `Caliper` over rect / arc / **radial** placements, typed `RejectReason`, an
obliquity gate and sub-pixel stepping (the last three adapted from the `rtvt-pano` caliper).
`MetrologyModel` applies nominal geometry at a fixture pose and fits robustly.

The finding that shaped it: a rect caliper averages along a *chord*, which biases a curved
edge inward — 39.88 px on a nominal-40 disc. `MeasureRadial` averages along the arc:
39.990 px. See system-design.

`examples/inspect_canend` runs the whole chain on real frames: 100/100 measured across two
lighting conditions, σ ≈ 0.3 px on the rim radius, every caliper surviving the robust fit.

### B3 — `filter`: the absent workhorse
Separable and recursive (Deriche / van Vliet) Gaussian, sliding-window box mean, an O(1)-per-radius
histogram **median**, grayscale erode/dilate/open/close/tophat (van Herk–Gil-Werman).
`edge/conv1d.rs` folds in here. Feeds the pyramid pre-smooth and illumination correction.
Scope is deliberately median only, not the full rank-order family — see `backlog.md`.

**Accept:** each filter matches a naive reference bit-for-bit on random fixtures; median is
O(1) in radius, measured rather than asserted.

### B4 — `warp`: build once, apply per frame — `done`
```rust
pub enum Interp { Nearest, Bilinear }
impl Map {
    pub fn affine(w: usize, h: usize, m: &Affine2f) -> Self;
    pub fn projective(w: usize, h: usize, h_: &Projective2f) -> Self;
    pub fn polar(center: Point2f, r: Range<f32>, phi: Range<f32>, w: usize, h: usize) -> Self;
    pub fn from_fn(w: usize, h: usize, f: impl Fn(f32, f32) -> (f32, f32)) -> Self;
    pub fn apply<P: Pixel>(&self, src: &ImageView<'_, P>, dst: &mut [P],
                            interp: Interp, border: BorderMode<P>) -> Result<(), Error>;
    pub fn apply_with_mask<P: Pixel>(&self, src: &ImageView<'_, P>, dst: &mut [P],
                                      mask: &mut [u8], interp: Interp,
                                      border: BorderMode<P>) -> Result<(), Error>;
}
```
`polar` is the round-part unwrap, directly useful on canend. `from_fn` +
`metric::undistort_pixel` will give undistortion maps for free once B5 lands.

**Landed.** Every constructor is expressed over `from_fn` — `affine`/`projective` embed
their nalgebra transform, `polar` its bin-center `(angle, radius)` formula — so there is one
coordinate-precompute path and one hand-rolled bilinear gather (`apply`'s inner loop: a fast
all-taps-in-bounds path plus a border-fallback path, no per-pixel branch beyond that split).
Validity is first-class: `apply_with_mask` marks a destination pixel `255` iff every
interpolation tap it read (one tap for `Nearest`, four for `Bilinear`) fell inside the
source — derived for free from which of the two `apply` paths ran, not a second pass.
`dst`/`mask` are caller-owned flat buffers (`&mut [P]` / `&mut [u8]`), reused across frames —
`Map` itself allocates only once, at construction.

**Verified:** identity affine reproduces the source exactly (both interpolation modes); a
+0.5 px affine translation reproduces a linear ramp to 1e-4; affine ∘ affine⁻¹ recovers the
source to 1e-3 away from the clamped border; a projective embedding of an affine matrix
(bottom row `[0, 0, 1]`) matches the affine path bit-for-bit; `polar` followed by an inverse
built with `from_fn` recovers a synthetic two-axis (angle + radius) disc pattern to 0.79
intensity units (well inside the < 1.0 unit budget a 0.05 px position error implies for that
fixture's gradient); the mask marks exactly the out-of-source taps on a map built to leave
part of the destination outside the source. Bench (M4 Pro, VGA 640×480, bilinear, `Map`
built once outside the timed loop): `affine_apply_640x480_bilinear` ≈ 510 µs,
`polar_apply_640x480_bilinear` ≈ 494 µs — ~1.6-1.7 ns/destination-pixel, no per-apply
allocation. `match_shape` re-verified unchanged (this wave does not touch `matching`).

### B4.1 — rectify: canonical-pose crops (`matching` + `warp`) — `done`
```rust
pub struct CropSpec { pub rect: Rect2f /* model-frame coords */, pub px_per_unit: f32,
                       pub normalize_scale: bool /* default true */ }
impl ShapeMatch {
    pub fn model_frame_pose(&self, spec: &CropSpec) -> Similarity2f;
    pub fn model_frame_map(&self, spec: &CropSpec) -> Map;
}
```
`warp` moves pixels, `matching` says where the part is — this is the seam: a found
`ShapeMatch` rectified into a fixed-size, model-frame crop, the shape an anomaly model
needs identical across every frame. `spec.rect` is in the same reference-image frame
`ShapeMatch::pose` already consumes directly (no origin bookkeeping), so `output_size()`
depends only on the spec, never on which match produced the map. `normalize_scale = true`
(default) forces the crop to model scale by zeroing the found `Similarity2f`'s scale factor
while keeping its rotation and translation (`Similarity2f::from_isometry(pose.isometry,
1.0)`); `false` returns `pose` unchanged. `model_frame_pose` exists to hand back that exact
`dst -> scene` transform, so a detection made on the rectified crop maps back into the
scene without re-deriving it.

**Landed.** `crates/vision-metrology/src/matching/crop.rs`; `matching`'s Cargo feature now
implies `warp` (it was already a default-on transitive dependency; this makes the coupling
explicit rather than an accident of the default feature set). Recommended usage
(`Map::apply_with_mask` + `BorderMode::Constant`, not the crate's usual `Clamp` default) is
documented at the API — `Clamp` would fabricate texture by repeating the scene's edge
pixel, which an anomaly model trained on these crops would learn as normal.

**C1 headline number — rectify repeatability.** `tests/accuracy.rs`'s
`rectify_repeatability` row: one taught L-shape, re-rendered at 12 seeded subpixel poses
(±0.5 px translation, ±1° rotation, ±1% scale), found and rectified each time, per-pixel
σ across the 12 crops and mean `|crop − reference|` against a direct render of the taught
patch (valid-mask area only). **Measured (2026-08-20): bias 0.88, σ 1.69 (8-bit intensity
units, i.e. under 1% of full range)** — envelope pinned at 1.3 / 2.5. This is the number
that decides anomaly-pipeline viability: sub-pixel pose jitter contributes well under 1% of
dynamic range to the rectified crop, small enough that a downstream anomaly model's own
noise floor should dominate.

`examples/align_crops` runs teach → find → rectify on real glue-rig frames
(`data/42781`, one of the three vertically-stacked camera strips): 20/20 frames found,
mean crop validity 0.97.

### B5 — `metric`: the calibration bridge — `done`
Mirrors `PinholeIntrinsics`, `BrownConrady5`, `CameraModel`, `Pose3` (= `Isometry3f`,
camera-from-reference), `Plane3`, `PlaneGrid` on nalgebra 0.35. Alloc-free
`distort_pixel`/`undistort_pixel` (fixed 20-iteration Newton, converged < 1e-6
normalized), `pixel_to_ray`, `ray_plane_intersect`, `pixel_to_plane` (the exact
per-point path), `homography_plane_to_image` + `plane_grid_map` (the runtime
bird's-eye/rectify path over `warp::Map`, [`Plane3::xy`] only), `undistort_map`.
`metric::io` imports both calibration-rs's `RigExtrinsicsExport` (meters → mm on
import) and the `table_calibration` tool's `calibration.json` (already mm, by
inspection of the real fixture — documented, not asserted by the source format).
Offline/runtime split as recorded in system-design.

**Landed.** Golden-parity tests against a hand-crafted `RigExtrinsicsExport` fixture
(round-trips `distort_pixel`/`undistort_pixel` to < 1e-6 normalized over a grid,
`pixel_to_plane` by hand on axis-aligned cases) and a trimmed real
`table_calibration.json`. Python bindings: `CameraModel`/`PinholeIntrinsics`/
`BrownConrady5`/`Plane3`/`PlaneGrid` pyclasses, `Pose3` as a `(4, 4)` float64 array
(no dedicated class — the numpy-friendly choice), vectorized `pixel_to_plane` over
`(N, 2)` arrays (NaN row on a miss rather than a per-point exception), `plane_grid_map`/
`undistort_map` returning the existing `Map` pyclass, `load_rig_extrinsics`/
`load_table_calibration` (path or raw bytes). Lab: `POST/GET /api/calibration`
(format detected by shape), `POST /api/measure` augmented with `calibration_id`/
`camera_index`/`plane`, mm fields on fitted circle center/radius and hit caliper edges,
a px/mm toggle in the Measure tab.

**Rectify-first 3-D acceptance** (`tests/metric_rectify.rs`, plan decision 10): a
synthetic SDF L-shape target rendered on the reference frame's `z = 0` plane through a
tilted calibrated camera (pinhole + BC5, tilt 0/10/20/30/40° about the reference
x-axis) via the *forward* model (`pixel_to_plane` per raw pixel), then rectified back
to the plane grid with `plane_grid_map` + `apply_with_mask`. A `ShapeModel` taught on
the 0° rectified crop is found in every tilt's rectified crop:

| tilt | found | score | position error (px) |
|---|---|---|---|
| 0° | yes | 1.0000 | 0.0004 |
| 10° | yes | 1.0000 | 0.0037 |
| 20° | yes | 1.0000 | 0.0023 |
| 30° | yes | 1.0000 | 0.0120 |
| 40° | yes | 1.0000 | 0.0029 |

Found at every tilt, max position error **0.012 px** — an order of magnitude inside the
plan's 0.1 px target, on a noise-free synthetic fixture where render and rectify are
each other's exact geometric inverse. This is the number that says rectify-first closes
the planar 3-D case; it also bounds how little headroom homography refinement (next
session) has left to buy on a genuinely planar target.

**Track B accept:** `examples/inspect_canend` runs find → fixture → metrology model →
pass/fail on real canend data in pixels; the lab's `/api/measure` now closes the
millimetre step over an uploaded calibration.

---

## Track C — credibility and infrastructure — `planned`

### C1 — accuracy regression suite  ← the differentiator — `in progress — envelopes pinned, doc table pending`
Performance is measured and recorded; **accuracy is not**, and for a metrology library the
accuracy numbers *are* the product. Track A found three separate systematic biases that
every existing test passed straight through. Add `tests/accuracy.rs` and a
`docs/accuracy.md` table generated by an example, alongside the performance table:

| Operator | Sweep | Report |
|---|---|---|
| `Edge2DDetector`, `Edge1DDetector` | edge angle 0–90°, blur σ 0.5–3, noise 0–5 LSB | bias, σ (px) |
| `MeasureHandle::pos` | same, plus caliper width | bias, σ |
| `fit_circle` / `fit_ellipse` | point count, arc extent, noise, outlier fraction | radius bias, centre σ |
| `LaserExtractor` | stripe width, saturation, tilt | centre bias, σ |
| `ShapeMatcher` | sub-pixel translation and rotation sweep | pose bias, σ |

Gate each inside a recorded envelope. No open-source Rust CV crate publishes this.

**Landed so far:** `crates/vision-metrology/tests/accuracy.rs`, a data-driven table
(`ROWS: &[Row]`, one row per operator, `fn() -> Measured` + a pinned envelope — adding an
operator is one row) covering `Edge1DDetector`, `Edge2DDetector`, `Caliper` (rect), `fit_circle`
and `ShapeMatcher` (translation + rotation) over the swept axes above, each fixture an
antialiased Gaussian-CDF edge (or SDF+smoothstep L-shape for `ShapeMatcher`) with exact
subpixel ground truth. Envelopes were measured once and pinned at ~1.5x, except `fit_circle`'s:
its worst cell (30° arc + 10% outliers) is a genuinely near-degenerate circle fit — a nearly
straight chord where RANSAC's own consensus metric is ambiguous among near-collinear points,
not an implementation defect — so that envelope is intentionally looser. `Edge2DDetector`'s
sweep also found that `Hysteresis::Auto` is unusable at the heavy-blur/high-noise corner of the
grid: a weak true peak lets per-pixel noise dominate the frame's own max-response scaling and
hysteresis-chain across the whole image; the fixture instead characterises each blur level's
clean peak once and holds a fixed `Hysteresis::Manual` threshold, which is what a tuned real
system would do too. `LaserExtractor` and `fit_ellipse` are not yet covered, and
`docs/accuracy.md`'s generated table (an example, alongside the performance table) is not
started — both are the natural next session's work.

**Scale row (warp wave, decision 9).** Added `shape_matcher_scale_bias` /
`shape_matcher_scale_position`, sweeping 12 true scales geometrically spaced over 0.5–2.0×
at 3 rotations each, model taught at scale 1.0 with `scale_range = (0.45, 2.1)` — before any
matching-code change, per the plan's "measure first" instruction. **Measured result:** on
this clean, noise- and clutter-free synthetic L-shape, found-rate is **100% at every one of
the 12 scales** (36/36 finds), with scale bias ≤ 0.0014 (0.14%) and position bias ≤ 0.022 px
across the whole range — the plan's working hypothesis (real range ~[0.85, 1.2]) does not
hold here. This isolates one variable rather than contradicting the plan's field data
(canend, noisy and cluttered, and likely taught with the *default* `scale_range = (1, 1)`):
given a model whose own `scale_range` already matches what is searched, the discrete scan
itself is not the accuracy bottleneck on a clean scene. Envelopes pinned at ~1.5-2x the
measurement (scale bias 0.003, scale σ 0.001, position bias 0.04 px, position σ 0.01 px);
found-rate is pinned as a per-scale regression guard (`BASELINE_FOUND_RATE`) so a future
change that *shrinks* the found set — even one that stays inside these accuracy envelopes —
fails CI. Full per-scale table is in `tests/accuracy.rs`'s `BASELINE_FOUND_RATE` doc comment.

**Rectify row (rectify wave).** Added `rectify_repeatability` — see B4.1 above for the full
description. Measured bias 0.88 / σ 1.69 (8-bit intensity units, not pixels); envelope
pinned at 1.3 / 2.5.

### C2 — blob features
`ComponentStats` is bbox + centroid + count. Add second-order moments → orientation and
elongation, plus convex hull, min-area rect, circularity, rectangularity. Cheap on top of
the existing CCL and needed for blob-based inspection.

### C3 — bindings and CI
Python dtype dispatch **done** (Track D.1: `EdgeDetector`/`LsdDetector`/`ShapeModel`/
`ShapeMatcher`/`Caliper`/`MetrologyModel` all accept `uint8`/`uint16`/`float32` via one
`AnyImage`/`with_any_image!` dispatch helper in `vm-python/src/convert.rs`). Still open:
generate the vm-python config conversions instead of hand-mirroring them (now spread
across `src/config/*.rs`, ~750 lines but each file under the 600-line soft cap); a Python
binding for `laser` and for `segment::watershed`/region-growing (see `docs/backlog.md`);
`cargo publish --dry-run` in CI; miri over **all** unsafe, not just `laser::`.

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
