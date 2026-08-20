# System design

The architecture of `vision-metrology` as it **is**, and the record of why it is that way.
This file, together with [`roadmap.md`](roadmap.md) (where we are going) and
[`backlog.md`](backlog.md) (known debt), is the persistent context for anyone — human or
agent — starting a work session on this repository. Read all three before changing code.

Keep this file truthful: when a decision here is superseded, rewrite the entry (and say
what replaced it), don't append contradictions.

## Layering

Three publishable crates, strict one-way dependencies:

```
vm-primitives  ──►  vision-metrology  ──►  vm-python
(low-level)         (domain algorithms)    (PyO3 bindings)
```

| Crate | Modules | Contents |
|---|---|---|
| `vm-primitives` | `core` | Two private halves, one public path: `raster` (`Image<T>`/`ImageView<T>`, `Pixel` sealed over u8/u16/f32, sampling, `BorderMode`, `Error` — **no nalgebra**) and `geom` (nalgebra aliases `Point2f`/`Vec2f`/`Similarity2f`, `Vec2fExt`, transforms, `Circle2f`/`Ellipse2f`/`Conic2f`) |
| | `pyr` | `Pyramid`: 2×2 box-mean pyramid generic over `Pixel`, optional binomial pre-smooth, `level_to_base` |
| | `edge` | 1D/2D subpixel DoG edges, edgels, edge pairs, `DirectionField` |
| | `morph` | binary morphology (parameterized SE), chamfer distance, Zhang-Suen thinning |
| `vision-metrology` | `contour` | contour graph with T/Y junctions, per-edge geometry, polyline smoothing |
| | `laser` | stripe extraction via opposite-polarity edge pairs (rows/cols, ROI + prior) |
| | `matching` | `ShapeModel`/`ShapeMatcher`: gradient-orientation shape-based detection |
| | `segment` | Otsu/adaptive thresholding, CCL, watershed, edgel region growing |
| | `fit` | robust line / circle / ellipse fitting, residuals reported |
| | `measure` | calipers (rect / arc / radial), metrology model applied at a fixture pose |
| | `lsd` | LSD line-segment detection |
| | `warp` | `Map`: precomputed `dst → src` coordinate table (affine / projective / polar / `from_fn`), `apply`/`apply_with_mask` with a first-class validity mask |
| | `metric` | The calibration bridge: mirrored `CameraModel`/`Pose3`/`Plane3`/`PlaneGrid` on nalgebra 0.35, `pixel_to_plane` (exact), `plane_grid_map`/`undistort_map` (runtime `warp::Map` path), `io` importers for calibration-rs and `table_calibration` JSON |
| `vm-python` | — | numpy-in/numpy-out detectors; lib target named `vm_python` (see invariants) |

Each `vision-metrology` module is a default-on feature (see invariant 18). Names live at
their module path; `prelude` is a curated convenience and any crate-root re-export is an
explicit list, never a glob (invariant 17).

## Invariants

These are load-bearing. Breaking one is a design change, not a refactor, and needs a
matching update here.

1. **Pixel centers.** Integer coordinate `i` means position `i as f32`. Everything —
   edgels, ROIs, poses, pyramid coordinate mapping — assumes it.
2. **Pyramid coordinate mapping.** Level-`l` coordinate of a level-0 point:
   `L_l(p) = (p − (2^l − 1)/2) / 2^l`; candidate propagation `q_l = 2·q_{l+1} + 0.5`.
3. **Model/scene same-aliasing rule.** A shape model's level-`l` points come from running
   the edge detector on level `l` of the *reference ROI's own pyramid*, never from
   decimating level-0 points — model and scene must suffer identical box-downsample
   aliasing or coarse scores are systematically depressed. Corollary: the model build and
   the scene search must use the **same downsample kernel**.
4. **Score semantics.** The shape-matching score divides by the **full** model point count
   `n`, never by the contributing count. That is what makes `score ≈ 1 − occluded_fraction`
   and gives `min_score` its meaning. Spatially uniform point decimation (grid cells) is
   mandatory for the same reason.
5. **Rust-native.** No OpenCV, no FFI in the library crates.
6. **Hot paths are allocation-free per scan/row.** Detectors/extractors own reusable
   scratch; the only per-call allocation allowed is the output container.
7. **`unsafe` policy.** Only small, justified blocks with `// SAFETY:` comments; guarding
   `assert!`s and the block they protect move together in any refactor.
   `unsafe_op_in_unsafe_fn` is denied workspace-wide.
8. **Error type.** `vm_primitives::Error` everywhere; `&'static str` payloads only.
9. **`'static` public outputs.** No lifetimes in public result types (PyO3 compatibility).
10. **Config-struct + reusable-detector API pattern, and no sentinel values.** A config
    is a plain `pub` struct with `Default`, constructed with `..Default::default()`; a
    detector owns its scratch and is reused across calls. "Absent", "automatic" and
    "unlimited" are spelled in the type — `Option<T>`, `Option<NonZeroUsize>`, or a named
    enum — never as `0`, `0.0`, or an empty range. A magic value is indistinguishable from
    a legitimate one (`low_thresh = 0.0` meant *both* "no threshold" and "choose one for
    me"), does not survive a language boundary (`None` is what a Python caller writes),
    and cannot be checked by the compiler.
    A config that grows past ~8 fields splits: the fields that describe **what is being
    looked for** stay at the top level; the ones that describe **how hard the search
    works** move into a nested `tuning: XTuning` with its own `Default`. Numeric thresholds
    that depend on the pixel type carry a unit type (see `Contrast`) rather than a
    comment saying which pixel type they were tuned for.
11. **Default border mode is `Clamp`** in core/edge unless configured otherwise.
12. **Determinism.** No RNG in library code; tests use synthetic fixtures (seeded if
    randomness is unavoidable); f32 sort ties broken explicitly (`(−score, x, y)`).
13. **MSRV 1.91**, edition 2024, nalgebra 0.35 as workspace dependency. Do not
    re-implement linear algebra.
14. **File-size policy.** Soft cap ~600 lines per source file (tests excluded from the
    count). Crossing it is a signal to split as part of the same change, not later.
    Current known offenders are tracked in `backlog.md`.
15. **vm-python parity.** A PR that adds public Rust API updates the bindings and a Python
    test in the same PR.
16. **Docs-as-memory.** A PR that changes scope, decisions, or invariants updates
    `system-design.md` / `roadmap.md` / `backlog.md` in the same PR.
17. **One canonical path per name.** No glob re-exports across crate boundaries. A name
    lives in its module; `prelude` is a curated convenience, and any crate-root re-export
    is an explicit list. `vision-metrology` re-exports the `vm_primitives` *crate*, not its
    contents.
18. **Every domain module is feature-gated**, default-on. A new module ships with its
    feature, its `required-features` on any example/bench/test that uses it, and a row in
    the crate-doc feature table. CI checks each feature alone, not just `--all-features`.
19. **One entry point per algorithm**, generic over `Pixel`. No `_u8`/`_u16`/`_f32`
    variants of the same operation; a `_f32` suffix is only for something that genuinely
    only takes `f32` (a pyramid level).
20. **Storage f32, accumulation f64.** Pixel coordinates are stored as `f32`; every
    normal-equation, moment, or residual sum runs in `f64`. Squared coordinates near
    2000 px exhaust an `f32` mantissa long before a fit converges.
21. **Every measurement reports its residual.** A fitting or measuring operation returns
    the deviation statistics that qualify it (`rms`, `max_dev`, how many points were used),
    not just the parameters. Without them the caller cannot tell a measurement from a
    number, and cannot gate on form tolerance at all.

## Decisions and why

### Chamfer matcher → gradient-orientation shape matching (PR #18, 2026-08)
The distance-only chamfer matcher was replaced wholesale (deleted, not deprecated: crate
was 0.1.0 and unpublished). Its metric ignored gradient orientation, it had no pyramid, no
greedy abort, and its angle range could not cross ±π. The replacement implements Steger's
DAGM 2001 similarity measure (the algorithm behind HALCON `create_shape_model` /
`find_shape_model`): score `S = (1/n) Σ (R·tᵢ)·ĝ(pᵢ)` with three polarity modes, greedy
early termination factored to one FMA + compare per 8-point chunk, coarse-to-fine pyramid
search with 3-D (x, y, α) local maxima, and correspondence-free least-squares pose
refinement (f64 normal equations, 4→3→2 DOF Cholesky fallback for symmetric parts).
`morph::chamfer_distance_u8` survives as an independent primitive.

### `DirectionField` lives in vm-primitives, not in matching
The matcher needs a dense gated unit-gradient field per pyramid level, stored across
levels. `GradientBuffers` borrows `&mut Edge2DDetector` and cannot be held per level; the
full edge pipeline (NMS + hysteresis + edgel build) is ~2× the necessary work. The field is
a pure image primitive with no matching semantics, hence vm-primitives.

### Model `min_contrast` is the knob that decides real-world performance
On low-relief parts (canend dataset), edge-detector auto-thresholds admit model points on
faint non-repeating surface shading. Because of invariant 4 those points *dilute* the
score rather than merely not helping. Raising `ShapeModelConfig::min_contrast` took
set1/dome from 0.785 → 0.998 median score and CP34 from 35/39 found to 39/39. Documented
with a tuning table in `shape-matching.md`.

### Cross-repo dependencies are crates.io releases only
The maintainer owns `corrmatch`, `box-image-pyramid`, and `calibration-rs`. When a feature
is missing there, it is implemented upstream, released, and bumped here — never a git or
path dependency in committed code. Keeps this workspace publishable and CI reproducible.

### MSRV 1.89 → 1.91 (2026-08)
nalgebra 0.35 needs only 1.89, but `corrmatch` and `box-image-pyramid` (planned
dev-dependencies for pose auditing and the u8 scene-path pyramid) declare 1.91. The crates
here were unpublished at the time, so no compatibility promise was broken. The MSRV CI job
builds `--all-targets`, which is why even dev-dependencies constrain the floor.

### vision-calibration: offline/runtime split
`calibration-rs` has the laser-plane pipeline we need (`LaserPlane`,
`LaserlinePlaneSolver`, `pixel_to_gripper_point`), but it is structurally pinned to
nalgebra **0.34** (tiny-solver/faer chain; nalgebra types cross its public API) and MSRV
1.93. Decision: vision-calibration remains the **offline** calibration system; a small
runtime `metric` module here mirrors only the parameter types (intrinsics, distortion,
plane) on nalgebra 0.35 and loads calibration-rs JSON exports, pinned by a golden-file
test. A direct dependency replaces the mirror when upstream reaches nalgebra 0.35.

### `metric`: the calibration bridge — units, pose direction, and the homography split (2026-08)
B5 (roadmap). Four decisions that have to be right before any of the arithmetic matters,
because getting any one backwards is a silent, plausible-looking bug (right shape, wrong
numbers) rather than a compile error or a crash.

**Units: millimetres, not the wire format's meters.** Every 3-D/plane quantity in
`metric` (`Pose3` translations, `Plane3::d`, `PlaneGrid`, `pixel_to_plane`'s output) is
millimetres; pixel-domain quantities (`PinholeIntrinsics`, pixel coordinates) stay in
pixels. `io::import_rig_extrinsics` converts calibration-rs's meters on import (the wire
schema documents the unit explicitly); `io::import_table_calibration` needed a judgment
call instead — the format documents no unit at all, and the real fixture's
camera-to-camera translation is ~111 in magnitude, which is a plausible two-camera
benchtop baseline in mm and an absurd 111-**meter** rig if read as meters. Recorded at
the importer, not asserted as a property of the format.

**`Pose3` is camera-from-reference (`cam_se3_rig`, `T_C_R`), matching calibration-rs's
own convention rather than inventing a new one.** `pose * p_ref` gives `p_ref` in the
camera's own frame. Every function here and every Python binding follows this
direction; getting it backwards would silently mirror every downstream projection
through the camera center, the same failure shape `warp`'s own `dst → src` direction
trap describes. `io::import_table_calibration` treats `camera0`'s `sensor2camera`
(the identity in every observed export) as defining the reference frame, so every other
camera's `sensor2camera` **is** `Pose3` directly with no re-composition — a fact about
this specific dataset (camera0 is always identity), not a general property of the file
format, and documented as such at the importer.

**A homography cannot carry Brown-Conrady distortion — so there are two, deliberately
different, plane-to-image paths, not one.** [`pixel_to_plane`] is the exact path:
per-pixel `undistort_pixel` → `pixel_to_ray` → `ray_plane_intersect`, correct for any
`Plane3`, the one to use for a point-wise measurement. [`homography_plane_to_image`] /
[`plane_grid_map`] are the whole-image runtime path — a homography (undistorted only)
composed with forward `distort_pixel` per destination pixel via `warp::Map::from_fn` —
restricted to `PlaneGrid`, which is always the reference frame's own `Plane3::xy`
(`z = 0`), *not* an arbitrary plane, because the homography's linearity depends on
substituting `z = 0` into the pose's affine map. A caller who needs a tilted or offset
plane's bird's-eye view has no `plane_grid_map` shortcut; that is intentional; and the
module docs say so at both entry points rather than leaving a caller to discover the
restriction from a wrong answer.

**`pixel_to_plane`'s in-plane basis is deterministic, not caller-supplied.** For a
general `Plane3`, the 2-D coordinate system needs an origin and two axes; the origin is
the plane's closest point to the reference-frame origin (`p0 = -d·n`), and the axes are
built by projecting the reference frame's own `x` axis onto the plane (or `y`, if `x` is
within 0.9 of parallel to the normal) and completing a right-handed pair with a cross
product. Purely a function of `plane.n`, so two calls with the same plane always agree
— important since `PlaneGrid`'s own `z = 0` case sidesteps this entirely (its axes are
the reference frame's `x`/`y` directly), so the two paths only need to agree in the
`Plane3::xy` case, which they do by construction (`plane_basis` reduces to the identity
basis when `n = (0, 0, 1)`).

**Rectify-first 3-D acceptance, measured rather than assumed** (`tests/metric_rectify.rs`):
see roadmap B5 for the full per-tilt table. Found at every tilt 0–40°, max position
error 0.012 px — an order of magnitude inside the 0.1 px target, and small enough that
homography refinement (next session) has very little left to buy on a genuinely planar
target; its value is for the *non*-planar case this synthetic fixture cannot exercise.

### Shape-matching preprocessing dominates the search
Measured (M4 Pro, 1280×1024, full 360°): `find_u8` ≈ 7.8 ms = ~5.3 ms direction-field
pyramid + ~2.5 ms search. Below the top pyramid level the search reads only small windows
around candidates, so full-frame fine-level fields are mostly wasted work. The performance
plan (roadmap Track 2) is lazy tiled fields first, integer u8 Scharr second, quantized
directions + SIMD only if still needed — in that order, each gated on measurement.

### The chain runs end to end on real data (2026-08)
`examples/inspect_canend` closes the loop the library exists for, on the can-end frames
rather than a synthetic fixture: find the tab → take its pose as a fixture → measure the can
rim **in the tab's frame** → report the fit and its residuals → pass/fail on `max_dev`.

Teaching the rim relative to the tab is what makes the numbers mean something: every frame
re-derives the rim from wherever the tab turned up, so the spread across frames measures the
fixture *and* the measurement, not where the part happened to sit.

set1, 96 calipers, Tukey(2 px), tolerance 2 px on `max_dev`:

| folder | frames | measured | mean radius | σ | per-frame rms |
|---|---|---|---|---|---|
| normal/dome | 50 | 50/50 | 365.24 px | 0.28 px | 0.20–0.54 px |
| normal/dark | 50 | 50/50 | 365.70 px | 0.31 px | 0.28–0.60 px |

All 96 calipers survived the robust fit in every frame. σ ≈ 0.3 px is an upper bound on
repeatability — it is measured over *different physical cans*, so real part-to-part rim
variation is inside it. For comparison, the Track 3 rim-relative σ was 0.8–3.3 px; that
statistic measured the tab's position relative to the rim, which compounds two locations,
where this measures the rim directly.

Units are pixels. Millimetres arrive with `metric` (B5).

### `measure`: calipers, and why a curved edge needs its own placement (2026-08)
The module that turns detection into inspection. [`Caliper`] places a geometry, averages
intensity across it into a 1-D profile, and runs the existing `Edge1DDetector` along that
profile; [`MetrologyModel`] distributes calipers over nominal primitives held in the part's
own frame, applies them at a fixture pose (`ShapeMatch::pose`), and fits the measured points
with the `fit` module. `find → pose → apply → Fit + residuals` is now a closed loop.

**`MeasureRadial` exists because a rectangle cannot measure a circle without bias.** A rect
averages along a straight *chord*: on a circle of radius 40, a sample 5 px to the side sits
at radius 40.31 — the wrong side of the edge — so the averaged profile is contaminated and
the measured radius reads low. Measured on an anti-aliased disc, a 32-caliper circle fit
came out **39.88 px instead of 40.00**. `MeasureRadial` scans radially and averages *along
the arc*, so every averaged sample sits at the same radius: **39.990 px**, a twelvefold
reduction, and the bias no longer grows with caliper width. `MetrologyShape::Circle` uses it.

This also forced the test fixtures to change. A hard-thresholded disc puts its edge wherever
the pixel grid falls — 0 to 0.5 px inside the nominal radius, depending on the centre's
sub-pixel offset — so it cannot be used to assert subpixel accuracy at all. The fixtures are
anti-aliased now (coverage ramps across one pixel), which puts the true edge exactly at the
nominal radius and let the tolerances drop from 0.3 px to 0.03 px.

**Ideas taken from the `rtvt-pano` caliper** (`crates/rtvt-glue/src/caliper.rs`), which
solves the same problem for glue-bead cross-sections:
- **Typed rejection reasons.** A caliper that finds nothing is a *result*, and which gate
  rejected it is the difference between "the part is missing" and "the search window is too
  short". `Caliper::measure` returns `Err(RejectReason)` — and there is no variant that
  discards it (see below).
- **An obliquity gate.** A caliper crossing an edge at a glancing angle reports a position
  along its own axis rather than the edge normal, and the two differ by `1/cos θ`; at a
  corner there is no meaningful crossing at all. `max_obliquity_deg` compares the local image
  gradient against the scan direction and rejects the rest.
- **Sub-pixel profile stepping** (`step`), for oversampling a sharp edge.

Not taken: the two-pass centreline refinement and the mask-based background-padding gate.
Both are properties of a *tracked contour* rather than of a caliper, and belong in whatever
bead/stripe tool is built on top of this one.

### `fit`: geometric refinement, and what robustness actually requires (2026-08)
`shape` held algebraic conic fitting and a RANSAC ellipse wrapper; there was **no line fit
and no circle fit at all** — the two most common metrology measurements. The new `fit`
module supplies all three, and `shape` narrows to LSD (renamed `lsd` in the v0.3 reset). `Circle2f`, `Ellipse2f` and `Conic2f`
move down to `vm-primitives::core` (types belong below the algorithms that produce them);
`Circle2f` is new.

Every fitter is algebraic-init → geometric refine, and returns `Fit<M>` with `rms`,
`max_dev` and `n_used` (invariant 21). `fit_circle` uses Taubin, which is near-unbiased on
short arcs where the Kåsa fit collapses toward the chord, then Gauss–Newton on the true
residual `‖p − c‖ − r`: on a 30° arc that is the difference between visible bias and
< 0.05 px.

**Two things had to be discovered by testing rather than designed.**

*Tukey cannot start from a contaminated fit.* The rejection radius is applied to the
algebraic initialisation, which was computed from all the points including the outliers — so
a small radius throws away the **inliers** and keeps whatever the corrupted fit passed
through. Measured: one outlier 70 px off a 40-point circle left a Tukey fit standing on
2 points. Fixed with graduated non-convexity — `RobustLoss::annealed` starts the radius wide
enough to admit every point and shrinks it geometrically to the configured value.

*Reweighting cannot fix a flipped axis.* 30 points on `y = 10` plus one at `(15, 60)` gives
the outlier enough y-variance that total least squares returns a near-**vertical** line. The
starting guess is not inaccurate, it is orthogonal, and no IRLS scheme recovers. That is what
RANSAC is for — and `fit_line` was silently ignoring `FitConfig::ransac`, exactly the class
of dead config field this reset removed from LSD. It is honoured now, with 2-point
hypotheses.

Bench (M4 Pro): `fit_circle_500pts` 2.6 µs, `+tukey` 4.2 µs, `fit_line_500pts` 1.7 µs,
`fit_ellipse_100pts` 1.6 µs, `fit_ellipse_ransac_1000pts` 430 µs.

`examples/measure_circles` now fits circles rather than ellipses and reports `rms`/`max_dev`
per measurement; it also gates on `rms` — a filter only possible because the fit reports one.

### Module hygiene: preludes, explicit re-exports, feature gates (2026-08)
`vision-metrology` re-exported `vm_primitives::*` with a glob. That gave every name two
paths, made every addition upstream a potential collision downstream, and hid what this
crate's surface actually was. Replaced by three things: `pub use vm_primitives;` (the crate
itself, so one dependency is still enough), an explicit curated re-export list, and a
`prelude` on both crates whose contents follow the feature gates.

Every domain module is now an opt-in feature, default-on: `contour`, `laser`, `matching`,
`segment` (implies `contour` — region growing consumes a `ContourGraph`), `lsd`, and
`serde` (implies `matching`). Examples, benches and integration tests carry
`required-features`, so `--no-default-features` skips them instead of failing.

The gates are only worth having if they are checked. `--all-features` can never catch a
`#[cfg]` that forgot a gated import, so CI gained a `cargo hack --each-feature` job over
`vision-metrology` and a `--feature-powerset` over `vm-primitives`.

`Error` is `#[non_exhaustive]`, so adding a variant stops being a breaking change.

### `shape` → `lsd`, and the tiling protocol becomes a session (2026-08)
Two renames of different weight.

`shape` stopped being a true name when conic fitting and the RANSAC ellipse wrapper moved
into `fit`: what was left was one algorithm. The module and its feature are now `lsd`, and
`shape/lsd.rs` (the `lsd::lsd` path) is `lsd/detect.rs`.

`DirectionField`'s lazy tiling was a three-step stateful protocol on one type —
`begin_tiled_f32(img)`, then `ensure_rect_f32(img, …)` repeatedly, then read — where the
ordering was documentation, the image was passed twice and checked with a runtime
`assert!`, and reading a field that was never put into tiled mode was a silently-all-zero
result rather than a compile error. `begin_tiled_f32` now returns a
[`TiledField<'a>`] session borrowing **both** the field and the image; `ensure_rect` lives
only on the session and takes no image, so it cannot be given the wrong one; the session
derefs to the field, so scoring code is unchanged; and `#[must_use]` catches the caller
who enters tiled mode and drops the session.

The matcher builds one session per lazily-tiled level up front (`split_at_mut` separates
the fully-built top level from the tiled ones), which cost one small `Vec` per `find` and
removed the per-`ensure_rect` dimension assert. Measured on `match_shape`, M4 Pro:

| Bench | before | after |
|---|---|---|
| `shape_find_1280x1024_360deg` | 3.408 ms | 3.370 ms |
| `..._360deg_clutter` | 6.542 ms | 6.471 ms |
| `..._tracked_roi` | 1.452 ms | 1.447 ms |
| `..._360deg_greedy0` | 5.286 ms | 5.247 ms |
| `..._scale_0p8_1p25` | 17.57 ms | 16.83 ms |

Every case held or improved, so the safety came free.

### `core` splits into `raster` and `geom`, and the raster half names no nalgebra (2026-08)
`core` was seven flat files in which the dependency on nalgebra was invisible. It is now
two private submodules with one rule between them: **`core::raster` mentions no
linear-algebra type at all.** Buffers, pixel types, borders and sampling on one side;
aliases, transforms and shapes on the other.

The rule is not aesthetic. The ecosystem around this workspace already carries five
near-duplicate `ImageView` types, and `rtvt-image` is a knowing fork of this very crate
that exists *only* because it is pinned to nalgebra 0.34 while this one is on 0.35. An
image buffer has nothing to do with linear algebra; the reason it could not be shared was
that the two were in the same module. A raster layer that names no nalgebra type is the
piece that can cross a major-version boundary, and this split is what makes extracting it
later a move rather than a rewrite.

The audit came out clean: `sample_bilinear_f32` and friends already took bare `f32`
coordinates, so nothing had to change to satisfy the rule. `core::sample_bilinear_at`
(taking a `Point2f`) is the geometry-side convenience that keeps the raster signature
that way rather than "improving" it later.

Both submodules stay **private** — every name keeps its single canonical `core::…` path
(invariant 17). This is a rule about dependencies, not about import paths, and no public
path changed.

### One model-format bump, batched — opacity, read-only levels, R3 (2026-08)
A stored `ShapeModel` is the one artefact of this crate that outlives a build, so every
change to it invalidates files on disk. The v0.3 reset therefore did all of them at once:
`FORMAT_VERSION` 2 → **3**, and no more bumps in this series.

**The format is opaque.** `to_json` / `from_json` and the public
`SHAPE_MODEL_FORMAT_VERSION` are replaced by `save` / `load` and `to_bytes` /
`from_bytes`. The encoding is JSON today and that is not a promise: a documented wire
format makes every internal field of the model a compatibility obligation, and the version
constant existed only to let callers hand-assemble an envelope this crate should be the
only writer of. Nothing a caller could do with the number is not better answered by `load`
returning an error, so the constant is `pub(crate)` and the Rust and Python tests now
assert the property that matters — a foreign document is refused, not mis-read.

**`ModelPoint` and `ShapeModelLevel` are readable, not writable.** Their fields are
`pub(crate)` with public accessors. The lab and the overlay code draw model points, so
reading them has to stay easy; *writing* them does not, because the point order is
load-bearing (greedy termination evaluates a prefix, which must sample the whole contour)
and `radius` / `angle_step` / `scale_step` are derived quantities the search trusts.

**R3 is wired (backlog item closed).** `ShapeModelConfig::pre_smooth` chooses the pyramid
pre-filter, the built model stores it, and `ShapeMatcher::find` reads it off the *model*
rather than from its own config. That is invariant 3 made unbreakable: a model taught with
`Binomial121` cannot be searched against a box-mean scene, whatever the caller's search
config says. The default stays `PreSmooth::None`, so nothing about existing behaviour
moves; `Binomial121` is now available for the fine-toothed contours that alias away at
levels 3–4.

### A measurement that found nothing is a result, not an empty slice (2026-08)
`Caliper` had `measure` returning `&[MeasureEdge]` and `measure_checked` returning
`Result<&[MeasureEdge], RejectReason>` — the same computation, one of them throwing away
the diagnosis. `MetrologyModel` had the same pair, with the lossy `apply` additionally
*renumbering*: objects whose fit failed were skipped, so `results[i]` was not object `i`
and the caller could not tell which one was missing.

Both lossy twins are deleted. `Caliper::measure` returns the `Result`; `Ok(&[])` is
unrepresentable because an extraction that found nothing always has a `RejectReason`
(`NoEdge`, `TooOblique`, `WrongPolarity`, `OffImage`, `ProfileTooShort`). `MetrologyModel::apply`
returns `Vec<Result<MetrologyResult, Error>>`, one entry per object in `objects()` order.

`MetrologyModel::hits()` — a parallel array of caliper edges from the last call, which the
caller had to keep aligned by hand and which a second `apply` silently invalidated — folded
into the result: `MetrologyResult { fit: MetrologyFit, hits: Vec<MeasureEdge> }`. The
`Line`/`Circle` enum is now `MetrologyFit`, and `rms()`/`max_dev()`/`n_used()` are on both.

This is the general rule the reset applies to diagnostics: cheap borrowing accessors
(`Caliper::profile()`, `ShapeMatcher::truncated()`) stay, diagnostic *computation* lives in
a `diagnostics` module off the hot path, and anything a result was already computing
travels with that result instead of in a side channel.

### Configs say what they mean: the split, the sentinels, and `Contrast` (2026-08)
Three problems in one pass, all of them in the *type* rather than the algorithm.

**The split.** `ShapeSearchConfig` had 14 flat fields, six of which (`max_candidates`,
`coarse_score_factor`, `greediness`, the two step overrides, `last_level`) are search
*effort* — they trade run time against the chance of missing a match `min_score` says
should be reported. Presenting them next to `min_score` invites tuning by field name.
They now live in `tuning: ShapeSearchTuning`, which has its own `Default`, so the common
case is unchanged and the advanced case is one word longer. `LaserExtractConfig` split the
same way: `axis` / `min_width` / `max_width` / `min_score` say what a stripe *is*;
`tuning` holds the coarse method, ROI half-width, jump and gap limits, prior weight, edge
config and smoothing. No builders — the plain-struct-plus-`..Default::default()` form is
what makes serde and the Python mirror cheap.

**The sentinels.** `0`/`0.0` meaning "auto" is gone (invariant 10 rewritten):
`num_levels`, `max_points`, `max_matches` are `Option<NonZeroUsize>`; `angle_step`,
`scale_step` are `Option<f32>`; `Edge2DConfig`'s two threshold fields became one
`Hysteresis::{Auto, Manual{low, high}}`, which also fixes the case where a caller set one
of the pair and silently got neither meaning. `NonZeroUsize::new(512)` *is* an
`Option<NonZeroUsize>`, so struct literals stayed readable.

`Edge2DConfig::pre_smooth: bool` alongside `smooth_kind: SmoothKind` was two encodings of
one decision, with `pre_smooth: false, smooth_kind: Binomial3` representable and
meaningless. `SmoothKind::None` already said it; the bool is gone.

`LaserExtractConfig::enable_smoothing: bool` turned out to be real — a median-of-5 over
each contiguous run of valid samples — with the window hard-coded where no caller could
see it. It is now `CenterSmoothing::{None, Median { half_window }}`, and the filter is
parameterised by it.

**`Contrast`.** `min_contrast` was documented as "Scharr response units on the input pixel
scale", which is to say: a number whose meaning changes by 257× between `u8` and `u16`
data of identical physical contrast. `Contrast::Raw(f32)` keeps exactly that behaviour and
is the default; `Contrast::FractionOfRange(f)` resolves to `f · 16 · (max − min)` of the
image being processed — 16 being Scharr's response to an ideal unit step — and therefore
transfers between pixel types unchanged. The model resolves it against the reference
*ROI*, not the frame; the search resolves it against the scene; `ShapeModel::from_edgels`
has no image and returns `InvalidConfig` rather than inventing a range from the very
strengths it is about to filter. `Raw` costs nothing, so the min/max pass only happens
when a fraction was actually asked for.

### The v0.3 visibility sweep: one path per name, and nothing else public (2026-08)
Invariant 17 said "one canonical path per name"; the crate root said otherwise. A flat
`pub use contour::{…}` / `laser::{…}` / `matching::{…}` block re-exported the whole domain
surface, so every type had two paths (`vision_metrology::ShapeMatcher` and
`vision_metrology::matching::ShapeMatcher`) and the crate root read as the API rather than
the modules. The block is gone. What remains at the root is the curated `vm_primitives`
list (the names a caller of *this* crate types constantly), the `vm_primitives` crate
itself, and `prelude` — now covering `fit`, `measure` and `segment` too, which it had
silently skipped.

Removed from the public surface in the same pass, because pre-release is when this is free:

- `laser::coarse_center_{u8,u16,f32}` and `laser::best_pair_with_prior` — pipeline stages,
  not API, and the typed triplet violated invariant 19 outright. Deleted rather than
  hidden: the generic `coarse_center_in_range` they wrapped is the real implementation and
  had no other caller.
- `contour::MAX_KERNEL_PTS` — an implementation detail of the smoothing scratch buffer.
- `pyr::downsample2x2_mean{,_into,_to_f32_into}` — `Pyramid` is the entry point. The two
  same-type variants had no caller anywhere and were deleted; the `f32` kernel is
  `pub(crate)`, with a `#[doc(hidden)]` benchmark hook so `benches/downsample.rs` can still
  measure per-pixel-type kernel throughput without the level-0 widening pass.
- `matching::create_shape_model` — a one-line duplicate of `ShapeModelBuilder::build`.
- `core::transform_point_iso` — `iso * p` with nalgebra in the public API already.
- `edge`'s submodules (`edge::edge2d::Edgel` → `edge::Edgel`). Splitting a module across
  files is a file-size decision (invariant 14); it should not show up in import paths.
- `matching::match_point_scores` moved to `matching::diagnostics::match_point_scores`.
  Diagnostics are a module, not a feature, and they do not belong at the root of the
  algorithm they instrument.

### `Point2f` / `Vec2f` are nalgebra aliases (2026-08)
```rust
pub type Point2f = nalgebra::Point2<f32>;
pub type Vec2f   = nalgebra::Vector2<f32>;
```
The crate previously carried its own two-field structs plus seven public functions to convert
to and from nalgebra — while invariant 13 says not to re-implement linear algebra, and
`Similarity2f` already put nalgebra in the public API. Aliasing removes the parallel type
system: 7 conversion functions and ~250 lines of hand-written operators are gone, and points
now cross into `calibration-rs` / `corrmatch` / `chess-corners-rs` untouched.

Costs, accepted knowingly:
- 147 struct literals became `Point2f::new(x, y)`. Mechanical.
- `dot` takes a reference: `a.dot(&b)`.
- `perp`, `cross` and `normalized_or_zero` moved to the `Vec2fExt` trait, which must be
  imported.
- The model wire format changed — nalgebra serializes a vector as a flat `[x, y]` rather than
  `{"x":…,"y":…}` — so `SHAPE_MODEL_FORMAT_VERSION` is **2** and version-1 documents are
  rejected.
- `shape_model_create_1280x1024` 490 → 529 µs (+8%). Model building is a one-time cost;
  `find`, the per-frame path, is unchanged (3.36 ms) and clutter moved +0.5%.

**The trap this introduced, and the guard against it.** nalgebra's `normalize` divides
unconditionally, so a zero vector yields `NaN` where the old hand-written version returned
zero. Four call sites normalize a possibly-degenerate gradient or tangent and then reject it
with `t.norm() < 0.5` — and `NaN < 0.5` is **false**, so those guards would have silently
stopped rejecting and let `NaN` directions into the model and the score.
`Vec2fExt::normalized_or_zero` restores the old semantics and is what every such site now
calls; the distinction is documented on the trait method and pinned by a test.

Canend set1/dome after the change: 50/50, shape p50 0.998, ZNCC p50 0.961 — identical to
four decimal places.

### One generic entry point per algorithm (2026-08)
Every detector used to expose `_u8` / `_u16` / `_f32` variants of the same method. Counting
the affected functions across both library crates: **40 entry points became 16**, and the
three that remain `_f32` (`build_image_f32`, `begin_tiled_f32`, and — until the v0.3 reset
turned it into `TiledField::ensure_rect` — `ensure_rect_f32`) genuinely
only ever see pyramid levels, which are always `f32`.

`vm_primitives::Pixel` is a sealed trait over `u8`/`u16`/`f32` carrying `to_f32`,
`from_f32_sat`, an `Acc` accumulator, and `as_f32_slice` (the zero-copy hook that keeps an
`f32` laser scan from being widened into scratch it does not need). Sealed, so adding a pixel
type stays non-breaking downstream.

`laser`'s private `ScanPixel` trait is gone — `Pixel` covers everything it did except the
per-type gather scratch, which is now a `Vec<P>` local to the column scan: one allocation per
call, next to the output vector, rather than three buffers on the extractor.

**A measurement that overrode the tidier design.** Gathering columns directly as `f32` looks
like it saves a widening pass. It does not: it quadruples the gather's write traffic, while
the widening it avoids happens only over the much smaller ROI segment.
`laser_extract_cols_gather_512x1280` went 151 → 183 µs (+21%) before this was caught by an
A/B against a baseline measured in the same session. Gathering the pixel type as-is is back,
and the reasoning is recorded in `laser/gather.rs` so it is not "simplified" again.

Everything else held or improved: `shape_find_1280x1024_360deg` 3.42 → 3.36 ms,
`..._clutter` 6.61 → 6.42 ms, `laser_extract_rows` +0.3%, `laser_extract_cols` +0.3%,
`shape_model_create` +0.6%.

### LSD downsamples through `pyr`, and its config stopped lying (2026-08)
`lsd/detect.rs` (then `shape/lsd.rs`) carried its own 2×2 downsample under a comment claiming it was "the same as a
`vm_primitives::pyr` level". It was not: `div_ceil` + border clamp against `pyr`'s drop-odd,
so on odd input the two disagreed on both output size and edge values. It then mapped
positions back with a bare `p · (W / w)`, missing the `(2^l − 1)/2` term of invariant 2.

Measured on a vertical step edge whose true subpixel position is 59.5, mean reported endpoint
x, at the **default** config:

| width | before | after |
|---|---|---|
| 128 (even) | −0.500 px | 0.000 px |
| 129 (odd) | −0.954 px | 0.000 px |

Two config fields were also fiction. `scale: f32` was documented as a downscale factor and
defaulted to `0.8`, but the code only tested `scale < 1.0` and then always halved — so the
default meant 0.5, not 0.8. `sigma_scale: f32` was documented as a Gaussian pre-smooth,
defaulted to `0.6`, and was **never read**. Both are replaced by
`downscale_levels: u32` (0 = full resolution) and `pre_smooth: PreSmooth`, which say what
they do and are honoured. `pre_smooth` defaults to `Binomial121`, delivering the
anti-aliasing `sigma_scale` had only promised.

`detect_u8` / `detect_f32` collapse into one generic `detect<P: Pixel>`. The detector now owns
a `Pyramid`, so there is one downsample kernel in the workspace instead of two.
`lsd_detect_u8_1280x1024` 1.49 ms (was 1.49 ms — the pyramid replaces an equivalent copy).
Pinned by `endpoints_are_unbiased_at_every_downscale_level`, which sweeps levels 0–2 × both
pre-smooth modes × even/odd widths.

### `multiscale` deleted, not fixed (2026-08)
`MultiScaleEdgeDetector` ran the 2-D detector on every pyramid level and merged the results
back to level 0. Three things were wrong with it and only the first was a bug:

1. It mapped a level-`l` edgel to level 0 as `p · 2^l`, omitting the `(2^l − 1)/2` term that
   invariant 2 requires. Level-2 positions were off by 1.5 px, level-3 by 3.5 px. Mixed with
   correct level-0 edgels this produced a *systematic* bias, not noise: `examples/measure_circles`
   measured every circle centre 0.07–0.10 px low in both axes, growing with radius as the
   coarse levels contributed more. With the module removed the same example measures 0.00 px.
   Its assertions (5 px on centre, 1.5 px on radius) were far too loose to notice.
2. Deduplication keyed on `idx * 2^l`, so a level-3 edgel claimed a single level-0 pixel
   rather than the 8×8 block it stood for, and `merge_duplicates` barely merged. The test
   only asserted `merged <= all`.
3. The reported `scale = base_sigma · 2^l` was fiction. A box-mean pyramid with a fixed-σ DoG
   at each level is not a Gaussian scale space, so the number had no operational meaning.

Fixing (1) alone would have left a module that is neither a scale space nor a sound merge,
with no consumer inside the workspace. The one sound idea — the level↔level-0 mapping — is
now `pyr::level_to_base` / `base_to_level`, the single implementation of invariant 2. If
genuine scale selection is ever needed, design it as a real scale space rather than reviving
this. `examples/measure_circles` and its Python twin now use `Edge2DDetector` directly, with
assertions tightened to 0.02 px on centre and 0.10 px on radius.

### Release profile is tuned, and benches inherit it (2026-08)
`[profile.release] lto = "thin", codegen-units = 1` in the root manifest, with
`[profile.bench] inherits = "release"` so measurements match what users ship. Measured on
`match_shape` before adopting: clean 360° 3.51 → 3.42 ms (−2.6%), clutter 6.74 → 6.61 ms
(−1.9%). Small, but free and permanent; all later numbers are against this profile. It costs
release build time, which is the trade we want on a library whose detection budget is ~5 ms.

### `warp`: `dst → src` gather, the mask as a free by-product, and the Clamp exception (2026-08)
B4 (roadmap). Every image-warping call in this workspace answers "for this **destination**
pixel, where in the **source** do I look?" (`dst[y][x] = src.sample(map(x, y))`) rather than
scattering source pixels forward — the gather direction is what lets `apply` visit every
destination pixel exactly once with no holes, independent of how the map compresses or
expands space. `Map::affine`/`Map::projective` therefore take the transform that carries a
*destination* coordinate to the matching *source* coordinate, the inverse of a fixture pose
or any other forward (source-into-destination) transform — documented prominently in the
module doc, with the one-line fix (`nalgebra`'s `.inverse()`) right next to the warning,
because getting this backwards is a silent, plausible-looking bug: the output has the right
shape and mostly-right pixels, just mirrored/scaled the wrong way.

**The validity mask is not a second pass, it falls out of the existing branch.** `apply`'s
inner loop already splits on whether every tap a sample needs (one for `Nearest`, four for
`Bilinear`) lands inside the source — that split is what lets the fast path skip
`BorderMode` entirely. `apply_with_mask` writes `255`/`0` from exactly that same branch,
so correctness of the mask is inseparable from correctness of the fast path, not a
duplicate bounds check that could drift out of sync with it. This matters because a
destination pixel that took the border-fallback path is not measurement data — a caliper or
future variation-model pixel average that mixes `Constant`-fabricated intensity in with real
image content corrupts the result with no signal that it happened. `apply` (no mask) is
still offered for callers who only want a filled canvas (e.g. a preview image) and do not
need to distinguish real from fabricated pixels.

**Every builder routes through `from_fn`.** `affine` and `projective` embed their nalgebra
transform in a closure, `polar` its `(angle, radius)` bin-center formula — so there is one
coordinate-precompute path, not four independent ones, and `from_fn` itself is exercised by
every test the other three constructors have. The cost is paid once, at construction;
`apply`'s hot loop never touches a matrix or a trig function.

**`polar`'s bin centers, not the range boundaries, are what gets sampled.** Destination `x`
sweeps `phi` and `y` sweeps `r`, both as `(index + 0.5) / count` fractions of the given
range — the same convention a histogram uses for its bins. A full-turn `phi` range therefore
never samples both `0` and `2π` (which would duplicate the seam column); a partial range
never samples its own end exactly either, which is a deliberate, documented trade rather
than an oversight — a caller who needs an exact boundary sample can request a fractionally
wider range.

**Prefiltering is a doc note, not a dependency.** `apply`'s per-pixel gather is a plain
resample, not a resampling filter — minifying by more than roughly 2× through it aliases the
same way naive nearest/bilinear scaling always does. `pyr::Pyramid` already covers
decimation; B3 (`filter`) is not a prerequisite for B4 and is not needed for the common case
this module exists for (rectifying a fixture pose at roughly unit scale).

**Accepted deviation from invariant 11 is deferred, not taken here.** Invariant 11 defaults
border handling to `Clamp`; B4 keeps that default (`apply`'s `border` parameter has no
implicit default — every call states its `BorderMode` explicitly, same as `sample_bilinear_f32`
and `sample_nearest`). The plan's rectify-crop design (`ShapeMatch::model_frame_map`, wave
W2) is expected to choose `Constant` + the mask deliberately, since replicating edge pixels
into a canonical crop fabricates texture a variation model would then treat as real — that
choice belongs to the caller building the map for that purpose, not to `warp` itself, so it
is recorded here as a heads-up for W2 rather than a decision this wave made.

### `CropSpec` / `ShapeMatch::model_frame_map`: rectify closes the seam (2026-08)
B4.1 (roadmap), the W2 rectify wave. `matching` says where the part is, `warp` moves pixels
— `CropSpec { rect: Rect2f, px_per_unit: f32, normalize_scale: bool }` plus
`ShapeMatch::{model_frame_map, model_frame_pose}` is the seam, in `matching::crop` rather
than a new `align` module: the geometry is a couple of lines over `Map::from_fn`, and a
found pose calling straight into `Map` construction (no intermediate type) is simpler than
inventing an abstraction two functions wide.

**Output size is a property of the spec, never of the match.** `CropSpec::output_size()`
computes `(rect.width, rect.height) * px_per_unit`, rounded, with no dependence on any
particular `ShapeMatch` — the requirement driving this (roadmap decision 3) is anomaly
learning needing identical tensor shapes across every frame a part is found in, regardless
of the pose it was found at that frame. Fixing size at "spec alone" rather than "spec plus
match" is what makes that guarantee structural instead of a convention callers have to
maintain.

**`normalize_scale` reuses the pose's own isometry, not a re-derivation from decomposed
parts.** `ShapeMatch::pose` is a `Similarity2f` = `Translation(position) ∘ scale·R ∘
Translation(−origin)`; expanded, its stored `isometry.translation` already equals `position
− scale·R·origin` — the origin dependence is baked in once, at match time. So
`model_frame_pose` for `normalize_scale = true` is exactly
`Similarity2f::from_isometry(pose.isometry, 1.0)`: same rotation, same translation, scale
forced to `1.0` — no `ShapeModel` (and therefore no `origin`) needs to be threaded through
`model_frame_map`'s signature at all. Rebuilding the pose from `(x, y, angle, scale)` plus a
caller-supplied `origin` (the way the existing Python `ShapeMatch.matrix()` binding has to,
predating this wave) would need that extra parameter and get the *wrong* canonical crop
besides — zeroing `scale` in the decomposed form without correcting the translation leaves
a residual offset proportional to `(scale − 1) · R · origin`. Keeping the raw pose is what
avoids both problems, and is why the Python binding added a hidden (non-`get`) `pose` field
to its `ShapeMatch` pyclass alongside the existing decomposed ones (`vm-python/src/types.rs`)
rather than reconstructing a pose in Python.

**`BorderMode::Constant`, deliberately not the workspace's `Clamp` default** — the heads-up
recorded in the `warp` entry above, exercised here: `model_frame_map`'s docs recommend
`Map::apply_with_mask` with `Constant`, since `Clamp` would replicate the scene's edge pixel
into out-of-scene crop area, fabricating texture an anomaly model would learn as normal
signal rather than "no data here".

**C1 headline: rectify repeatability, 0.88 bias / 1.69 σ (8-bit intensity units) under
sub-pixel pose jitter** — see roadmap B4.1 for the full sweep description. This is the
number that decides anomaly-pipeline viability, and it was measured (not assumed) before
being pinned as a regression envelope, same discipline as every other C1 row.

## Performance numbers (M4 Pro, single thread, release)

Record per release. The target use case budgets ~30 ms for a full multi-stage
image analysis; detection is stage 1 and must leave room for the rest.

| Bench | post-#18 | post-tiling (Track 2) | post-v0.3 reset (Track D) |
|---|---|---|---|
| `shape_model_create_1280x1024` | 0.49 ms | 0.49 ms | 0.49 ms |
| `shape_find_1280x1024_360deg` | 7.8 ms | **3.46 ms** | **3.37 ms** |
| `shape_find_1280x1024_360deg_clutter` | 10.4 ms | **6.57 ms** | **6.47 ms** |
| `shape_find_1280x1024_tracked_roi` | — | **1.49 ms** | **1.45 ms** |
| `shape_find_1280x1024_360deg_greedy0` | 11.2 ms | 5.47 ms | 5.25 ms |
| `shape_find_1280x1024_scale_0p8_1p25` | 23.1 ms | 16.95 ms | 16.83 ms |
| `direction_field_1280x1024` (full frame) | 4.0 ms | 4.0 ms (lazily skipped in find) | 4.0 ms |
| `edge2d_detect_u8_1280x1024` | 5.6 ms | 5.6 ms | 5.6 ms |

Canend real data, full 360°, median per frame: set1 dome 15 → **5.6 ms**,
bright 16.9 → 9.2 ms, dark 15 → 11.5 ms, set2 dome 63 → **25.5 ms**, conveyor
10.6 ms, CP34 9.2 ms. Detection 256/256 preserved; per-frame scores
bit-identical to the pre-tiling code (verified against `main` on identical
flags).

Re-validated after the v0.2 substrate reset (`Pixel` trait, generic pyramid,
LTO profile), set1 `normal`, `--model-min-contrast 400`:

| folder | frames | found | shape p50 | ZNCC p50 | ms p50 |
|---|---|---|---|---|---|
| dome | 50 | 50/50 | 0.998 | 0.961 | 5.50 |
| bright | 50 | 50/50 | 0.883 | 0.912 | 9.08 |
| dark | 50 | 50/50 | 0.823 | 0.954 | 11.30 |

Every folder is at or slightly below its pre-reset median (5.6 / 9.2 / 11.5 ms)
and no detection was lost.

Re-validated again after the **v0.3 API reset**, same flags: dome 50/50, shape
p50 **0.998**, median 5.7 ms — identical to three decimals. `inspect_canend` on
set1 `normal`: dome 50/50 measured, mean radius 365.237 px, σ 0.282 px; dark
50/50, 365.696 px, σ 0.307 px — both matching the pre-reset numbers.

A caution recorded while measuring: the `shape_matching` example's median score
depends on which frame is taught. On set1/bright the same folder reads p50
0.875 / 0.839 / 0.847 depending on whether the model comes from frame 1, 2 or
25. Comparing runs across commits is only meaningful with the reference frame
pinned.

Where the remaining time goes (cluttered fixture, per stage): top-level sweep
2.3 ms, candidate descent 4.2 ms, everything else <0.5 ms. The descent cost is
dominated by well-scoring candidates that legitimately never trigger the
greedy abort — reducing it further means quantized/SIMD scoring
(see backlog).
