# System design

The architecture of `vision-metrology` as it **is**, and the record of why it is that way.
This file, together with [`roadmap.md`](roadmap.md) (where we are going) and
[`backlog.md`](backlog.md) (known debt), is the persistent context for anyone — human or
agent — starting a work session on this repository. Read all three before changing code.

Keep this file truthful: when a decision here is superseded, **rewrite** the entry (and say
what replaced it), don't append a contradiction. What *landed*, with its measured numbers,
belongs in [`CHANGELOG.md`](../CHANGELOG.md); what is *decided*, and why, belongs here.
Say a thing once: if it is already written down somewhere, link there instead of restating
it.

## Layering

Three publishable crates, strict one-way dependencies:

```
vm-primitives  ──►  vision-metrology  ──►  vm-python
(low-level)         (domain algorithms)    (PyO3 bindings)
```

**This table is the canonical module map.** `AGENTS.md`, `CLAUDE.md`, `CONTRIBUTING.md`
and the crate READMEs point here rather than keeping their own copies, which is how four
of them came to list a `::shape` module that had been renamed two waves earlier.

| Crate | Modules | Contents |
|---|---|---|
| `vm-primitives` | `core` | Two private halves, one public path: `raster` (`Image<T>`/`ImageView<T>`, `Pixel` sealed over u8/u16/f32, sampling, `BorderMode`, `Error` — **no nalgebra**) and `geom` (nalgebra aliases `Point2f`/`Vec2f`/`Similarity2f`, `Vec2fExt`, transforms, `Circle2f`/`Ellipse2f`/`Conic2f`) |
| | `pyr` | `Pyramid`: 2×2 box-mean pyramid generic over `Pixel`, optional binomial pre-smooth, `level_to_base` |
| | `edge` | 1D/2D subpixel DoG edges, edgels, edge pairs, `DirectionField` |
| | `morph` | binary morphology (parameterized SE), chamfer distance, Zhang-Suen thinning |
| `vision-metrology` | `contour` | contour graph with T/Y junctions, per-edge geometry, polyline smoothing |
| | `laser` | stripe extraction via opposite-polarity edge pairs (rows/cols, ROI + prior) |
| | `matching` | `ShapeModel`/`ShapeMatcher`: gradient-orientation shape-based detection, masked teaching, canonical-pose crops (`matching::crop`) |
| | `segment` | Otsu/adaptive thresholding, CCL, watershed, edgel region growing |
| | `fit` | robust line / circle / ellipse fitting, residuals reported |
| | `measure` | calipers (rect / arc / radial), metrology model applied at a fixture pose, `diagnostics::layout` |
| | `lsd` | LSD line-segment detection |
| | `warp` | `Map`: precomputed `dst → src` coordinate table (affine / projective / polar / log-polar / `from_fn`), `apply`/`apply_with_mask` with a first-class validity mask |
| | `metric` | The calibration bridge: mirrored `CameraModel`/`Pose3`/`Plane3`/`PlaneGrid` on nalgebra 0.35, `pixel_to_plane` (exact), `plane_grid_map`/`undistort_map` (runtime `warp::Map` path), `io` importers for calibration-rs and `table_calibration` JSON |
| | `corr` | Cross-correlation matching + inter-frame `displacement` over `corrmatch` |
| | `scale` | Scale estimation for `matching` (moments / log-polar) + `find_scale_invariant`: estimate once, resample, verify narrow |
| `vm-python` | — | numpy-in/numpy-out detectors; lib target named `vm_python` (see invariants) |

Each `vision-metrology` module is a default-on feature (see invariant 18). Names live at
their module path; `prelude` is a curated convenience and any crate-root re-export is an
explicit list, never a glob (invariant 17).

## Invariants

These are load-bearing. Breaking one is a design change, not a refactor, and needs a
matching update here.

**The numbering is an API.** Source files and docs cite these by number (`invariant 4`), and
nothing in the compiler checks that. So: numbers are **append-only**. Never renumber, never
reuse. A new invariant is appended with the next number; a retired one keeps its number and
gets a `**(retired)**` tombstone saying what replaced it. `tools/check-invariants.py` (run in
CI) verifies the list is contiguous and that every `invariant N` citation in the repository
resolves to one that exists.

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

Each entry answers "why is it like this", not "what landed when" — that is
[`CHANGELOG.md`](../CHANGELOG.md)'s job. Numbers appear here only where the number *is* the
argument.

### Chamfer matcher → gradient-orientation shape matching (PR #18)
The distance-only chamfer matcher was replaced wholesale (deleted, not deprecated: the crate
was unpublished). Its metric ignored gradient orientation, it had no pyramid, no greedy
abort, and its angle range could not cross ±π. The replacement implements Steger's DAGM 2001
similarity measure (the algorithm behind HALCON `create_shape_model` / `find_shape_model`):
`S = (1/n) Σ (R·tᵢ)·ĝ(pᵢ)` with three polarity modes, greedy early termination factored to
one FMA + compare per 8-point chunk, coarse-to-fine pyramid search with 3-D (x, y, α) local
maxima, and correspondence-free least-squares pose refinement (4→3→2 DOF Cholesky fallback
for symmetric parts). `morph::chamfer_distance_u8` survives as an independent primitive.

### `DirectionField` lives in vm-primitives, not in matching
The matcher needs a dense gated unit-gradient field per pyramid level, stored across levels.
`GradientBuffers` borrows `&mut Edge2DDetector` and cannot be held per level; the full edge
pipeline (NMS + hysteresis + edgel build) is ~2× the necessary work. The field is a pure
image primitive with no matching semantics, hence vm-primitives.

### Model `min_contrast` is the knob that decides real-world performance
On low-relief parts (canend), edge-detector auto-thresholds admit model points on faint
non-repeating surface shading. Because of invariant 4 those points *dilute* the score rather
than merely not helping. Raising `ShapeModelConfig::min_contrast` took set1/dome from
0.785 → 0.998 median score and CP34 from 35/39 found to 39/39. Tuning table in
`shape-matching.md`.

### Cross-repo dependencies are crates.io releases only
The maintainer owns `corrmatch`, `box-image-pyramid`, and `calibration-rs`. When a feature is
missing there it is implemented upstream, released, and bumped here — never a git or path
dependency in committed code. Keeps this workspace publishable and CI reproducible. The
`corr` entry below records the one deliberate change of *status* (corrmatch became a real
dependency rather than dev-only), not of mechanism.

### MSRV 1.89 → 1.91
nalgebra 0.35 needs only 1.89, but `corrmatch` and `box-image-pyramid` declare 1.91. The
crates here were unpublished at the time, so no compatibility promise broke. The MSRV CI job
builds `--all-targets`, which is why even dev-dependencies constrain the floor; clippy
enforces it through `incompatible_msrv`. **This is the one place that rationale is written
down** — `AGENTS.md` and `CONTRIBUTING.md` link here.

### vision-calibration: offline/runtime split
`calibration-rs` has the laser-plane pipeline we need, but it is structurally pinned to
nalgebra **0.34** (tiny-solver/faer chain; nalgebra types cross its public API) and MSRV
1.93. Decision: vision-calibration remains the **offline** calibration system; the runtime
`metric` module here mirrors only the parameter types on nalgebra 0.35 and loads
calibration-rs JSON exports, pinned by a golden-file test. A direct dependency replaces the
mirror when upstream reaches nalgebra 0.35.

### `metric`: units, pose direction, and the homography split
Four decisions that must be right before any of the arithmetic matters, because getting one
backwards is a silent, plausible-looking bug (right shape, wrong numbers) rather than a
crash.

**Units: millimetres, not the wire format's meters.** Every 3-D/plane quantity (`Pose3`
translations, `Plane3::d`, `PlaneGrid`, `pixel_to_plane`'s output) is mm; pixel-domain
quantities stay in pixels. `io::import_rig_extrinsics` converts calibration-rs's meters on
import (its schema documents the unit). `io::import_table_calibration` needed a judgment call
instead — that format documents no unit at all, and the real fixture's camera-to-camera
translation is ~111 in magnitude: a plausible benchtop baseline in mm, an absurd 111-**meter**
rig in meters. Recorded at the importer, not asserted as a property of the format.

**`Pose3` is camera-from-reference (`T_C_R`)**, matching calibration-rs rather than inventing
a convention: `pose * p_ref` gives `p_ref` in the camera's own frame. Backwards, it would
silently mirror every downstream projection through the camera centre — the same failure
shape as `warp`'s `dst → src` trap. `import_table_calibration` treats `camera0`'s
`sensor2camera` (identity in every observed export) as defining the reference frame, so every
other camera's `sensor2camera` **is** `Pose3` directly — a fact about this dataset,
documented at the importer, not a property of the file format.

**A homography cannot carry Brown-Conrady distortion — so there are two deliberately
different plane-to-image paths.** `pixel_to_plane` is the exact one (per-pixel
`undistort_pixel` → `pixel_to_ray` → `ray_plane_intersect`), correct for any `Plane3`, the
one to use for a point-wise measurement. `homography_plane_to_image` / `plane_grid_map` are
the whole-image runtime path — an undistorted-only homography composed with forward
`distort_pixel` per destination pixel via `warp::Map::from_fn` — restricted to `PlaneGrid`,
always the reference frame's own `Plane3::xy` (`z = 0`), because the homography's linearity
depends on substituting `z = 0` into the pose's affine map. A caller needing a tilted or
offset plane's bird's-eye view has no shortcut; that is intentional, and both entry points
say so rather than leaving it to be discovered from a wrong answer.

**`pixel_to_plane`'s in-plane basis is deterministic, not caller-supplied.** Origin is the
plane's closest point to the reference origin (`p0 = -d·n`); axes come from projecting the
reference `x` axis onto the plane (or `y`, if `x` is within 0.9 of parallel to the normal),
completed right-handed. Purely a function of `plane.n`, so two calls with the same plane
always agree — and it reduces to the identity basis at `n = (0, 0, 1)`, the only case where
it must agree with `PlaneGrid`'s own axes.

**Rectify-first closes the planar 3-D case, measured rather than assumed.**
`tests/metric_rectify.rs` finds a model at every tilt 0–40° with max position error 0.012 px,
an order of magnitude inside the 0.1 px target, on a fixture where render and rectify are
exact geometric inverses. Homography refinement therefore has little left to buy on a
genuinely planar target; its value is for the *non*-planar case.

### Shape-matching preprocessing dominates the search
Measured (1280×1024, full 360°): `find` ≈ 7.8 ms = ~5.3 ms direction-field pyramid + ~2.5 ms
search. Below the top level the search reads only small windows around candidates, so
full-frame fine-level fields are mostly wasted work. The plan was lazy tiled fields first,
integer u8 Scharr second, quantized directions + SIMD only if still needed — each gated on
measurement. The first landed; the rest is in `backlog.md`.

### The chain runs end to end on real data — the canend baseline
`examples/inspect_canend` closes the loop the library exists for, on real frames: find the
tab → take its pose as a fixture → measure the can rim **in the tab's frame** → report the
fit and its residuals → pass/fail on `max_dev`. Teaching the rim relative to the tab is what
makes the numbers mean something: every frame re-derives the rim from wherever the tab turned
up, so the spread measures the fixture *and* the measurement, not where the part sat.

**These are the reference numbers every later wave is checked against, bit-for-bit.** set1
`normal`, 96 calipers, Tukey(2 px), tolerance 2 px on `max_dev`:

| folder | frames | measured | mean radius | σ | per-frame rms |
|---|---|---|---|---|---|
| normal/dome | 50 | 50/50 | **365.237 px** | **0.282 px** | 0.20–0.54 px |
| normal/dark | 50 | 50/50 | **365.696 px** | **0.307 px** | 0.28–0.60 px |

All 96 calipers survived the robust fit in every frame. σ ≈ 0.3 px is an *upper bound* on
repeatability — measured over different physical cans, so part-to-part rim variation is
inside it. A wave claiming "canend parity" means these exact figures. Related caution: the
`shape_matching` example's median score depends on which frame is taught (set1/bright reads
p50 0.875 / 0.839 / 0.847 from frames 1, 2, 25), so cross-commit comparison needs the
reference frame pinned.

### `measure`: calipers, and why a curved edge needs its own placement
`Caliper` places a geometry, averages intensity across it into a 1-D profile, and runs
`Edge1DDetector` along that profile; `MetrologyModel` distributes calipers over nominal
primitives held in the part's own frame, applies them at a fixture pose, and fits with `fit`.

**`MeasureRadial` exists because a rectangle cannot measure a circle without bias.** A rect
averages along a straight *chord*: on a circle of radius 40 a sample 5 px to the side sits at
radius 40.31 — the wrong side of the edge — so the profile is contaminated and the radius
reads low: **39.88 px instead of 40.00** on a 32-caliper fit over an anti-aliased disc.
`MeasureRadial` averages *along the arc*, so every sample sits at the same radius:
**39.990 px**, and the bias no longer grows with caliper width. This also forced the fixtures
to change: a hard-thresholded disc puts its edge 0–0.5 px inside the nominal radius depending
on sub-pixel offset, so it cannot assert subpixel accuracy at all. Anti-aliased fixtures put
the true edge at the nominal radius and let tolerances drop from 0.3 px to 0.03 px.

**Ideas taken from the `rtvt-pano` caliper**: typed rejection reasons (a caliper that finds
nothing is a *result*, and which gate rejected it separates "the part is missing" from "the
window is too short"); an **obliquity gate**, because a caliper crossing an edge at a glancing
angle reports a position along its own axis rather than the edge normal and the two differ by
`1/cos θ`; and sub-pixel profile stepping. Not taken: two-pass centreline refinement and the
mask-based background-padding gate — both properties of a *tracked contour*, not of a caliper
(see `backlog.md`).

**Caliper placement is written exactly once.** `measure_one`'s placement loop is
`model::caliper_placements` (`pub(crate)`), called by both `MetrologyModel::apply` and
`measure::diagnostics::layout`, so they cannot disagree about where a caliper is. `layout`
needs **no image**, deliberately: it is exactly what `apply` computes *before* measuring,
which is what a UI needs to draw a caliper before an image is loaded, or one that will go on
to reject every sample. `MeasureRadial::center` is the *circle's* centre, not each caliper's
point on the boundary — the measurement geometry's convention, which the lab overlay already
drew against, so reproducing it verbatim kept the lab rewrite a provable refactor rather than
a redesign. **This is the single account of caliper placement**; `backlog.md` tracks only the
still-missing per-caliper *explain* half, and `lab/README.md` links here.

### `fit`: what robustness actually requires
Every fitter is algebraic-init → geometric refine and returns `Fit<M>` with `rms`, `max_dev`,
`n_used` (invariant 21). `fit_circle` uses Taubin — near-unbiased on short arcs where Kåsa
collapses toward the chord — then Gauss–Newton on the true residual `‖p − c‖ − r`: on a 30°
arc that is the difference between visible bias and < 0.05 px.

**Two things had to be discovered by testing rather than designed.** *Tukey cannot start from
a contaminated fit*: the rejection radius is applied to the algebraic initialisation, computed
from all points including outliers, so a small radius throws away the **inliers** and keeps
whatever the corrupted fit passed through — one outlier 70 px off a 40-point circle left a
Tukey fit standing on 2 points. Fixed with graduated non-convexity (`RobustLoss::annealed`
starts wide, shrinks geometrically). *Reweighting cannot fix a flipped axis*: 30 points on
`y = 10` plus one at `(15, 60)` gives the outlier enough y-variance that total least squares
returns a near-**vertical** line — the starting guess is not inaccurate, it is orthogonal, and
no IRLS scheme recovers. That is what RANSAC is for, and `fit_line` was silently ignoring
`FitConfig::ransac`; it is honoured now.

### Module hygiene: preludes, explicit re-exports, feature gates
`vision-metrology` re-exported `vm_primitives::*` with a glob, giving every name two paths and
hiding what this crate's surface actually was. Replaced by `pub use vm_primitives;` (the crate
itself), an explicit curated re-export list, and a `prelude` on both crates whose contents
follow the feature gates (invariants 17, 18). Feature implications: `segment` → `contour`,
`serde` → `matching`, `matching` → `warp` (`matching::crop` builds a `Map`). The gates are
only worth having if checked, and `--all-features` can never catch a `#[cfg]` that forgot a
gated import, so CI runs `cargo hack --each-feature` over `vision-metrology` and
`--feature-powerset` over `vm-primitives`. `Error` is `#[non_exhaustive]`.

### The tiling protocol became a session (and `shape` became `lsd`)
`shape` stopped being a true name when conic fitting moved into `fit`: one algorithm was
left. `DirectionField`'s lazy tiling was a three-step stateful protocol on one type where the
ordering was documentation, the image was passed twice and checked with a runtime `assert!`,
and reading a field never put into tiled mode was silently all-zero rather than a compile
error. `begin_tiled_f32` now returns a `TiledField<'a>` session borrowing **both** the field
and the image; `ensure_rect` lives only on the session and takes no image, so it cannot be
given the wrong one; the session derefs to the field, so scoring code is unchanged;
`#[must_use]` catches a caller who enters tiled mode and drops the session. Every
`match_shape` case held or improved, so the safety came free.

### `core` splits into `raster` and `geom`, and the raster half names no nalgebra
`core` was seven flat files in which the dependency on nalgebra was invisible. It is now two
private submodules with one rule between them: **`core::raster` mentions no linear-algebra
type at all.**

The rule is not aesthetic. The ecosystem around this workspace already carries five
near-duplicate `ImageView` types, and `rtvt-image` is a knowing fork of this very crate that
exists *only* because it is pinned to nalgebra 0.34 while this one is on 0.35. An image
buffer has nothing to do with linear algebra; the reason it could not be shared was that the
two were in the same module. A raster layer that names no nalgebra type is the piece that can
cross a major-version boundary, and this split makes extracting it later a move rather than a
rewrite. The audit came out clean — `sample_bilinear_f32` already took bare `f32`
coordinates, and `core::sample_bilinear_at` (taking a `Point2f`) is the geometry-side
convenience that keeps it that way. Both submodules stay **private**, so every name keeps its
single canonical `core::…` path: this is a rule about dependencies, not import paths.

### The stored model format: opaque, read-only, versioned 1 → 5
A stored `ShapeModel` is the one artefact of this crate that outlives a build, so every change
to it invalidates files on disk. The version-by-version table is the doc comment on
`matching::model::persist::FORMAT_VERSION`; the decisions behind it are:

**The format is opaque.** `to_json`/`from_json` and the public `SHAPE_MODEL_FORMAT_VERSION`
became `save`/`load` and `to_bytes`/`from_bytes` at format 3. The encoding is JSON today and
that is not a promise: a documented wire format makes every internal field a compatibility
obligation, and the version constant existed only to let callers hand-assemble an envelope
this crate should be the only writer of. Nothing a caller could do with the number is not
better answered by `load` returning an error, so it is `pub(crate)` and the tests assert the
property that matters — a foreign document is refused, not mis-read.

**`ModelPoint` and `ShapeModelLevel` are readable, not writable.** Overlay code draws model
points, so reading stays easy; *writing* does not, because point order is load-bearing (greedy
termination evaluates a prefix, which must sample the whole contour) and
`radius`/`angle_step`/`scale_step` are derived quantities the search trusts. The model also
carries its own `PreSmooth` (format 3), which `ShapeMatcher::find` reads off the *model*
rather than its own config — invariant 3 made unbreakable.

**Bumps are batched, and since format 3 they are backward-loading.** Format 3 was a single
batched break taken while the crates were unpublished, and is the floor `load` still accepts
(`MIN_SUPPORTED_FORMAT_VERSION`), because below it the document *shape* differs rather than a
field being absent. Every bump since adds an optional `#[serde(default)]` field, so an older
document loads and reads the value that field always implicitly meant: format 4 added the
pre-decimation `teach_points` (a format-3 document reads `teach_point_count() == 0`, and
`resample_at`/`estimate_scale_logpolar` then refuse cleanly rather than resampling from
already-decimated points), and format 5 added `reference_angle` (a format-4 document reads
`0.0`, exactly what it meant). An earlier version of this entry claimed format 3 would be the
last bump in the series; that was wrong, and the rule that actually holds is the one stated
here — batch the breaking bumps, make the additive ones loadable.

### A measurement that found nothing is a result, not an empty slice
`Caliper` had `measure` returning `&[MeasureEdge]` and `measure_checked` returning
`Result<&[MeasureEdge], RejectReason>` — the same computation, one of them throwing away the
diagnosis. `MetrologyModel` had the same pair, with the lossy `apply` additionally
*renumbering*: objects whose fit failed were skipped, so `results[i]` was not object `i` and
the caller could not tell which one was missing.

Both lossy twins are deleted. `Caliper::measure` returns the `Result`; `Ok(&[])` is
unrepresentable, because an extraction that found nothing always has a `RejectReason`.
`MetrologyModel::apply` returns `Vec<Result<MetrologyResult, Error>>`, one entry per object in
`objects()` order, and `hits()` — a parallel array the caller had to keep aligned by hand and
which a second `apply` silently invalidated — folded into the result.

This is the general rule for diagnostics: cheap borrowing accessors stay, diagnostic
*computation* lives in a `diagnostics` module off the hot path, and anything a result was
already computing travels with that result instead of in a side channel.

### Configs say what they mean: the split, the sentinels, and `Contrast`
Invariant 10 states the rules; this is what applying them found. `ShapeSearchConfig` had 14
flat fields, six of which are search *effort* — they trade run time against the chance of
missing a match `min_score` says should be reported, and presenting them next to `min_score`
invites tuning by field name. They live in `tuning: ShapeSearchTuning`; `LaserExtractConfig`
split the same way.

Two encodings of one decision, and one field meaning two things, were the real finds:
`Edge2DConfig`'s two threshold fields (a caller could set one of the pair and silently get
neither meaning) became one `Hysteresis::{Auto, Manual{low, high}}`;
`Edge2DConfig::pre_smooth: bool` alongside `smooth_kind` made `pre_smooth: false, smooth_kind:
Binomial3` representable and meaningless; `LaserExtractConfig::enable_smoothing: bool` hid a
real median-of-5 filter whose window no caller could see, now `CenterSmoothing`.

**`Contrast`.** `min_contrast` was documented as "Scharr response units on the input pixel
scale" — a number whose meaning changes by 257× between `u8` and `u16` data of identical
physical contrast. `Contrast::Raw(f32)` keeps that behaviour and is the default;
`Contrast::FractionOfRange(f)` resolves to `f · 16 · (max − min)` of the image being processed
(16 being Scharr's response to an ideal unit step) and therefore transfers between pixel types
unchanged. The model resolves it against the reference *ROI*, the search against the scene;
`ShapeModel::from_edgels` has no image and returns `InvalidConfig` rather than inventing a
range from the very strengths it is about to filter.

### `Point2f` / `Vec2f` are nalgebra aliases — and the NaN trap that came with them
The crate previously carried its own two-field structs plus seven public conversion functions
— while invariant 13 says not to re-implement linear algebra and `Similarity2f` already put
nalgebra in the public API. Aliasing removed the parallel type system (7 conversion functions
and ~250 lines of hand-written operators gone) and lets points cross into `calibration-rs` /
`corrmatch` / `chess-corners-rs` untouched. Costs accepted knowingly: 147 struct literals,
`dot` taking a reference, `perp`/`cross`/`normalized_or_zero` moving to the `Vec2fExt` trait,
and a model wire-format change (nalgebra serializes a vector as a flat `[x, y]`).

**The trap this introduced, and the guard against it.** nalgebra's `normalize` divides
unconditionally, so a zero vector yields `NaN` where the old hand-written version returned
zero. Four call sites normalize a possibly-degenerate gradient or tangent and then reject it
with `t.norm() < 0.5` — and `NaN < 0.5` is **false**, so those guards would have silently
stopped rejecting and let `NaN` directions into the model and the score.
`Vec2fExt::normalized_or_zero` restores the old semantics, is what every such site calls, and
is documented on the trait method and pinned by a test.

### One generic entry point per algorithm, and the measurement that overrode the tidier design
Invariant 19's rule cost **40 entry points → 16**; `vm_primitives::Pixel` is the sealed trait
that made it possible, carrying `to_f32`, `from_f32_sat`, an `Acc` accumulator, and
`as_f32_slice` (the zero-copy hook that keeps an `f32` laser scan from being widened into
scratch it does not need).

Gathering laser columns directly as `f32` looks like it saves a widening pass. It does not: it
quadruples the gather's write traffic, while the widening it avoids happens only over the much
smaller ROI segment. `laser_extract_cols_gather_512x1280` went 151 → 183 µs (+21%) before an
A/B against a baseline measured in the same session caught it. Gathering the pixel type as-is
is back, and the reasoning is recorded in `laser/gather.rs` so it is not "simplified" again.

### Why invariant 2 has exactly one implementation: LSD's private downsample, and `multiscale`
Both were found by the v0.2 audit and both were the *same* defect — a second, divergent copy
of the pyramid coordinate map. `lsd/detect.rs` carried its own 2×2 downsample under a comment
claiming it matched a `pyr` level (it did not: `div_ceil` + border clamp against `pyr`'s
drop-odd) and mapped positions back with a bare `p · (W / w)`, missing the `(2^l − 1)/2` term:
on a step edge whose true position is 59.5, the default config reported −0.500 px at width 128
and −0.954 px at width 129. `MultiScaleEdgeDetector` had the same missing term, and mixed with
correct level-0 edgels that is a *systematic* bias — `measure_circles` measured every centre
0.07–0.10 px low, growing with radius, under assertions (5 px on centre) far too loose to
notice.

LSD was fixed (it now downsamples through `pyr`, and its two fictional config fields — a
`scale` that always halved whatever it was set to, and a `sigma_scale` that was never read —
became honest `downscale_levels` / `pre_smooth`). `multiscale` was **deleted instead**,
because two further things were wrong: dedup keyed on `idx * 2^l`, so a level-3 edgel claimed
one level-0 pixel instead of the 8×8 block it stood for, and the reported
`scale = base_sigma · 2^l` was meaningless, since a box-mean pyramid with a fixed-σ DoG is not
a Gaussian scale space. Fixing only the bias would have left a module that is neither a scale
space nor a sound merge, with no consumer in the workspace. `pyr::level_to_base` /
`base_to_level` is now the single implementation; genuine scale selection, if ever needed,
gets designed as a real scale space rather than reviving this.

### Release profile is tuned, and benches inherit it
`[profile.release] lto = "thin", codegen-units = 1`, with `[profile.bench] inherits =
"release"` so measurements match what users ship. Measured before adopting: clean 360°
3.51 → 3.42 ms (−2.6%), clutter 6.74 → 6.61 ms (−1.9%). Small, but free and permanent; all
later numbers are against this profile, and it costs only release build time — the right trade
on a library whose detection budget is ~5 ms.

### `warp`: `dst → src` gather, the mask as a free by-product, and the Clamp exception
Every image-warping call in this workspace answers "for this **destination** pixel, where in
the **source** do I look?" (`dst[y][x] = src.sample(map(x, y))`) rather than scattering source
pixels forward — the gather direction is what lets `apply` visit every destination pixel
exactly once with no holes, independent of how the map compresses or expands space.
`Map::affine`/`Map::projective` therefore take the transform carrying a *destination*
coordinate to the matching *source* coordinate, the inverse of a fixture pose or any other
forward transform — documented prominently with the one-line fix (`.inverse()`) next to the
warning, because getting this backwards is a silent, plausible-looking bug: right shape,
mostly-right pixels, mirrored or scaled the wrong way.

**The validity mask is not a second pass, it falls out of the existing branch.** `apply`'s
inner loop already splits on whether every tap a sample needs (one for `Nearest`, four for
`Bilinear`) lands inside the source — that split is what lets the fast path skip `BorderMode`
entirely. `apply_with_mask` writes `255`/`0` from exactly that branch, so correctness of the
mask is inseparable from correctness of the fast path rather than a duplicate bounds check
that could drift out of sync. This matters because a destination pixel that took the
border-fallback path is not measurement data — a caliper or variation-model average that mixes
`Constant`-fabricated intensity with real image content corrupts the result with no signal
that it happened. `apply` (no mask) stays for callers who only want a filled canvas.

**Every builder routes through `from_fn`** — `affine`/`projective` embed their nalgebra
transform in a closure, `polar`/`log_polar` their bin-center formula — so there is one
coordinate-precompute path, not five, and `from_fn` is exercised by every test the others
have. The cost is paid once, at construction; `apply`'s hot loop never touches a matrix or a
trig function. `polar` samples **bin centers, not range boundaries**: destination `x` sweeps
`phi` and `y` sweeps `r`, both as `(index + 0.5) / count` fractions, the convention a histogram
uses. A full-turn `phi` range therefore never samples both `0` and `2π` (which would duplicate
the seam column); a partial range never samples its own end exactly either — a deliberate,
documented trade, since a caller needing an exact boundary sample can widen the range
fractionally.

**Prefiltering is a doc note, not a dependency.** `apply`'s per-pixel gather is a plain
resample, not a resampling filter — minifying by more than roughly 2× aliases the way naive
scaling always does. `pyr::Pyramid` already covers decimation, so B3 (`filter`) is not a
prerequisite for `warp`.

**The `Clamp` exception, stated once here.** Invariant 11 defaults border handling to `Clamp`,
and `apply` keeps no implicit default at all — every call states its `BorderMode`. The rectify
path (`ShapeMatch::model_frame_map`) deliberately recommends `Constant` + the mask instead:
`Clamp` would replicate the scene's edge pixel into out-of-scene crop area, fabricating
texture an anomaly model would learn as normal signal rather than "no data here". That choice
belongs to the caller building the map, not to `warp`.

### `CropSpec` / `ShapeMatch::model_frame_map`: rectify closes the seam
`matching` says where the part is, `warp` moves pixels — `CropSpec` plus
`ShapeMatch::{model_frame_map, model_frame_pose}` is the seam, in `matching::crop` rather than
a new `align` module: the geometry is a couple of lines over `Map::from_fn`, and a found pose
calling straight into `Map` construction is simpler than an abstraction two functions wide.

**Output size is a property of the spec, never of the match.** `CropSpec::output_size()` is
`(rect.width, rect.height) * px_per_unit`, rounded, with no dependence on any particular
`ShapeMatch` — the requirement driving this is anomaly learning needing identical tensor
shapes across every frame a part is found in, whatever pose it was found at. Fixing size at
"spec alone" is what makes that guarantee structural instead of a convention callers must
maintain.

**`normalize_scale` reuses the pose's own isometry, not a re-derivation from decomposed
parts.** Rebuilding the pose from `(x, y, angle, scale)` plus a caller-supplied `origin` would
need an extra parameter *and* produce the wrong canonical crop, because zeroing `scale` in the
decomposed form without correcting the translation leaves a residual offset proportional to
`(scale − 1) · R · origin`. Keeping the raw pose avoids both; the algebra is documented at
`ShapeMatch::model_frame_pose`. It is also why the Python binding carries a hidden `pose`
field on its `ShapeMatch` pyclass rather than reconstructing a pose in Python.

### `corr`: corrmatch as the standard cross-correlation engine
**Supersedes the earlier framing of `corrmatch` as validation-only tooling** — a user
decision, not an architectural default: corrmatch moved from dev-dependency to a real,
optional dependency (`corr` feature, default-on). An earlier proposal argued for a native port
to keep the matcher's validator independent of what it validates; that is explicitly rejected
— `tests/corrmatch_bridge.rs` still cross-checks the shape matcher against corrmatch (the
bridge is about two *different algorithms* agreeing, not about build-graph independence), and
a native port would duplicate mature SIMD/pyramid/beam-search code with a strictly worse copy
for no metrology benefit.

**Wrapper boundary: thin, and the boundary is deliberate.** `CorrTemplate`/`find`/`find_topk`
translate coordinates and errors at the edge and otherwise call straight into corrmatch. Two
things are genuinely new: the zero-copy-when-contiguous `ImageView` adapter (row-by-row copy
only when the caller's view is strided), and `CorrConfig`/`CorrTemplateConfig` as this crate's
own Option-sentinel config types (invariant 10) rather than re-exporting corrmatch's
`#[non_exhaustive]` structs — which would leak their struct-literal restrictions into this API
and couple a semver bump there to one here. `CorrMatch` is a new type, not a re-export,
precisely because it is *not* interchangeable with `ShapeMatch`: its score is a raw
correlation coefficient, not `1 − occluded_fraction`, and there is no scale search. The doc
comment says so, so a caller is corrected at the type rather than at runtime.

**`u8`-only, honestly.** corrmatch's published API (0.2.5) is `u8`-only; silently converting
`u16`/`f32` down here would be exactly the "tuned on one pixel type, wrong on another" bug
`Contrast::FractionOfRange` exists to prevent. `u16` support is corrmatch's own backlog, not
something a wrapper should paper over with a quantizing cast.

**`displacement`'s two stages, and why Lucas-Kanade is implemented here rather than requested
upstream.** Stage 1 (corrmatch, rotation off) is bounded to a `search`-pixel margin around the
window's previous position — a fresh sub-image ROI, not a whole-scene search — because
inter-frame motion is small by construction and a whole-scene search risks locking onto a
distant unrelated peak that happens to score higher. Stage 2 is translation-only
inverse-compositional Lucas-Kanade: Hessian and steepest-descent images come from the
*template's own* gradient, computed once, so every iteration after the first costs one bilinear
resample plus a 2x2 solve. This corrects what corrmatch's subpixel refinement cannot: a
quadratic fit to a *discrete* correlation surface is a known biased estimator (pixel-locking
toward integer positions), and only a second, differently-biased estimator removes that bias.
Application-level accuracy work, not a corrmatch gap.

### Mosaic: nearest-camera-centre priority, no blending
**Not a library module** — deliberately: `metric::plane_grid_map` per camera plus
`warp::Map::apply_with_mask` are already everything a bird's-eye composite needs, so the only
genuinely new logic is the *compositing rule*, small enough (~40 lines) that a module would be
more API surface than the logic it wraps, and with no *library* consumer at all.

**Priority rule: nearest-camera-centre, ties by index, no blending by default.** For each
destination grid pixel, among the cameras whose validity mask is set there, keep the one whose
reprojection of that plane point (through `pose`, perspective-divide, `distort_pixel` — the
same forward geometry `plane_grid_map` composes internally) lands closest to its own principal
point. Deterministic, needs no new `warp::Map` accessor (the reprojection is recomputed from
public `metric` functions, not read out of `Map`'s private per-pixel table), and physically
sensible: a camera's image is most reliable near its optical axis.

**Why no blending is the important half of the decision.** A metrology library's mosaic exists
to be measured on, and an averaged pixel at a seam cannot be traced back to one camera's
calibration — which distortion model, which extrinsic, which pixel produced it becomes
ambiguous exactly where two independently-calibrated views disagree most. The `source_id` map
(camera index, `255` for uncovered) is therefore first-class output, not a debugging aid:
every mosaic pixel that is not `255` traces to exactly one camera, which is what makes a
caliper or fit placed on the mosaic a real, attributable measurement. **Feathering exists only
as an opt-in, display-only mode.**

**The fixture-vs-real-data seam disparity gap is the metric's property, not the geometry's.**
The synthetic fixture measures p95 seam disparity 0.000 on antialiased fiducials; the
real-data example (a hard-edged checkerboard) measures p95 251.00 under the *same* compositing
rule. Both are correct: a razor edge turns any sub-pixel disagreement into a full-range
intensity swing, where an antialiased pattern turns the same disagreement into a few intensity
units. This is why C1 pins envelopes to the *measured* number for a given fixture rather than
to an assumed "small is always right" threshold.

### Tauri desktop shell: commands/events, not a second HTTP backend
The lab has a second shell, `lab/frontend/src-tauri` (crate `vm-lab-desktop`), alongside the
FastAPI/browser one — **one frontend, two transports**, not two frontends. Browser stays on
FastAPI; desktop talks to `vision-metrology` through **native Tauri commands and events**,
never HTTP — no sidecar process, no port to discover, no CORS policy to keep in sync with a
dev-server port. Neither path re-implements an algorithm: a command is the same shape as a
FastAPI router (build a config, call the library, translate to a serializable DTO) without the
PyO3 or HTTP hop, which is what makes "the two shells agree" mechanically checkable rather
than asserted.

**Contract fixtures are the check.** `lab/contract/fixtures/` holds golden request/response
JSON captured from the FastAPI backend over small deterministic synthetic images, replayed by
two independent tests over the *same* JSON — `lab/backend/tests/test_contract_fixtures.py` and
`lab/frontend/src-tauri/tests/contract_parity.rs` (plain functions over `&AppState`, no GUI).
Both existing at once is the point: a change to `vm_lab`'s response shape *or* to the Rust
command layer is caught by whichever replay it broke, not found later as a UI-only bug in one
shell. It earned its keep on day one: `commands::rectify::rectify` locked `state.images`, then
called `run_find`, which locks it again on the same thread — `std::sync::Mutex` is not
reentrant, so that is a guaranteed self-deadlock on the path both `rectify` and (transitively)
`measure` share. The test hung rather than failing fast (`cargo test` at ~0% CPU: a
stuck-not-crashing test process under a mutex is a deadlock, not a slow computation).

**Standalone Cargo workspace, verified, not assumed.** `src-tauri/Cargo.toml` declares its own
empty `[workspace]` table specifically so Cargo's upward directory search cannot sweep it into
the repo-root workspace — checked with `cargo metadata --no-deps` from the repo root.

**Superseded (roadmap Track E).** Two first-wave simplifications — PNG tiers re-encoded per
request with no cache, and a synchronous-`imageUrl` blob-cache bridge returning a 1x1
placeholder on a miss — are being replaced by a content-addressed on-disk tier cache served
through Tauri's asset protocol, plus `async` command wrappers over `spawn_blocking`. The
mosaic compositor still has no Tauri command; that gap is tracked once, in
[`backlog.md`](backlog.md).

### Scale-invariance: estimate-then-verify, not a wider scan
C1's scale-sweep row found that a *taught-wide* model already found 100% of a clean synthetic
sweep across 0.5–2.0×, contradicting the working hypothesis that the discrete scan degrades
badly away from 1.0. That narrowed the real motivations to three: search **cost** across a
wide range is linear in how wide it is; a model taught with the **default**
`scale_range = (1, 1)` cannot be found away from 1.0 by a scan at all, however wide the search
config asks (`matcher::intersect` clamps to the model's own range); and `(scale·d).round()`
**point-collapse** at `scale < 1` silently inflates the score. The strategy is "estimate once,
resample, verify narrow" — constant cost regardless of how wide a *would-be* scan would have
needed to be.

`ShapeModel::resample_at(s)` rebuilds every level by scaling each stored `TeachPoint.d` and
feeding the result through `matching::build`'s own geometry-only assembly — the exact pipeline
`from_directed_points` already uses, not a second implementation of "decimate a point cloud
into levels". Why that is *exact* rather than approximate (pyramid coordinates are affine, so
the additive term cancels on an offset) is derived in `matching::resample`'s module docs, where
it belongs. The resampled model's `scale_range` is pinned to `(0.95, 1.05)`, or the
constant-cost claim stops holding. `find_scale_invariant` chains an estimator, `resample_at`
and the narrow verify, then rebuilds the returned pose into the **original** model's own
reference frame — not the resampled model's, which is a fictional coordinate system scaled by
`ŝ` that would corrupt anything downstream assuming taught coordinates.

**Two independent scale estimators**, because they need different things from a scene.
`estimate_scale_moments` segments an isolated blob and compares its **outer radius** (maximum
distance from its own centroid to any foreground pixel) against `model.level(0).radius()`. The
first implementation used a *radius of gyration* — the literal "second moment" — and was wrong
by construction: a filled disc's is `R/√2` while its boundary ring's is `R`, so comparing a
filled scene silhouette against boundary-only model points biased the result (recovered 1.13
against a true 1.6). Outer radius needs no filled-area assumption, which is also why this
estimator works on **any** `ShapeModel` — it never reads `teach_points`.
`estimate_scale_logpolar` needs no segmentation but does need teach data and an approximate
centre, and correlates **synthesized edge-density rasters** rather than photometric patches — a
deliberate deviation from "resample the taught patch", forced by what the model stores: there
is no reference *image* in a `ShapeModel`, only edge points. Both sides are splatted,
log-polar-unwrapped (`warp::Map::log_polar` — logarithmic radial spacing, so a uniform scale
change is a constant additive row shift: Fourier-Mellin without an FFT) and correlated with
`corr::find`. Comparing edge-density fields is also what `ShapeMatcher` does, so the
estimator's photometric assumptions stay consistent with the matcher it feeds.

**Decision 9g — offset-collapse dedup: three designs tried, all three rejected by measurement,
none shipped.** *(Authoritative account; `backlog.md` records only the open question a fourth
attempt must answer.)* Rotating and scaling a model point rounds it to an integer pixel; at
`scale < 1` two points distinct at their build-time grid resolution (invariant 4) can round
onto the *same* pixel, and reading that scene pixel's gradient twice inflates a score for a
rounding coincidence rather than real coverage — real, but small: worst measured position bias
0.022 px, scale bias 0.14%, on the clean synthetic C1 fixture. The discipline is "measure
before assuming a fix is worth its cost", and every fix failed that measurement, differently:

1. **Dedup inside `matching::score::rotate_into` whenever `scale < 1`.** Simple, but
   `refine::interpolate`'s subpixel parabola fit perturbs an *already-found* pose by one
   `scale_step` in each direction, which dips below 1.0 even when the model's own `scale_range`
   never leaves `(1, 1)` — canend's own config — so this measurably moved `inspect_canend`'s
   rim radius by 0.002 px on a search that never asked for a scale scan. The gate is "identical
   to the recorded baseline", not "close to it".
2. **Dedup in `rotate_into`, but only for the coarse-to-fine sweep**, with a reused scratch
   buffer so it does not allocate. This fixed (1), but `rotate_into` is the hot inner loop
   `match_shape`'s benches measure — once per (angle, scale) grid point — and even a reused
   `O(m log m)` sort there measured **+35%** on `shape_find_1280x1024_scale_0p8_1p25`
   (→ 23.4 ms; baseline in the performance table) for a correction thrown away the moment a
   better candidate is found: the sweep's scores exist only to *rank* candidates.
3. **Dedup in `score::score_pose` only** — the function computing the score actually attached
   to a reported `ShapeMatch`, called a handful of times per `find()`. Cheap: every bench held
   within noise. But **measured worse than the bug it fixed**: `matcher::run` rejects a
   candidate whose `score_pose` result falls below `min_score` *after* the sweep has already
   picked it as best using the sweep's own still-undeduped scores. Honestly *lowering* that
   score can push it under `min_score`, discarding the search's actual best answer in favour of
   whatever next-best candidate — often at a substantially different position or scale — still
   clears the threshold. Measured on the C1 fixture: position error up to **0.46 px** at some
   swept scales (worst, `scale ≈ 0.73`), roughly 20x the ≤0.022 px the *unfixed* inflation ever
   cost. Reverted.

**What shipped: nothing changed in `matching::score`.** `rotate_into`/`score_pose` are
bit-for-bit the pre-wave code; the inflation is documented on both functions and pinned by
`score::tests::offset_collapse_at_reduced_scale_is_a_known_unfixed_score_inflation`, so a
future change cannot silently reintroduce design 3's regression while believing it fixed the
original bug.

### Masked teaching and the model's own reference angle
Two additions for the same reason: a model taught from a bare rectangle learns whatever
background the rectangle contains, and because of invariant 4 those points *dilute* every later
score. `ShapeModelBuilder::build_with_mask` takes an optional inclusion mask, tested at the
level-0 position a point was aggregated from and dilated, because a coarse level's point can
sit up to half its own pixel from the fine edge that produced it — being stingy silently
deletes the coarse levels, far worse than admitting the odd neighbouring edgel.
`ShapeModelConfig::reference_angle` rotates the model *frame* onto a caller-chosen canonical
orientation at build time, so a found pose reads as "how far from canonical" rather than "how
far from however the reference frame happened to be shot". It rotates the frame; it does not
filter points, and `reference_geometry` still reports the taught geometry in the reference
image's own frame while `model_geometry` reports it in the rotated model frame.

## Performance numbers (M4 Pro, single thread, release)

Record per release. The target use case budgets ~30 ms for a full multi-stage image analysis;
detection is stage 1 and must leave room for the rest. **This table is the single place bench
numbers are kept** — a decision entry that needs one links here rather than restating it.

| Bench | post-#18 | post-tiling | post-v0.3 reset |
|---|---|---|---|
| `shape_model_create_1280x1024` | 0.49 ms | 0.49 ms | 0.49 ms |
| `shape_find_1280x1024_360deg` | 7.8 ms | **3.46 ms** | **3.37 ms** |
| `shape_find_1280x1024_360deg_clutter` | 10.4 ms | **6.57 ms** | **6.47 ms** |
| `shape_find_1280x1024_tracked_roi` | — | **1.49 ms** | **1.45 ms** |
| `shape_find_1280x1024_360deg_greedy0` | 11.2 ms | 5.47 ms | 5.25 ms |
| `shape_find_1280x1024_scale_0p8_1p25` | 23.1 ms | 16.95 ms | 16.83 ms |
| `direction_field_1280x1024` (full frame) | 4.0 ms | 4.0 ms (lazily skipped in find) | 4.0 ms |
| `edge2d_detect_u8_1280x1024` | 5.6 ms | 5.6 ms | 5.6 ms |

Later module benches (all post-v0.3): `fit_circle_500pts` 2.6 µs, `+tukey` 4.2 µs,
`fit_line_500pts` 1.7 µs, `fit_ellipse_100pts` 1.6 µs, `fit_ellipse_ransac_1000pts` 430 µs;
`affine_apply_640x480_bilinear` ≈ 510 µs, `polar_apply_640x480_bilinear` ≈ 494 µs;
`corr::find` (VGA scene, 64x64 template) 4.60 ms rotation-off / 22.1 ms rotation-on;
`corr::displacement` (320x97 window) 1.60 ms quadratic / 1.71 ms +Lucas-Kanade.

Canend real data, full 360°, median per frame: set1 dome 15 → **5.6 ms**, bright
16.9 → 9.2 ms, dark 15 → 11.5 ms, set2 dome 63 → **25.5 ms**, conveyor 10.6 ms, CP34 9.2 ms.
Detection 256/256 preserved, scores bit-identical to the pre-tiling code. Re-validated after
each reset wave (set1 `normal`, `--model-min-contrast 400`): dome 50/50 at shape p50 0.998 and
~5.5–5.7 ms, bright 50/50 at p50 0.883 / 9.08 ms, dark 50/50 at p50 0.823 / 11.30 ms — at or
below the pre-reset medians, no detection lost, `inspect_canend` matching the canend baseline
table above.

Where the remaining time goes (cluttered fixture, per stage): top-level sweep 2.3 ms, candidate
descent 4.2 ms, everything else <0.5 ms. The descent is dominated by well-scoring candidates
that legitimately never trigger the greedy abort — reducing it further means quantized/SIMD
scoring (see `backlog.md`).
