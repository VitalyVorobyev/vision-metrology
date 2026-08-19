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
                           fixture: &Similarity2f) -> &[MetrologyResult];
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
Separable and recursive (Deriche / van Vliet) Gaussian, sliding-window box mean, histogram
median and rank (O(1) per pixel), grayscale erode/dilate/open/close/tophat
(van Herk–Gil-Werman). `edge/conv1d.rs` folds in here. Feeds the pyramid pre-smooth and
illumination correction.

**Accept:** each filter matches a naive reference bit-for-bit on random fixtures; median and
rank are O(1) in radius, measured rather than asserted.

### B4 — `warp`: build once, apply per frame
```rust
impl Map {
    pub fn affine(w, h, m: &Affine2f) -> Self;
    pub fn projective(w, h, h: &Projective2f) -> Self;
    pub fn polar(center: Point2f, r: Range<f32>, phi: Range<f32>, w, h) -> Self;
    pub fn from_fn(w, h, f: impl Fn(f32, f32) -> (f32, f32)) -> Self;
    pub fn apply<P: Pixel>(&self, src, dst, interp: Interp, border: BorderMode<P>);
}
```
`polar` is the round-part unwrap, directly useful on canend. `from_fn` +
`metric::undistort_pixel` gives undistortion maps for free once B5 lands.

**Accept:** `polar` followed by its inverse recovers the source within 0.05 px; an affine
map composed with its inverse is the identity to 1e-4.

### B5 — `metric`: the calibration bridge
Mirror `PinholeIntrinsics`, `BrownConrady5`, `LaserPlane` on nalgebra 0.35; alloc-free
`undistort_pixel`, `pixel_to_ray`, `ray_plane_intersect`, `laser_line_to_profile`,
`pixel_to_plane_mm`. Offline/runtime split as recorded in system-design.

**Accept:** golden-file numeric parity with a real calibration-rs export; the laser →
3-D-profile demo runs from Python.

**Track B accept:** `examples/inspect_canend` runs find → fixture → metrology model →
pass/fail on real canend data — **done in pixels**; the millimetre step waits on B5.

---

## Track C — credibility and infrastructure — `planned`

### C1 — accuracy regression suite  ← the differentiator
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

### C2 — blob features
`ComponentStats` is bbox + centroid + count. Add second-order moments → orientation and
elongation, plus convex hull, min-area rect, circularity, rectangularity. Cheap on top of
the existing CCL and needed for blob-based inspection.

### C3 — bindings and CI
Python dtype dispatch (the bindings still accept only `uint8` while Rust is generic);
generate the vm-python config conversions instead of hand-mirroring 588 lines;
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
