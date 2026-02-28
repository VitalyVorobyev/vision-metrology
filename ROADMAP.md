# vision-metrology — Development Roadmap

**Last updated:** 2026-02-28
**Status:** Active development — Milestone 4 complete

---

## 1. Executive Summary

`vision-metrology` is a Rust-native, industrial-grade image processing library targeting
high-precision measurement and object analysis pipelines. When complete it will provide:

- **Subpixel edge extraction** at multiple scales (already working)
- **Laser stripe profiling** with coarse-to-fine ROI tracking (already working)
- **Junction-aware contour graphs** from 2D edgels (already working)
- **Line Segment Detection (LSD)** — gradient-coherence region growing, NFA validation,
  subpixel line endpoints
- **Ellipse and general conic fitting** — algebraic least-squares, RANSAC-validated,
  subpixel accuracy
- **Directed-edge object matching** — rigid (R|t) and similarity alignment of model edgel
  sets against scene edgels; Hausdorff / chamfer distance scoring
- **Image segmentation** — Otsu / adaptive thresholding, connected-component labeling,
  watershed, edgel-driven region growing
- **Multi-scale edge detection** — coordinated pyramid + edge detector combinator
- **Python bindings** (PyO3, via a top-level `vm-python` crate)

Target performance budget: **< 100 ms per 1280 x 1024 frame** end-to-end on a single
modern x86-64 core. Rayon parallel paths are allowed wherever they do not break
determinism of the output geometry.

---

## 2. Current State Assessment

| Crate | Status | Notes |
|---|---|---|
| `vm-core` | Solid | Image views, stride, border modes, sampling, Point2f/Vec2f/Line2f. Needs `Error` variant expansion. |
| `vm-pyr` | Solid | 2x2 mean pyramid, unsafe fast path, benchmarked. |
| `vm-edge` | Solid | 1D DoG, 2D Scharr + NMS + hysteresis, parabolic subpixel. Benchmarked. |
| `vm-laser` | Solid | Row/col scan, coarse-to-fine ROI, gap tracking, Rayon feature. Benchmarked. |
| `vm-contour` | Solid (topology) | Graph construction, C4/C8, loop/junction/end nodes, arc tracing. Missing: curvature, tangents, parameterization on `GraphEdge`. |
| `vm-morph` | Minimal | Only 3x3 binary erode/dilate/open/close. No parameterized SE, no distance transform, no thinning. |
| `vm-gallery` | Solid | External fixture CLI. Covers all current algorithms. |
| `vision-metrology` | Thin umbrella | Re-exports. Needs selective re-export strategy once API surface grows. |
| `vm-multiscale` | Missing | New crate. Pyramid + edge combinator. |
| `vm-shape` | Solid | LSD, Bookstein + Fitzgibbon conic fitting, RANSAC. 26 tests, benchmarks. |
| `vm-match` | Solid | EdgeModel, chamfer-based coarse grid search, ICP refinement, normal-coherence scorer. 14 unit tests + 1 doctest, benchmarked. |
| `vm-segment` | Solid | Otsu/adaptive thresholding, CCL (union-find), watershed, edgel region growing. 24 unit tests + 4 doctests, benchmarked. |
| `vm-python` | Solid (type-checks) | PyO3 0.22 bindings: PyEdgeDetector, PyLsdDetector, PyConicFitter, PyRigidMatcher. Builds via maturin. |

---

## 3. Architectural Decisions

### AD-1: Feature area priority — parallel build on shared infrastructure

All three new feature families (LSD, ellipse fitting, directed-edge matching) proceed in
parallel across milestones. The pattern is: build the shared geometric infrastructure
first (Milestone 2), then land all three detectors in Milestone 3, then harden and add
matching in Milestone 4. No feature area blocks another; they share the contour graph and
multi-scale edge stack.

### AD-2: Segmentation scope

All four sub-domains are in scope:
1. Binary thresholding (Otsu global, adaptive local)
2. Connected-component labeling (CCL) on binary masks
3. Watershed / gradient-flow region splitting
4. Edgel-based region growing from the contour graph

Segmentation lives in `vm-segment`. CCL is a prerequisite for watershed and region
growing, so it lands first within that crate.

### AD-3: Object matching — rigid first, generalize later

Initial `vm-match` targets translation + rotation (rigid) alignment only. The design must
not foreclose similarity (+ scale) or affine generalization; specifically, the distance
and scoring functions must be decoupled from the transformation model so the transformation
type can be swapped without rewriting the scorer. Affine/projective generalization is
Milestone 5 (P4).

### AD-4: API design — config struct + reusable detector

New crates follow the established pattern:
```rust
// Config is cheap to clone, all fields pub.
let cfg = LsdConfig { sigma: 1.2, nfa_threshold: 1e-3, ..LsdConfig::default() };
// Detector owns scratch buffers; reused across calls.
let mut det = LsdDetector::new();
let segments: Vec<LineSegment2f> = det.detect_u8(&img.as_view(), &cfg);
```

No builder pattern, no trait-based polymorphism at the detector level. Composability is
achieved by passing `&[Edgel]` or `&ContourGraph` between stages, not by nesting
trait objects.

### AD-5: Performance target

**< 100 ms** per 1280x1024 frame on a single x86-64 core for the full pipeline
(multi-scale edges + shape detection + one-shot match query). Rayon is permitted for
shape fitting and matching stages. Benchmarks must be present for every new hot path.
The existing `cargo bench -p vm-laser --bench extract` baseline
(row scan: ~2 ms on a 1280x512 image) is the reference for the laser path.

### AD-6: Deployment targets — pure Rust binary + PyO3 bindings

No `no_std` requirement. The public types in `vm-core`, `vm-edge`, `vm-contour`,
`vm-shape`, and `vm-match` must be **lifetime-free** (no `'a` in public struct fields)
so PyO3 can derive `#[pyclass]` without fighting the borrow checker. Concretely:
- `Edgel`, `LineSegment2f`, `Ellipse2f`, `EdgeMatchResult` own their data (no borrows).
- `ImageView<'_, T>` is an internal/call-site type only; Python receives a copy.

A `vm-python` crate wrapping these types lands in Milestone 4.

### AD-7: Multi-scale edge detection — new `vm-multiscale` crate

`vm-pyr` and `vm-edge` remain separate. `vm-multiscale` combines them:
```rust
pub struct MultiScaleEdgeDetector { pyr: PyramidF32, det: Edge2DDetector, ... }
impl MultiScaleEdgeDetector {
    pub fn detect_u8(&mut self, img: &ImageView<'_, u8>, cfg: &MultiScaleConfig) -> Vec<Edgel>;
}
```
Each level's edgels are mapped back to level-0 coordinates with the correct scale factor
before being returned as a single merged `Vec<Edgel>`. The caller never sees pyramid
internals.

### AD-8: Error handling — extend `vm-core::Error`

`vm-core::Error` gains new variants as needed:
```rust
pub enum Error {
    SizeMismatch { expected: usize, actual: usize },
    OutOfBounds,
    InvalidStride,
    InsufficientData { need: usize, got: usize },   // NEW — fitting/RANSAC
    Degenerate(&'static str),                        // NEW — singular matrix, etc.
    InvalidConfig(&'static str),                     // NEW — bad parameter combination
}
```
Fallible public functions return `Result<T, vm_core::Error>`. Internal helpers may still
use `Option<T>` or `assert!` for programmer-error invariants.

---

## 4. Milestone Plan

### Milestone 1 — Foundation Hardening (P0/P1)
**Goal:** Close correctness gaps and missing infrastructure before building new features.
Everything currently shipping should be production-quality before the next layer goes on.

**Success criteria:**
- `vm-core::Error` extended; all public fallible functions return `Result`.
- `vm-contour::GraphEdge` carries curvature and tangent data.
- `vm-morph` supports parameterized structuring elements.
- All workspace tests pass, zero clippy warnings.

**Crates modified:** `vm-core`, `vm-contour`, `vm-morph`

**Tasks:**

- [x] **vm-core: Extend `Error` enum**
  Added `InsufficientData`, `Degenerate`, `InvalidConfig` variants (AD-8).
  File: `crates/vm-core/src/error.rs`

- [x] **vm-core: Add geometry types via nalgebra**
  Added `Rect2f`, `Angle`/`wrap_angle` in `geom.rs`.
  nalgebra type aliases: `Isometry2f`, `Similarity2f`, `Affine2f`, `Projective2f`.
  Conversion helpers: `to_na_point`, `from_na_point`, `to_na_vec`, `from_na_vec`.
  nalgebra added as workspace dependency.
  File: `crates/vm-core/src/geom.rs`

- [x] **vm-contour: Add curvature, tangent, arc-length to `GraphEdge`**
  `tangents`, `curvatures`, `arc_params` fields on `GraphEdge` (all `Option<Vec<_>>`).
  `ContourBuildConfig::record_geometry` flag; `GraphEdge::compute_geometry()`;
  `ContourGraph::compute_all_geometry()`, `iter_edges_by_length()`,
  `filter_edges_min_length()`.
  File: `crates/vm-contour/src/graph.rs`, `crates/vm-contour/src/build.rs`

- [x] **vm-morph: Parameterized structuring elements**
  `StructuringElement` enum (`Square(usize)`, `Disk(usize)`).
  `erode_binary_u8`, `dilate_binary_u8`, `open_binary_u8`, `close_binary_u8`.
  Backward-compatible `erode3x3_binary_u8` etc. wrappers retained.
  File: `crates/vm-morph/src/se.rs`, `crates/vm-morph/src/lib.rs`

- [x] **vm-morph: Distance transform (chamfer 3-4-5)**
  `chamfer_distance_u8` returning `Image<f32>` in pixel units.
  File: `crates/vm-morph/src/distance.rs`

- [x] **vm-morph: Binary thinning (Zhang-Suen)**
  `thin_binary_u8` iterative skeletonization.
  File: `crates/vm-morph/src/thin.rs`

- [x] **Tests for all M1 items** — 59 unit tests + 2 doctests, zero clippy warnings.

**Dependencies:** None (this milestone has no external prerequisites).

---

### Milestone 2 — Multi-Scale Edge Infrastructure (P1)
**Goal:** Introduce `vm-multiscale`, giving shape detectors and matchers a single call
site for scale-invariant edgel extraction.

**Success criteria:**
- `MultiScaleEdgeDetector::detect_u8` returns merged edgels in level-0 coordinates.
- `ScaleAnnotatedEdgel` carries `scale: f32` alongside position and normal.
- Benchmark shows < 10 ms for 1280x1024 at 3 scales.
- `vm-gallery` extended with a `multiscale_edges` subcommand.

**Crates created:** `vm-multiscale`
**Crates modified:** `vision-metrology` (re-exports), `vm-gallery`

**Tasks:**

- [x] **Create `crates/vm-multiscale/Cargo.toml`**
  Deps: `vm-core`, `vm-pyr`, `vm-edge`. Optional `rayon` feature.

- [x] **Define `ScaleAnnotatedEdgel`**
  Flat struct: `p`, `n`, `strength`, `scale`, `level`, `idx` (all level-0 coords).
  File: `crates/vm-multiscale/src/edgel.rs`

- [x] **Define `MultiScaleConfig`**
  `num_levels` (total including level-0), `base_sigma`, `edge: Edge2DConfig`,
  `merge_duplicates`. Thresholds auto-scaled per level (÷ 2^l).
  File: `crates/vm-multiscale/src/config.rs`

- [x] **Implement `MultiScaleEdgeDetector`**
  Owns `PyramidF32`, `Edge2DDetector`, dedup scratch `cell_used: Vec<bool>`.
  `detect_u8` / `detect_u16` / `detect_f32` methods.
  File: `crates/vm-multiscale/src/detector.rs`

- [x] **Deduplication** — keep finest-scale (level-0 first) per level-0 pixel cell.

- [x] **Benchmarks** — `multiscale_detect_u8_1280x1024_3levels` and `_1level`.
  File: `crates/vm-multiscale/benches/detect.rs`

- [x] **Update `vision-metrology/src/lib.rs`** — `pub use vm_multiscale::*`

- [ ] **`vm-gallery` `multiscale_edges` subcommand** — deferred to M5 polish pass.

**Dependencies:** Milestone 1 (affine types, extended Error).

---

### Milestone 3 — Shape Detection: LSD and Conic Fitting (P1)
**Goal:** Ship a production LSD implementation and a RANSAC-validated ellipse/conic fitter
in a new `vm-shape` crate. Both work on `&[Edgel]` or `&[ScaleAnnotatedEdgel]`.

**Success criteria:**
- LSD detects lines on synthetic binary-step images with endpoint error < 0.5 px.
- Ellipse fitter recovers axis parameters within 0.3% on blurred synthetic circles.
- Full pipeline (multi-scale edges + LSD + ellipse) runs < 40 ms on 1280x1024.
- `vm-gallery` extended with `lsd` and `ellipse` subcommands.

**Crates created:** `vm-shape`
**Crates modified:** `vm-gallery`, `vision-metrology`

**Tasks:**

#### LSD (Line Segment Detector)

- [x] **Define `LineSegment2f`**
  `p1`, `p2`, `normal`, `width`, `nfa`, `length`, `angle` fields.
  File: `crates/vm-shape/src/lsd.rs`

- [x] **Define `LsdConfig`**
  `scale=0.8`, `sigma_scale=0.6`, `ang_th=22.5°`, `log_eps=0.0`, `density_th=0.7`,
  `n_bins=1024`, `min_length=3.0`.
  File: `crates/vm-shape/src/lsd.rs`

- [x] **Implement gradient angle image**
  Scharr gradients; level-line angle `θ = atan2(-gx, gy)` normalized to `(-π/2, π/2]`.
  Bucket pseudo-sort by magnitude into `n_bins` bins.

- [x] **Implement region growing**
  8-connected BFS from strongest unused pixel; angular coherence guard.
  All scratch buffers (`region`, `queue`, `used`, `buckets`) owned by `LsdDetector`.

- [x] **Implement line fitting on a region**
  Weighted inertia tensor; principal axis via half-angle formula
  `θ = 0.5 * atan2(2*ixy, ixx-iyy)` (robust to degenerate axis-aligned cases).

- [x] **Implement NFA validation**
  `log10(NFA) = log10(N_T) + log10_binomial(n,k) + k*log10(p) + (n-k)*log10(1-p)`.
  Level-line probability `p = 2*ang_th/π` (half-circle range).
  `libm::lgamma` for the log-binomial. Accept when `log10(NFA) < log_eps`.
  File: `crates/vm-shape/src/nfa.rs`

- [x] **Implement `LsdDetector`**
  Scratch: `buf`, `gx`, `gy`, `mag`, `ang`, `buckets`, `used`, `region`, `queue`.
  Public: `detect` (u8), `detect_f32` returning `Vec<LineSegment2f>`.
  File: `crates/vm-shape/src/lsd.rs`

- [x] **LSD tests** — horizontal/vertical step edge detection, short-segment rejection,
  noise image false-detection control, f32/u8 consistency, NFA sign check.

- [x] **Benchmark `lsd_detect_u8_1280x1024`** and `lsd_detect_u8_512x512`.
  File: `crates/vm-shape/benches/detect.rs`

#### Ellipse and Conic Fitting

- [x] **Define `Conic2f` and `Ellipse2f`**
  `Conic2f { coeffs: [f32; 6] }` with `eval`, `discriminant`, `is_ellipse`, `grad_norm`, `to_ellipse`.
  `Ellipse2f { center, semi_axes, angle }` with `to_conic`, `from_conic`, `point_at`, `contains`.
  File: `crates/vm-shape/src/conic.rs`

- [x] **Define `ConicFitConfig`**
  `use_bookstein=true`, `ransac_iters=0`, `inlier_tol=1.0`, `min_inliers=5`, `rng_seed=42`.
  File: `crates/vm-shape/src/fitter.rs`

- [x] **Implement Direct Least Squares (Fitzgibbon et al.)**
  Normalise coordinates (centroid + RMS scale). Schur decomposition of
  `M = C11_inv * (S11 - S12 S22_inv S21)` for real eigenvalues; eigenvector
  recovered via SVD null-space of `(M - λI)`. Rescale to `a1^T C11 a1 = 1`.
  File: `crates/vm-shape/src/fit_conic.rs`

- [x] **Implement Bookstein fit**
  Coordinate-normalised scatter matrix; smallest eigenvector via `SymmetricEigen`
  with explicit minimum-eigenvalue search (nalgebra does not guarantee sort order).
  File: `crates/vm-shape/src/fit_conic.rs`

- [x] **Implement RANSAC wrapper**
  Seeded `Lcg` PRNG; sample-5 loop; Sampson distance `|F(p)|/‖∇F(p)‖` inlier metric;
  re-fit on best inlier set. `inlier_scratch: Vec<usize>` reused across calls.
  File: `crates/vm-shape/src/ransac.rs`

- [x] **Implement `ConicFitter` detector struct**
  `inlier_scratch` reused across RANSAC calls. `fit` and `fit_ellipse_ransac` methods.
  File: `crates/vm-shape/src/fitter.rs`

- [x] **Conic fitting tests** — circle/ellipse recovery, RANSAC with outliers,
  insufficient-data error, non-ellipse rejection, noisy-data fit, fitter reuse.

- [x] **Benchmark `conic_ransac_1000pts`** and `conic_direct_bookstein_100pts`.
  File: `crates/vm-shape/benches/detect.rs`

**Dependencies:** Milestone 2 (MultiScaleEdgeDetector for integration), Milestone 1 (Error variants, contour geometry).

---

### Milestone 4 — Segmentation and Directed-Edge Matching (P1/P2) ✅ COMPLETE
**Goal:** Ship `vm-segment` with all four segmentation modes and `vm-match` with rigid
directed-edge matching. Add `vm-python` PyO3 bindings over the stable API.

**Success criteria:** All met.
- Otsu threshold produces correct segmentation on synthetic bimodal histogram. ✅
- CCL runs in < 5 ms on 1280x1024 binary image (union-find path compression). ✅
- Rigid edgel-match locates a translated/rotated rectangular model in a synthetic scene. ✅
- Python bindings expose `Edge2DDetector`, `LsdDetector`, `ConicFitter`, and
  `RigidEdgeMatcher` with numpy array I/O. ✅ (type-checks cleanly; built via maturin)

**Crates created:** `vm-segment`, `vm-match`, `vm-python`
**Crates modified:** `vision-metrology` (re-exports vm-match, vm-segment)

#### vm-segment

- [x] **Binary thresholding**
  `otsu_threshold_u8`: histogram scan (O(256) + O(N)), allocation-free.
  `adaptive_threshold_u8`: 2D integral-image local mean, clamp border.
  File: `crates/vm-segment/src/threshold.rs`

- [x] **Connected-component labeling**
  Two-pass Rosenfeld-Pfaltz union-find with path-halving + union-by-rank.
  `CcLabel { label_map: Image<u32>, num_labels: u32 }`. C4 and C8 connectivity.
  File: `crates/vm-segment/src/ccl.rs`

- [x] **Component statistics**
  `ComponentStats { label, pixel_count, bbox: Rect2f, centroid: Point2f }`.
  `component_stats(cl: &CcLabel, min_area: u32) -> Vec<ComponentStats>`.
  File: `crates/vm-segment/src/ccl.rs`

- [x] **Watershed segmentation**
  Beucher-Meyer priority-queue flood-fill. Seeds pre-labelled; only neighbours
  pushed into heap. Output: `Image<i32>` (-1 = boundary, ≥0 = region label).
  File: `crates/vm-segment/src/watershed.rs`

- [x] **Edgel-based region growing**
  Rasterises `ContourGraph` edgels → chamfer mask → BFS flood fill on gap-filled
  binary mask. Small regions below `min_area` relabelled to -1.
  File: `crates/vm-segment/src/region.rs`

- [x] **Segmentation tests and benchmarks** — 24 unit tests + 4 doctests.
  Benchmarks: `otsu_1280x1024`, `adaptive_threshold_1280x1024`, `ccl_1280x1024`,
  `watershed_1280x1024_4seeds`.
  File: `crates/vm-segment/benches/segment.rs`

#### vm-match

- [x] **`EdgeModel`**
  Centroid-subtracted edgels in model-local coords. Pre-computed chamfer map with
  configurable margin. `map_offset` for model-local → map-pixel conversion.
  File: `crates/vm-match/src/model.rs`

- [x] **`RigidMatchConfig`**
  `angle_range`, `angle_step`, `position_search: Rect2f`, `chamfer_threshold`,
  `min_score`, `refine_icp`, `top_k`, `resolution_factor`.
  File: `crates/vm-match/src/rigid.rs`

- [x] **Chamfer-based coarse grid search**
  Iterates over angle × position grid. Per-candidate: rotate model edgels, look up
  scene chamfer map, accumulate mean chamfer score. Top-K heap with eviction.
  File: `crates/vm-match/src/matcher.rs`

- [x] **Normal-coherence Hausdorff scorer**
  Mean dot product of rotated model normals vs. nearest scene edgel normal.
  Rejects candidates with flipped polarity (score < 0).
  File: `crates/vm-match/src/score.rs`

- [x] **ICP refinement**
  Closed-form 2D: cross-covariance H → `θ = atan2(H01-H10, H00+H11)`.
  Cumulative transform tracking. Max 20 iterations. No SVD needed.
  File: `crates/vm-match/src/icp.rs`

- [x] **`RigidMatchResult`** — `transform: Isometry2f`, `score`, `inlier_count`,
  `chamfer_mean`.

- [x] **`RigidEdgeMatcher`** — owns scene chamfer scratch buffer; reused across calls.
  File: `crates/vm-match/src/matcher.rs`

- [x] **Matching tests** — 14 unit tests + 1 doctest.
  Synthetic rectangle localization, normal-coherence rejection, ICP convergence.

- [x] **Benchmark `rigid_match_1280x1024_20edgel_model`**
  File: `crates/vm-match/benches/match.rs`

#### vm-python (PyO3)

- [x] **Crate scaffold** — `crate-type = ["cdylib", "rlib"]`, pyo3 0.22 + numpy 0.22.
  Workspace lint override: `unsafe_op_in_unsafe_fn = "allow"` for pyo3 macro compat.
  Clippy override: `useless_conversion = "allow"` for pyo3 `PyResult` type alias.
  File: `crates/vm-python/Cargo.toml`, `crates/vm-python/src/lib.rs`

- [x] **Numpy array I/O** — `image_from_numpy_u8` (copies to owned `Image<u8>`).
  GIL held during detect; zero-copy view with copy-on-detect strategy documented.
  File: `crates/vm-python/src/convert.rs`

- [x] **`PyEdgeDetector`** — wraps `Edge2DDetector` + `Edge2DConfig`. Reuses scratch
  buffers across calls. Returns list of dicts `{x, y, nx, ny, strength}`.
  File: `crates/vm-python/src/detector.rs`

- [x] **`PyLsdDetector`** — wraps `LsdDetector` + `LsdConfig`. Returns list of dicts
  `{x1, y1, x2, y2, width, nfa, angle}`.
  File: `crates/vm-python/src/shape.rs`

- [x] **`PyConicFitter`** — wraps `ConicFitter` + `ConicFitConfig`. `fit_ellipse(pts)`
  accepts `(N, 2) float32` array; returns dict `{cx, cy, a, b, angle}` or `None`.
  File: `crates/vm-python/src/shape.rs`

- [x] **`PyRigidMatcher`** — full pipeline: edge detect scene + chamfer grid search +
  ICP. Accepts model edgels as `(N, 5) float32` + scene as `(H, W) uint8`.
  Returns dict `{tx, ty, angle, score}` or `None`.
  File: `crates/vm-python/src/match_py.rs`

- [x] **Python smoke tests** — `tests/test_bindings.py`.

**Design notes for M4:**
- OQ-3 resolved: scene chamfer map always at full resolution (configurable via
  `resolution_factor` field in `RigidMatchConfig`, currently fixed at 1.0).
- OQ-4 resolved: GIL held; caller copies data via `image_from_numpy_u8` (simplest
  correct strategy; documented in `convert.rs`).
- PyO3 0.22 + Python 3.14: requires `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1`.
- `cargo check -p vm-python` succeeds; `cargo build` produces expected linker error
  (extension module must be built with `maturin develop` or `maturin build`).

**Dependencies:** Milestone 3 (LSD, conic fitting) for vm-python bindings; Milestone 1 (distance transform) for vm-match chamfer map.

---

### Milestone 5 — Hardening, Generalization, and Documentation (P2/P3/P4)
**Goal:** Production hardening of all new crates, similarity/affine matching,
documentation coverage, and a comprehensive integration example.

**Success criteria:**
- All workspace benchmarks establish regression baselines in `bench_baselines.md`.
- Similarity (scale + R|t) matching added to `vm-match`.
- All public types and functions have `///` doc comments.
- Integration example `cargo run -p vision-metrology --example measure_circles` runs
  end-to-end on a synthetic PCB image.

**Tasks:**

- [ ] **Similarity and affine matching in `vm-match`**
  Extend `RigidMatchConfig` to `MatchConfig` with `transform_type: TransformType`.
  `TransformType::Similarity` adds a scale search dimension.

- [ ] **Multi-object matching**
  `RigidEdgeMatcher::match_all_instances` returns `Vec<RigidMatchResult>` with
  non-maximum suppression on overlapping detections.

- [ ] **vm-contour: contour smoothing**
  Add `smooth_polyline(points: &[Point2f], sigma: f32) -> Vec<Point2f>` using
  1D Gaussian convolution along arc length. Used by shape detectors to reduce
  digitization noise before fitting.

- [ ] **vm-edge: Expose gradient buffers from `Edge2DDetector`**
  Add `detect_u8_with_gradients` returning `(Vec<Edgel>, GradientView)` where
  `GradientView` wraps `&gx`, `&gy`, `&mag` without allocation. Required by LSD
  for angle image computation without recomputing gradients.

- [ ] **Performance audit and SIMD annotation**
  Profile full pipeline on 1280x1024. For any stage exceeding its budget, annotate
  hot loops with `#[target_feature(enable = "avx2")]` or add explicit SIMD via
  `std::simd` (Rust portable SIMD, stable as of Rust 1.78).

- [ ] **Regression benchmark baseline document**
  `bench_baselines.md` at repo root. Auto-generated by `cargo bench` output
  capture script.

- [ ] **Integration example: `measure_circles`**
  Full pipeline: load image, multi-scale edges, LSD + ellipse detection, fit circle
  radii, output JSON measurement report. Documented with inline comments.
  File: `crates/vision-metrology/examples/measure_circles.rs`

- [ ] **Public API documentation**
  Doc comments on all `pub` items in `vm-core`, `vm-edge`, `vm-contour`, `vm-shape`,
  `vm-match`, `vm-segment`. `cargo doc --no-deps --open` must produce complete docs.

- [ ] **README update**
  Update root `README.md` with pipeline diagram, crate dependency graph, and
  quick-start code snippets for LSD, ellipse fitting, and matching.

**Dependencies:** Milestones 3 and 4 complete.

---

## 5. Prioritized Backlog

The backlog is ordered by priority class (P0 first) then by milestone dependency. Items
within the same priority class are ordered by estimated impact / unblocking value.

| # | Priority | Item | Milestone | Crate |
|---|---|---|---|---|
| 1 | P0 | ~~Extend `vm-core::Error` with new variants~~ ✅ | M1 | vm-core |
| 2 | P0 | ~~Add geometry types + nalgebra aliases to vm-core~~ ✅ | M1 | vm-core |
| 3 | P1 | ~~Curvature + tangent on `GraphEdge`~~ ✅ | M1 | vm-contour |
| 4 | P1 | ~~Arc-length parameterization on `GraphEdge`~~ ✅ | M1 | vm-contour |
| 5 | P1 | ~~`vm-morph` parameterized structuring elements~~ ✅ | M1 | vm-morph |
| 6 | P1 | ~~Chamfer distance transform in `vm-morph`~~ ✅ | M1 | vm-morph |
| 7 | P1 | ~~`vm-multiscale` crate + `MultiScaleEdgeDetector`~~ ✅ | M2 | vm-multiscale |
| 8 | P1 | ~~LSD detector (`LsdDetector`, `LineSegment2f`)~~ ✅ | M3 | vm-shape |
| 9 | P1 | ~~Direct Least Squares conic fitter (Fitzgibbon + Bookstein)~~ ✅ | M3 | vm-shape |
| 10 | P1 | ~~RANSAC wrapper for conic fitting~~ ✅ | M3 | vm-shape |
| 11 | P1 | `ConicFitter::detect_ellipses_in_graph` | M5 | vm-shape |
| 12 | P1 | ~~Otsu + adaptive thresholding~~ ✅ | M4 | vm-segment |
| 13 | P1 | ~~Connected-component labeling (union-find)~~ ✅ | M4 | vm-segment |
| 14 | P1 | ~~`EdgeModel` + chamfer map construction~~ ✅ | M4 | vm-match |
| 15 | P1 | ~~Chamfer-based coarse rigid match~~ ✅ | M4 | vm-match |
| 16 | P1 | ~~ICP refinement~~ ✅ | M4 | vm-match |
| 17 | P2 | ~~Watershed segmentation~~ ✅ | M4 | vm-segment |
| 18 | P2 | ~~Edgel-based region growing~~ ✅ | M4 | vm-segment |
| 19 | P2 | ~~Normal-coherence Hausdorff scorer~~ ✅ | M4 | vm-match |
| 20 | P2 | ~~PyO3 bindings (`vm-python`)~~ ✅ | M4 | vm-python |
| 21 | P2 | Expose gradient buffers from `Edge2DDetector` | M5 | vm-edge |
| 22 | P2 | Contour polyline smoothing | M5 | vm-contour |
| 23 | P2 | Similarity + affine matching | M5 | vm-match |
| 24 | P2 | Multi-object NMS matching | M5 | vm-match |
| 25 | P2 | Performance audit + SIMD annotation | M5 | all |
| 26 | P3 | `measure_circles` integration example | M5 | vision-metrology |
| 27 | P3 | Full public API documentation | M5 | all |
| 28 | P3 | Regression benchmark baselines document | M5 | all |
| 29 | P3 | README pipeline diagram and quick-start | M5 | — |
| 30 | P3 | ~~`vm-morph` thinning / skeletonization~~ ✅ | M1 | vm-morph |
| 31 | P3 | ~~`Angle` newtype and `angle_diff` utility~~ ✅ | M1 | vm-core |
| 32 | P4 | C FFI layer (`vm-ffi` crate, `#[repr(C)]` types) | post-M5 | vm-ffi |
| 33 | P4 | WASM / `wasm-bindgen` target | post-M5 | vm-wasm |
| 34 | P4 | Phase correlation / frequency-domain matching | post-M5 | vm-freq |
| 35 | P4 | GPU offload (wgpu compute shaders) | post-M5 | vm-gpu |

---

## 6. API Design Principles

The following principles are binding for all new crates and apply retroactively to
modifications of existing crates.

### P-1: Config struct + reusable detector

Every algorithm is exposed as a pair:
```rust
pub struct FooConfig { pub field: Type, ... }  // all fields pub, Default derived
impl Default for FooConfig { ... }

pub struct FooDetector { /* scratch buffers, private */ }
impl FooDetector {
    pub fn new() -> Self;
    pub fn detect_u8(&mut self, img: &ImageView<'_, u8>, cfg: &FooConfig) -> Vec<FooResult>;
    pub fn detect_f32(&mut self, img: &ImageView<'_, f32>, cfg: &FooConfig) -> Vec<FooResult>;
}
```
No builder pattern. No trait objects at the detector level.

### P-2: Allocation-free per-scan hot paths

No `Vec` allocation inside per-row or per-scan loops. All scratch `Vec<T>` fields on
detector structs must be resized with `resize` / `clear` + `extend`, never replaced with
`= Vec::new()` inside a detect call. Per-call allocation (e.g., output `Vec<Edgel>`) is
acceptable only in the final collection step.

### P-3: Lifetime-free public output types

All structs returned by `pub` functions (and stored in `Vec` outputs) must own their
data. No `'a` lifetime parameters in public struct definitions. This is required for
PyO3 compatibility (AD-6).

```rust
// Good
pub struct LineSegment2f { pub p1: Point2f, pub p2: Point2f, ... }

// Bad — breaks PyO3
pub struct LineSegment<'a> { pub points: &'a [Point2f], ... }
```

### P-4: `ImageView<'_, T>` stays internal to call sites

`ImageView` is a call-site borrow type. It must not appear in public struct fields or
return types. Python callers copy the image data via `Image<T>`, which is then lent as
a view for the duration of a detect call.

### P-5: Fallible public API returns `Result<T, vm_core::Error>`

Every public function that can fail (fitting, threshold computation on empty image,
invalid config combination) returns `Result<T, vm_core::Error>`. Internal helpers may
use `Option<T>` or `assert!` for programmer-error invariants only (not for domain errors
that a caller should handle gracefully).

### P-6: Default border mode is `Clamp`

All new detectors and convolution helpers default to `BorderMode::Clamp`. Any function
that silently uses a different mode must document it explicitly in its doc comment.

### P-7: Pixel-center coordinates everywhere

Integer coordinate `i` refers to the **center** of pixel `i`. Subpixel positions are
`i as f32 + delta` where `delta ∈ [-0.5, 0.5]`. No code shall use `i + 0.5` as a
pixel-center offset. Any violation is a blocker.

### P-8: Tests use deterministic synthetic fixtures

Tests must not depend on external image files, network access, or system entropy.
All test images are generated inline (step edges, circles, rectangles at known
coordinates). Every test comment must state the expected geometry in plain language
before the assertion so the intent is self-documenting.

### P-9: Benchmarks for every new hot function

Any function whose inner loop operates on more than one image row must have a Criterion
benchmark at a representative size (minimum 512x512; 1280x1024 preferred). Benchmarks
live in `crates/<name>/benches/`. No benchmark may allocate in its measured closure
(use `black_box` on a pre-built detector and pre-built image).

### P-10: Unsafe requires `// SAFETY:` comment

Every `unsafe` block must be preceded immediately by a `// SAFETY:` comment explaining
the invariant that makes the block sound. Unsafe blocks lacking this comment are a
blocker in code review.

### P-11: `vm-core::Error` is the only error type

Do not define per-crate error enums. Extend `vm-core::Error` when a new variant is
needed. This keeps the `Result` chain flat and makes the PyO3 error mapping trivial.

---

## 7. Open Questions

The following questions were not resolved in the initial design session and will need
answers before the relevant milestone can be finalized.

### OQ-1: NFA computation precision (RESOLVED)

Decision: `lgamma`-based log-binomial (`libm::lgamma`) throughout. For all region
sizes tested (n up to a few thousand), the approximation is numerically accurate to
machine precision relative to exact factorials for n ≤ ~170. Single-term NFA
(mode of the binomial, not tail sum) is used; this is standard practice for LSD
and gives the correct false-alarm guarantee in the a-contrario sense.

Correction: level-line angles span `(-π/2, π/2]` (half-circle), so the alignment
probability is `p = 2*ang_th/π`, not `ang_th/π` as in the original LSD paper
(which uses full gradient directions). This was found and fixed during M3.

### OQ-2: Conic fitting constraint choice (RESOLVED)

Decision: expose both. `ConicFitConfig::use_bookstein=true` (default) selects
Bookstein `‖c‖=1` (similarity-invariant, works for any conic);
`use_bookstein=false` selects Fitzgibbon `4AC-B²=1` (guarantees ellipse output).
RANSAC always uses Fitzgibbon for its sample-5 hypotheses. Both methods apply
coordinate normalisation (translate to centroid, scale by RMS distance) before
forming the scatter matrix for numerical stability at large pixel coordinates.

### OQ-3: Chamfer map resolution for matching (RESOLVED — M4)

Decision: full image resolution always. `RigidMatchConfig::resolution_factor` field
reserved (currently fixed at 1.0) to allow coarse-map optimization in M5 without
breaking the API. ICP refinement compensates for coarse grid quantization.

### OQ-4: PyO3 numpy type for `Image<u8>` (RESOLVED — M4)

Decision: GIL held during detect; `image_from_numpy_u8` copies the numpy array data
into an owned `Image<u8>`. Simpler, correct, and safe. Callers needing true
parallelism should release the GIL in Python and batch-copy before calling.
Documented in `crates/vm-python/src/convert.rs`.

### OQ-5: Edgel normal direction convention in vm-match (Milestone 5)

Current `Edgel.n` points dark-to-bright (increasing intensity). The directed-edge
matching score uses normal coherence to reject mirrored false-positives. If the
model is built from a CAD boundary (no photometric context), which convention should
model normals follow — outward from the part, or always dark-to-bright?
This affects how the model is constructed and what the scoring function computes.

### OQ-6: Watershed seed generation (RESOLVED — M4)

Decision: user-supplied seed list only (`&[(usize, usize)]`). Keeps `watershed`
composable. Helpers for auto-seed-from-CCL or auto-seed-from-minima are deferred
to M5 as separate utility functions in vm-segment.

### OQ-7: Python bindings — manylinux wheels vs. source-only (Milestone 4 — vm-python)

Will `vm-python` ship pre-built wheels on PyPI (requires manylinux CI), or is
source-only `pip install` (requires Rust toolchain on the user machine) acceptable?
This drives CI infrastructure decisions and must be decided before publishing.

---

*This roadmap is a living document. Update the status table in Section 2 and check off
backlog items in Section 5 as work lands. Revisit open questions at the start of each
milestone.*
