# vision-metrology — Development Roadmap

**Last updated:** 2026-02-28
**Status:** Active development — Milestone 2 complete

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
| `vm-shape` | Missing | New crate. LSD, ellipse/conic detection. |
| `vm-match` | Missing | New crate. Directed-edge matching, rigid/similarity alignment. |
| `vm-segment` | Missing | New crate. Thresholding, CCL, watershed, edgel region growing. |
| `vm-python` | Missing | New crate. PyO3 bindings over the stable public API. |

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

- [ ] **Define `LineSegment2f`**
  ```rust
  pub struct LineSegment2f {
      pub p1: Point2f,
      pub p2: Point2f,
      pub normal: Vec2f,    // unit normal to segment, consistent with edgel normals
      pub width: f32,       // estimated support region width in pixels
      pub nfa: f32,         // log10 of Number of False Alarms (lower = stronger)
      pub length: f32,      // Euclidean distance p1 to p2
  }
  ```
  File: `crates/vm-shape/src/line.rs`

- [ ] **Define `LsdConfig`**
  ```rust
  pub struct LsdConfig {
      pub scale: f32,           // image downscale before detection (default 0.8)
      pub sigma_scale: f32,     // sigma = sigma_scale / scale
      pub ang_th: f32,          // gradient angle tolerance in degrees (default 22.5)
      pub log_eps: f32,         // log NFA threshold (default 0.0 = 1 false alarm)
      pub density_th: f32,      // minimum aligned point density in region
      pub n_bins: usize,        // angle quantization bins for ordering
      pub min_length: f32,      // minimum accepted segment length in pixels
  }
  ```
  File: `crates/vm-shape/src/lsd.rs`

- [ ] **Implement gradient angle image**
  Compute `angle[y][x] = atan2(gy, gx)` from `gx`/`gy` already available from
  `Edge2DDetector` internals, or recompute via Scharr on the input. Quantize into
  `n_bins` orientation bins for sorting.
  Note: expose `gx`/`gy`/`mag` buffers from `Edge2DDetector` via a new
  `detect_u8_with_gradients` method that also returns gradient views.

- [ ] **Implement region growing**
  Sort pixels by decreasing gradient magnitude. Seed from unused strong-gradient
  pixels. Grow by angular coherence (`|angle[nb] - angle[seed]| < ang_th`).
  Mark pixels as used. Store region as `Vec<usize>` (linear pixel indices).
  Scratch buffer owned by `LsdDetector`; cleared per call, not reallocated.

- [ ] **Implement line fitting on a region**
  Compute weighted centroid and 2x2 inertia tensor of region pixels weighted by
  gradient magnitude. Principal axis = eigenvector of minimum eigenvalue.
  Endpoints = projection of extreme region pixels onto principal axis.

- [ ] **Implement NFA validation**
  `NFA(n, k, p) = N_5 * C(n,k) * p^k * (1-p)^(n-k)` where `N_5 = W*H*max_pts^2/2`.
  Use log-domain arithmetic to avoid overflow. Accept segment if `log10(NFA) < log_eps`.
  File: `crates/vm-shape/src/nfa.rs`

- [ ] **Implement `LsdDetector`**
  Owns gradient scratch buffers, region scratch `Vec`, pixel-used bitmask.
  Public: `detect_u8`, `detect_f32` returning `Vec<LineSegment2f>`.
  File: `crates/vm-shape/src/lsd.rs`

- [ ] **LSD tests**
  - Horizontal/vertical/diagonal step edge: endpoint error < 0.5 px.
  - Short segment below `min_length` must be rejected.
  - NFA filter: noise-only image produces zero segments.

- [ ] **Benchmark `lsd_detect_u8_1280x1024`**

#### Ellipse and Conic Fitting

- [ ] **Define `Conic2f` and `Ellipse2f`**
  ```rust
  pub struct Conic2f { pub coeffs: [f32; 6] }  // Ax²+Bxy+Cy²+Dx+Ey+F=0
  pub struct Ellipse2f {
      pub center: Point2f,
      pub semi_major: f32,
      pub semi_minor: f32,
      pub angle: f32,     // radians, major axis to x-axis
  }
  impl Ellipse2f { pub fn from_conic(c: &Conic2f) -> Result<Self, Error>; }
  ```
  File: `crates/vm-shape/src/conic.rs`

- [ ] **Define `ConicFitConfig`**
  ```rust
  pub struct ConicFitConfig {
      pub method: FitMethod,          // DirectLeastSquares | Bookstein | RANSAC
      pub ransac_iterations: usize,   // default 200
      pub ransac_inlier_dist: f32,    // algebraic distance threshold
      pub min_inliers: usize,         // default 6 (minimum for ellipse)
  }
  ```

- [ ] **Implement Direct Least Squares (Fitzgibbon et al.)**
  Solve `min ‖Da‖` subject to `aᵀCa = 1` (ellipse constraint).
  Requires 6x6 generalized eigenvalue solve. Use iterative power method or
  a compact 6x6 solver (no external LAPACK dependency — implement in-crate).
  Return `Result<Conic2f, Error>` (fails if degenerate or not ellipse).
  File: `crates/vm-shape/src/fit_conic.rs`

- [ ] **Implement RANSAC wrapper**
  Sample 5 edgels (minimum for unique conic), fit DLS, score by algebraic distance
  inlier count. Return best fit over `ransac_iterations` trials.
  Store random state as `u64` seed in `ConicFitter` for reproducibility.
  File: `crates/vm-shape/src/ransac.rs`

- [ ] **Implement `ConicFitter` detector struct**
  Owns scratch buffers (design matrix, score scratch). Public `fit_edgels` and
  `detect_ellipses_in_graph` (operates on `&ContourGraph`, segments each edge,
  fits per arc).
  File: `crates/vm-shape/src/conic.rs`

- [ ] **Conic fitting tests**
  - Fit to 20 points on a known ellipse: recover parameters within 0.3%.
  - RANSAC with 5 outliers in 20 points: still converges.
  - Degenerate input (< 5 points): returns `Err(Error::InsufficientData{..})`.
  - Non-ellipse conic: `Ellipse2f::from_conic` returns `Err(Error::Degenerate)`.

- [ ] **Benchmark `conic_ransac_1000pts`**

**Dependencies:** Milestone 2 (MultiScaleEdgeDetector for integration), Milestone 1 (Error variants, contour geometry).

---

### Milestone 4 — Segmentation and Directed-Edge Matching (P1/P2)
**Goal:** Ship `vm-segment` with all four segmentation modes and `vm-match` with rigid
directed-edge matching. Add `vm-python` PyO3 bindings over the stable API.

**Success criteria:**
- Otsu threshold produces correct segmentation on synthetic bimodal histogram.
- CCL runs in < 5 ms on 1280x1024 binary image (union-find path compression).
- Rigid edgel-match locates a 20-edgel rectangular model in a synthetic scene with
  translation up to 50 px and rotation up to 30° within 50 ms.
- Python bindings expose `Edge2DDetector`, `LsdDetector`, `ConicFitter`, and
  `RigidEdgeMatcher` with numpy array I/O.

**Crates created:** `vm-segment`, `vm-match`, `vm-python`
**Crates modified:** `vm-gallery`, `vision-metrology`

#### vm-segment

- [ ] **Binary thresholding**
  ```rust
  pub fn otsu_threshold_u8(img: &ImageView<'_, u8>) -> u8;
  pub fn adaptive_threshold_u8(img: &ImageView<'_, u8>, cfg: &AdaptiveThreshConfig)
      -> Image<u8>;
  ```
  Otsu: histogram scan (O(256) + O(N)), allocation-free.
  Adaptive: local mean or Gaussian-weighted mean in a configurable window.
  File: `crates/vm-segment/src/threshold.rs`

- [ ] **Connected-component labeling**
  ```rust
  pub struct CcLabel { pub label_map: Image<u32>, pub num_labels: u32 }
  pub fn label_connected_components_u8(
      binary: &ImageView<'_, u8>,
      connectivity: Connectivity,     // reuse vm-contour type
  ) -> CcLabel;
  ```
  Two-pass union-find with path compression (Rosenfeld-Pfaltz style).
  Scratch `equiv` table owned by a `CclScratch` struct; reusable.
  File: `crates/vm-segment/src/ccl.rs`

- [ ] **Component statistics**
  ```rust
  pub struct ComponentStats {
      pub label: u32,
      pub pixel_count: u32,
      pub bbox: Rect2f,
      pub centroid: Point2f,
  }
  pub fn component_stats(cl: &CcLabel, min_area: u32) -> Vec<ComponentStats>;
  ```
  File: `crates/vm-segment/src/ccl.rs`

- [ ] **Watershed segmentation**
  Marker-based watershed on a gradient magnitude image.
  Input: gradient `Image<f32>`, seed markers `&[(usize, usize)]`.
  Output: label `Image<i32>` (-1 = boundary, ≥0 = region label).
  Uses priority queue (BinaryHeap) flood-fill from markers.
  File: `crates/vm-segment/src/watershed.rs`

- [ ] **Edgel-based region growing**
  Grow regions from seed pixels using `ContourGraph` edges as boundaries.
  Returns a label image where boundaries coincide with detected edges.
  Input: `&ContourGraph`, `&ImageView<'_, u8>`, `RegionGrowConfig`.
  File: `crates/vm-segment/src/region.rs`

- [ ] **Segmentation tests and benchmarks**
  - Otsu on known bimodal histogram: verify threshold at valley.
  - CCL: synthetic binary rectangle produces 1 foreground component.
  - CCL performance benchmark at 1280x1024.

#### vm-match

- [ ] **Define `EdgeModel`**
  ```rust
  pub struct EdgeModel {
      pub edgels: Vec<Edgel>,        // model edgels in model-local coords
      pub centroid: Point2f,
      pub chamfer_map: Image<f32>,   // pre-computed chamfer distance map
      pub chamfer_w: usize,
      pub chamfer_h: usize,
  }
  impl EdgeModel {
      pub fn from_edgels(edgels: Vec<Edgel>, map_margin: usize) -> Self;
  }
  ```
  File: `crates/vm-match/src/model.rs`

- [ ] **Define `RigidMatchConfig`**
  ```rust
  pub struct RigidMatchConfig {
      pub angle_range: (f32, f32),   // radians
      pub angle_step: f32,
      pub position_search: Rect2f,   // search region in image coords
      pub chamfer_threshold: f32,    // max distance to count as inlier
      pub min_score: f32,            // minimum inlier fraction
      pub refine_icp: bool,          // run ICP refinement on best candidate
  }
  ```
  File: `crates/vm-match/src/rigid.rs`

- [ ] **Implement chamfer-based coarse match**
  For each (angle, coarse position) in search grid:
  1. Rotate model edgel positions by angle.
  2. Translate to candidate position.
  3. Look up chamfer distance for each model edgel in scene chamfer map.
  4. Score = fraction of edgels with distance < `chamfer_threshold`.
  Store top-K candidates. Rayon parallel over angle steps.
  File: `crates/vm-match/src/rigid.rs`

- [ ] **Implement Hausdorff / directed-normal scoring**
  Secondary scorer for shortlisted candidates: measure forward + backward Hausdorff,
  weighted by normal coherence (dot product of model vs. nearest scene edgel normal).
  This rejects false positives where geometry matches but polarity is wrong.
  File: `crates/vm-match/src/score.rs`

- [ ] **Implement ICP refinement**
  Point-to-point ICP on inlier edgel positions. Max 20 iterations. Returns refined
  `SimilarityTransform2f` (scale fixed at 1 for rigid). SVD-free: use
  cross-covariance + Jacobi 2x2 solver.
  File: `crates/vm-match/src/icp.rs`

- [ ] **Define `RigidMatchResult`**
  ```rust
  pub struct RigidMatchResult {
      pub transform: SimilarityTransform2f,
      pub score: f32,             // inlier fraction after refinement
      pub inlier_count: usize,
      pub chamfer_mean: f32,      // mean chamfer distance of inliers
  }
  ```

- [ ] **Implement `RigidEdgeMatcher`**
  Owns scene chamfer map scratch buffer. `match_model` takes `&EdgeModel`,
  `&[Edgel]` (scene), `&RigidMatchConfig` and returns `Option<RigidMatchResult>`.
  File: `crates/vm-match/src/matcher.rs`

- [ ] **Matching tests**
  - Synthetic rectangle model: locate after 30px translation, 15° rotation.
  - Normal-coherence filter: mirror image of model must score poorly.
  - ICP: residual after refinement < 0.2 px for clean data.

- [ ] **Benchmark `rigid_match_1280x1024_20edgel_model`**

#### vm-python (PyO3)

- [ ] **Crate scaffold with PyO3 dependency**
  Feature-gate with `pyo3/extension-module`.
  File: `crates/vm-python/Cargo.toml`, `crates/vm-python/src/lib.rs`

- [ ] **Numpy array I/O helpers**
  `fn image_from_numpy_u8(array: &PyArray2<u8>) -> Image<u8>`
  `fn edgels_to_numpy(edgels: &[Edgel]) -> PyResult<PyObject>`
  File: `crates/vm-python/src/convert.rs`

- [ ] **Expose `Edge2DDetector` and `Edge2DConfig` as `PyEdgeDetector`**
  `#[pyclass]`, `#[pymethods]` with `detect_u8(img: &PyArray2<u8>) -> PyResult<PyObject>`.

- [ ] **Expose `LsdDetector`, `ConicFitter`, `RigidEdgeMatcher` as Py classes**

- [ ] **Python smoke tests**
  `tests/test_bindings.py`: import `vm_python`, run detector on a synthetic numpy array,
  assert non-empty result.

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
| 8 | P1 | LSD detector (`LsdDetector`, `LineSegment2f`) | M3 | vm-shape |
| 9 | P1 | Direct Least Squares conic fitter | M3 | vm-shape |
| 10 | P1 | RANSAC wrapper for conic fitting | M3 | vm-shape |
| 11 | P1 | `ConicFitter::detect_ellipses_in_graph` | M3 | vm-shape |
| 12 | P1 | Otsu + adaptive thresholding | M4 | vm-segment |
| 13 | P1 | Connected-component labeling (union-find) | M4 | vm-segment |
| 14 | P1 | `EdgeModel` + chamfer map construction | M4 | vm-match |
| 15 | P1 | Chamfer-based coarse rigid match | M4 | vm-match |
| 16 | P1 | ICP refinement | M4 | vm-match |
| 17 | P2 | Watershed segmentation | M4 | vm-segment |
| 18 | P2 | Edgel-based region growing | M4 | vm-segment |
| 19 | P2 | Normal-coherence Hausdorff scorer | M4 | vm-match |
| 20 | P2 | PyO3 bindings (`vm-python`) | M4 | vm-python |
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

### OQ-1: NFA computation precision (Milestone 3 — LSD)

The standard LSD NFA formula requires computing `C(n,k)` for potentially large `n`
(thousands of region pixels). Log-gamma approximation is standard but introduces
small numerical error. Decision needed: use `lgamma`-based approximation (fast,
slight error) or exact log-sum for `n < 1000` with fallback? This affects the
false-alarm rate guarantee.

### OQ-2: Conic fitting constraint choice (Milestone 3 — Ellipse)

The Fitzgibbon DLS method uses the algebraic constraint `4AC - B² = 1` which
guarantees an ellipse solution but is not invariant to similarity transforms.
The Bookstein normalization (`A + C = 1`) is similarity-invariant but does not
guarantee ellipse. Decision needed before implementing `fit_conic.rs`: which
constraint, or expose both as `FitMethod` enum variants?

### OQ-3: Chamfer map resolution for matching (Milestone 4 — vm-match)

The scene chamfer map is computed at full image resolution. For large images (2048x2048)
and many match candidates, a coarse map (half resolution) + fine-resolution ICP
refinement may be faster. Decision needed: fixed full-resolution, or configurable
resolution factor in `RigidMatchConfig`?

### OQ-4: PyO3 numpy type for `Image<u8>` (Milestone 4 — vm-python)

PyO3 + numpy can pass images as `PyReadonlyArray2<u8>` (zero-copy view) or as a
new `Image<u8>` copy. Zero-copy is faster but requires the GIL to be held during
the detect call. Decision needed: accept the GIL constraint (simpler) or require
callers to copy (safer for multi-threaded Python)?

### OQ-5: Edgel normal direction convention in vm-match (Milestone 4)

Current `Edgel.n` points dark-to-bright (increasing intensity). The directed-edge
matching score uses normal coherence to reject mirrored false-positives. If the
model is built from a CAD boundary (no photometric context), which convention should
model normals follow — outward from the part, or always dark-to-bright?
This affects how the model is constructed and what the scoring function computes.

### OQ-6: Watershed seed generation (Milestone 4 — vm-segment)

Watershed requires seed markers. Options: (a) user-supplied seed list, (b) automatic
seeds from local minima of gradient magnitude, (c) seeds from CCL on a pre-thresholded
image. Should `watershed` accept all three via an enum, or only user-supplied seeds
(keeping it composable), with helper functions for modes (b) and (c)?

### OQ-7: Python bindings — manylinux wheels vs. source-only (Milestone 4 — vm-python)

Will `vm-python` ship pre-built wheels on PyPI (requires manylinux CI), or is
source-only `pip install` (requires Rust toolchain on the user machine) acceptable?
This drives CI infrastructure decisions and must be decided before publishing.

---

*This roadmap is a living document. Update the status table in Section 2 and check off
backlog items in Section 5 as work lands. Revisit open questions at the start of each
milestone.*
