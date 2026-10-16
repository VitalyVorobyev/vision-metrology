# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The seven library crates of 0.1.0 merge into two, `vm-primitives` and `vision-metrology`,
plus the `vm-python` wheel; the `vm-gallery` crate is gone. Items of `vm_core`, `vm_edge`,
`vm_pyr` and `vm_morph` move to `vm_primitives::…`, and items of `vm_contour` and
`vm_laser` to `vision_metrology::contour` and `vision_metrology::laser`. The changes below
are relative to 0.1.0.

### Added

- **Continuous derivative-of-Gaussian profiles.** `vm_primitives::edge::ProfileSampler`
  samples the smoothed intensity and its derivative along any direction at continuous
  points, with a Gaussian tapered by `(1 − d²/r²)²` to zero at `r = 4σ`, so both are C¹.
  A sample whose support leaves the image or touches a masked pixel is `None`. It is
  allocation-free once warm, with Python parity (`ProfileSampler`).
  `measure::Derivative::Continuous` (Python `derivative="continuous"`) uses it in every
  caliper placement and the bead tracker's stages, and refines each gradient peak on the
  continuous response. Over the phase sweep it stops locking to the pixel grid at σ 2.5 px,
  unblurred steps included, where the default strip still locks; the default does not
  change. The sweep's rows per σ, the bead tracker's and the costs are in
  `docs/performance.md`.
  `phase_histogram.py --refine sampler` runs the same diagnostic with the library's
  sampler.

- **Sub-pixel locking evidence.** `tests/accuracy/phase.rs` sweeps every sub-pixel edge
  locator over 20 phases × 4 angles × 3 blurs and pins its worst per-phase bias, σ and
  locking index (the max/min ratio of a 10-bin histogram of fractional positions over a
  dense phase sweep under noise). `tools/registration_eval/phase_histogram.py` computes
  the same histogram over a `sequence.json` manifest without ground truth, with a
  continuous derivative-of-Gaussian reference. Results are in `docs/performance.md`; the
  sweep found the `Edge2DDetector` defect under Fixed.

- **Local masked similarity, experimental** (`corr::SimilarityEstimator`, behind the
  non-default `corr-experimental` feature; in Python, `vision_metrology.experimental`).
  Outside the semver promise until its regional check is recalibrated. Previous-to-current
  `Similarity2f`, explicit prediction/bounds, robust photometry, motion rank checks,
  covariance and independent regional map comparisons; native scratch reuse and
  matching Python bindings. Development tooling (`tools/registration_eval`) reads a
  neutral sequence manifest and adds reciprocal seed anchors and a standalone overlay
  viewer.

- **Masked subpixel translation** (`corr::TranslationEstimator`): reusable scratch,
  independent per-image validity masks, bounded spatial registration with gain/offset
  refinement, typed acceptance, photometric residuals, covariance and regional motion
  consistency. Generic over u8/u16/f32, with Python bindings and allocation/accuracy gates.

- **Shape-based object detection** (`matching`).
  - `ShapeModel` / `ShapeMatcher`: gradient-orientation similarity with
    `Polarity::{Match, IgnoreGlobal, IgnoreLocal}`, coarse-to-fine search over translation,
    rotation and uniform scale, and `Refinement::{None, Interpolate, LeastSquares}`.
  - Models can be built from images, edgels, directed points or polylines.
  - `ShapeModelBuilder::build_with_mask` teaches from an inclusion mask.
  - `ShapeModelConfig::reference_angle` sets the model's canonical 0°.
  - `ShapeModel::{model_geometry, reference_geometry, reference_points}` expose the learned
    geometry.
  - `ShapeSearchConfig` keeps the search's effort settings in `tuning`
    (`ShapeSearchTuning`), and its `min_contrast` is a `Contrast::{Raw, FractionOfRange}`,
    so the threshold carries its unit.
  - `ShapeMatch::{model_frame_map, model_frame_pose}` with `CropSpec` produce
    canonical-pose crops.
  - `matching::diagnostics::match_point_scores`.
- **Model persistence** (`serde` feature): `ShapeModel::{save, load, to_bytes, from_bytes}`,
  in an opaque, versioned format.
- **Scale invariance** (`scale`): `ShapeModel::resample_at`, `estimate_scale_moments`,
  `estimate_scale_logpolar` and `find_scale_invariant`.
- **1-D edge operators** (`vm-primitives::edge`): `Derivative1D::SmoothThenCentral`
  (Gaussian, then central differences, both in `f64`), `SubpixRefine::Gaussian3`
  (log-parabola peak fit), `Edge1DDetector::{response, smooth_in_ref}`, and
  `DoGKernel1D::with_radius`.
- **Level crossings** (`vm-primitives::edge`): `LevelCrossing1D` with `end_levels` (the
  median of the samples at each end), `crossings` (every crossing of a level, linearly
  interpolated between the two samples it falls between) and `half_contrast` (an edge
  moved, iteration by iteration, to the crossing of the mean of its two flank medians),
  with `HalfContrastConfig`, `LevelEdge` and `LevelOutcome`.
- **Calipers and metrology models** (`measure`).
  - `Caliper` with `MeasureRect` / `MeasureArc` / `MeasureRadial` / `MeasureStrip`
    placements and an optional obliquity gate. `Caliper::measure` returns
    `Result<&[MeasureEdge], RejectReason>`, with a typed reason when nothing is found, and
    `Caliper::measure_pairs` returns `MeasurePair`s (bar and gap widths).
  - `MeasureStrip` (`Caliper::strip`, `Caliper::set_strip`): a straight scan with exact
    endpoints, optional explicit `samples` and `across` counts, and `t` measured from
    `start`.
  - `MeasureConfig` holds what counts as an edge (`threshold`, `polarity`, `select`,
    `locate`, `max_obliquity_deg`), and `MeasureConfig::profile` (`ProfileConfig`) how the
    profile is built: `sigma`, `derivative` (`Derivative`), `step`, `border` and
    `off_image` (`OffImage::{Fill, Reject}`, whether a placement that leaves the image is
    measured or rejected).
  - `Locate` chooses how an edge is located: `GradientPeak { refine }` (a derivative
    extremum with subpixel refinement), `MidpointCrossing` (one edge where the profile
    crosses the mean of its two end levels) or `HalfContrast` (gradient edges moved to
    their local half-contrast crossing), with `RejectReason::{LowContrast, NoCrossing}`
    and `Caliper::levels`.
  - `EdgeSelect::StrongestInOrder(EdgeSequence)`: one or two edges in scan order, each
    the strongest of its polarity after the previous one, with
    `RejectReason::IncompleteSequence` when a later one is missing.
  - `MetrologyModel` applies line and circle objects at a fixture pose and fits them.
    `MetrologyModel::apply` returns one `Result` per object, in object order.
  - `measure::diagnostics`: `layout` and `layout_object` give caliper placements without
    an image; `explain` traces one measurement (`CaliperTrace`: the profile, its smoothed
    version and derivative, the candidates before `select`, the level crossings, and the
    edges or rejection `measure` returns); `explain_model` traces a whole
    `MetrologyModel` in one pass, one `ObjectTrace` per object with every caliper measured
    once. `Caliper::spacing` gives the distance between profile samples.
  - With the `serde` feature, `CaliperTrace` serializes, and `MeasureEdge`,
    `RejectReason`, `EdgePolarity` and `LevelEdge` (de)serialize.
  - Python: `Caliper.{rect, arc, radial, strip}`, `Caliper.move_to_{rect, arc, radial,
    strip}`, `Caliper.{measure, measure_pairs, levels, spacing, explain}`,
    `MeasureConfig(select="in_order", sequence=[...], derivative=..., kernel_radius_px=...,
    off_image=...)`, `vm.Locate.{gradient_peak, midpoint_crossing, half_contrast}`,
    `MetrologyModel.{apply, layout, explain}`, `vm.CaliperTrace` and `vm.ObjectTrace`.
- **Bead tracking** (`measure`).
  - `BeadTracker` refines a prior polyline from caliper evidence and measures the bead
    along the refined curve. Stations uniform in arc length, a strip caliper along each
    station's normal, the bead's edge pair chosen through typed gates, one robust,
    regularised solve for the normal corrections (a banded `LDLᵀ` in `f64` under IRLS)
    with a fold guard, and a separate, stricter final stage that measures without moving
    the curve.
  - `BeadConfig`: `polarity` (`BeadPolarity::{Light, Dark}`), `min_width`, `max_width`,
    `spacing`, `clearance`, `min_margin`, a `BeadCaliper` per stage (`track`, `measure`;
    `BeadCaliper::to_measure_config`) and `BeadTuning` (passes, tolerance, damping, the
    tension and bending lengths, the `RobustLoss`, IRLS iterations, the tangent window and
    the minimum support).
  - `TrackedBead`: the refined `centerline`, which is the next frame's prior; one
    `BeadSample` per station with a `BeadHit` (a `MeasurePair`, its offset and confidence)
    or a `BeadReject` (`Caliper(RejectReason)`, `NoPair`, `Width`, `Offset`, `Clearance`,
    `Ambiguous`); a `BeadSummary` (support, longest gap, `BeadStats`, rejections by
    reason); and a `BeadTrack` (one `BeadPass` per pass, with a `BeadSolve` when the pass
    solved, and `BeadStop::{Converged, PassLimit, TooFewValid}`). `BeadReject` and
    `BeadStop` have `as_str`.
  - `measure::diagnostics::explain_bead` tracks once, through the same code as `track`,
    and returns a `BeadTrace`: `track`'s result to the bit; one `BeadPassTrace` per pass
    with each station's `BeadStationTrace` (point, tangent, normal, offset window, strip,
    `CaliperTrace`, and pair or rejection), the solve's weights and the corrections
    applied; and the final stage's stations. With the `serde` feature, `BeadTrace` and
    `MeasureStrip` serialize.
  - `RejectReason` derives `Hash`. With the `serde` feature, `MeasurePair` and the bead
    results (de)serialize. The prelude gains `BeadTracker`, `BeadConfig` and
    `TrackedBead`.
  - Python: `vm.BeadTracker(config).track(image, prior)` with `config` read and assigned
    and a `float32` or `float64` prior, `vm.TrackedBead` (per-station arrays, reject
    strings, statistics, and `vm.BeadPass` records with an optional `vm.BeadSolve`),
    `vm.BeadConfig`, `vm.BeadCaliper` (`to_measure_config`) and `vm.BeadTuning`.
    `BeadTracker.explain(image, prior)` returns a `vm.BeadTrace`: each pass as a
    `vm.BeadPassTrace` of per-station arrays with a `vm.CaliperTrace` per station, and the
    final stage as `vm.BeadStationTrace`s.
  - `docs/performance.md` gives the tracker's accuracy, convergence basins and speed, how
    it behaves on real concrete cracks (the DamSegment dataset), and how well a ridge
    detector finds a prior when there is none. `tools/bead_eval/` reproduces the
    real-data numbers offline.
- **Robust fitting** (`fit`): `fit_line`, `fit_circle` (Taubin then Gauss–Newton) and
  `fit_ellipse`, with `RobustLoss::{Huber, Tukey}` (annealed) and `RansacConfig`. Every fit
  reports `rms`, `max_dev` and `n_used`.
- **Segmentation** (`segment`): Otsu and adaptive (local-mean) thresholding,
  connected-component labelling with C4 or C8 connectivity and per-component
  `ComponentStats`, marker-based `watershed` that partitions plateaus and draws 1 px
  boundaries, and `grow_regions`, which grows regions bounded by a `ContourGraph`.
- **Contour geometry** (`contour`): `ContourBuildConfig::record_geometry` with per-edge
  `tangents`, `curvatures` and `arc_params`,
  `ContourGraph::{iter_edges_by_length, filter_edges_min_length}`, and
  `contour::smooth_polyline` (Gaussian polyline smoothing).
- **Image warping** (`warp`): `Map::{affine, projective, polar, log_polar, from_fn}` and
  `apply` / `apply_with_mask` with a validity mask.
- **Calibration bridge** (`metric`).
  - `CameraModel`, `PinholeIntrinsics`, `BrownConrady5`, `Pose3`, `Plane3`, `PlaneGrid`.
  - `pixel_to_plane`, `plane_grid_map`, `undistort_map`.
  - Importers for calibration-rs rig exports and `table_calibration` JSON.
- **Cross-correlation** (`corr`): `CorrTemplate`, `find`, `find_topk` over corrmatch, and
  `displacement` with optional Lucas–Kanade refinement.
- **LSD line segments** (`lsd`).
- **Primitives.**
  - `Pixel` trait over `u8`/`u16`/`f32` with one generic entry point per algorithm.
  - `Pyramid` with optional binomial pre-smoothing (`PyramidConfig`, `PreSmooth`), and
    `level_to_base` / `base_to_level`.
  - `DirectionField` with lazy tiling (`TiledField`).
  - `Circle2f` / `Ellipse2f` / `Conic2f`, `Rect2f`, transform helpers, and a `prelude` in
    both crates.
  - Morphology over a `StructuringElement` (`Square` or `Disk` of any radius):
    `erode_binary_u8`, `dilate_binary_u8`, `open_binary_u8`, `close_binary_u8`, plus
    `thin_binary_u8` (Zhang–Suen) and `chamfer_distance_u8` (Borgefors 3-4-5). The 3×3
    functions remain.
- **Python bindings** (`vm-python`, import name `vision_metrology`) for 2-D edges,
  morphology and every domain module except `laser`, with `uint8`/`uint16`/`float32`
  dispatch, `.pyi` stubs and `py.typed`. The vm-python README lists what is bound.
- **Examples:** one per module (`pyramid`, `edge_1d`, `edge_2d`, `contour_graph`,
  `morphology`, `line_segments`, `segmentation`, `laserline`, `shape_matching`,
  `measure_circles`); end-to-end programs (`inspect_canend`, `align_crops`, `pose_audit`,
  `birdseye_mosaic`); `bead_track`, which tracks a synthetic bead from a perturbed prior
  and draws its rejected stations by reason; and `caliperbench_run`, which runs strip
  calipers over a CaliperBench requests file through its JSONL protocol. Its
  `gradient_parabolic`, `gradient_integer` and `midpoint_crossing` methods return the same
  rows as CaliperBench's baselines (a golden cross-check pins it). Python scripts are in
  `examples/python/`.
- **Docs:** guides for shape matching, measurement and bead tracking, and a performance and
  accuracy page.
- **Public datasets** (`tools/datasets/`): a registry of the external datasets used for
  real-data numbers, with each one's licence as stated at its source, a standard-library
  fetch script that verifies SHA-256 checksums into the git-ignored `data/`, and committed
  dev/test splits of IDs. DamSegment (split by image) and HPatches (split by sequence)
  are registered.
- An accuracy regression suite with pinned envelopes; its strip and caliper rows run on
  CaliperBench's pixel-integrated image model: each `Locate` method on steps, bar centre
  and width, oblique strips, and rect, arc and radial calipers; its bead rows pin the
  tracker's centre, width and refined curve. Benches for matching, measure, bead tracking,
  warp, corr, segment, LSD with fitting, morph and edge1d.

### Changed

- **Breaking:** each domain module of `vision-metrology` is a default-on Cargo feature.
  Names live at their module path; the flat crate-root re-export of domain types is gone,
  and `prelude` replaces it.
- **Breaking:** the `_u8`/`_u16`/`_f32` variants of one operation merge into one function
  generic over `Pixel`, for example `Edge2DDetector::detect` and
  `Edge1DDetector::{detect_in, detect_in_ref}`.
- **Breaking:** `Edge2DConfig::hysteresis` (`Hysteresis::{Auto, Manual { low, high }}`)
  replaces `low_thresh` / `high_thresh`, whose `0.0` meant "choose for me", and
  `Edge2DConfig::pre_smooth` is removed: `SmoothKind::None` switches pre-smoothing off.
- **Breaking:** `LaserExtractConfig` keeps what counts as a stripe (`axis`, `min_score`,
  `min_width`, `max_width`) and moves the search fields into `tuning`
  (`LaserExtractTuning`). `enable_smoothing` becomes `tuning.smoothing`
  (`CenterSmoothing::{None, Median { half_window }}`); 0.1.0's smoothing is
  `Median { half_window: 2 }`.
- **Breaking:** `LaserExtractor::extract_line` replaces `extract_line_{u8,u16,f32}` and
  returns `Result`: a missing or mismatched transposed view is an `Error::InvalidConfig`
  instead of a panic. `best_pair_with_prior` and `coarse_center_{u8,u16,f32}` are not
  public.
- **Breaking:** `PyramidF32` becomes `Pyramid`, with one generic `build` (and
  `build_with` for a `PyramidConfig`) in place of `build_from_{u8,u16,f32}`; `ensure` is
  private. Levels are `f32`. The `downsample2x2_mean_*` functions are not public;
  `Pyramid` is the entry point.
- **Breaking:** `Edge1DConfig` gains a `derivative` field (`Derivative1D`), and the
  parabolic peak offset is clamped to ±0.5 samples, the most a strict local maximum can
  produce (±1.0 in 0.1.0).
- **Breaking:** `Point2f` / `Vec2f` are nalgebra aliases (`Point2<f32>` /
  `Vector2<f32>`), so nalgebra is a dependency. `Vec2fExt` adds `perp`, `cross` and
  `normalized_or_zero`, which returns the zero vector for a zero input; nalgebra's
  `normalize` returns `NaN` there.
- The crates accept nalgebra `>=0.34, <0.36`, so a crate that also uses calibration-rs
  (nalgebra 0.34) can depend on them with one nalgebra in its graph. Cargo picks 0.35 for
  a fresh lock; run `cargo update -p nalgebra@0.35.0 --precise 0.34.<n>` once to unify.
  CI tests 0.34.0 and the newest 0.35, and builds such a downstream crate.
- **Breaking:** `Error` is `#[non_exhaustive]` and adds `InsufficientData`, `Degenerate`
  and `InvalidConfig`.
- The crates declare `rust-version = "1.91"`.
- **Breaking:** `ShapeModel`, `ShapeModelLevel` and `ModelPoint` no longer implement
  `Serialize`/`Deserialize`. The stored format is private document types, unchanged byte
  for byte, so files saved before still load. Persist a model with
  `ShapeModel::{save, load, to_bytes, from_bytes}`.
- **Breaking:** `ShapeModelLevel` and `ModelPoint` are private, and
  `ShapeModel::{levels, level, smooth, pre_smooth}` are gone.
  `ShapeModel::level_info(i)` returns a `matching::diagnostics::LevelInfo` (`points`,
  `radius`, `angle_step`, `scale_step`); `point_count(i)` counts a level's points, and
  `model_geometry(i)` copies them out. Python adds `ShapeModel.level_info` and
  `vm.LevelInfo`.
- **Breaking:** `ShapeModelConfig` keeps what the model is (`min_contrast`, `origin`,
  `reference_angle`, `angle_range`, `scale_range`, `polarity`) and moves the extraction
  effort into `tuning` (`ShapeModelTuning`: `num_levels`, `edge`, `pre_smooth`,
  `max_points`, `min_points_per_level`), as `ShapeSearchConfig` does. Write
  `cfg.tuning.max_points`, or `tuning: ShapeModelTuning { max_points, ..Default::default() }`
  in a literal. Python's `ShapeModelConfig` keeps its flat keywords.
- **Breaking:** `Fit::inliers` always lists the indices of the points the fit used, in
  ascending order, so `inliers.len() == n_used`; it is no longer empty when every point
  was used. Drop any `inliers.is_empty()` "all points" check.
- Internal: `measure` holds two crate-private numerical cores (ADR-0020). One is an exact
  layered min-path with max-marginals, for the coupled two-boundary extraction and the
  bead-end rim. The other is a robust Whittaker smoother, on which `BeadTracker`'s
  regularised solve now runs, with its results unchanged to the bit. No public API
  changes, and there is nothing new to bind.

### Removed

- The unused `rayon` feature and the `vm-gallery` crate.
- The dev-only `trace-cands` feature and its stderr instrumentation in
  `ShapeMatcher::find`. Time the call from outside, as the `stage_timing` example does.
- The hidden `pyr::bench_downsample2x2_mean_to_f32_into` hook. `Pyramid::build` is the
  way in; the `downsample` bench measures one level through it under the same IDs.
- **Breaking:** `core::{to_f32, to_f32_u16}`. Level 0 of a `Pyramid` is the `f32` copy
  (`pyr.build(&view, 1); pyr.level(0)`), or map a row with `Pixel::to_f32`.
- **Breaking:** `edge::best_edge_pair_in_row_u8`. Call
  `best_edge_pair(det.detect_in_ref(view.row(y), &cfg), &pair_cfg)`.
- **Breaking:** `core::map_index` is crate-private; sample through `BorderMode` with
  `sample_nearest` / `sample_bilinear_f32`.
- `core::Angle`, `fit::fit_conic` and its Bookstein solver, and
  `metric::{normalized_to_pixel, pixel_to_normalized_linear}` (the last two are now
  crate-private). Use `wrap_angle` for angles, `fit_ellipse` for ellipses (it reports
  residuals), and `distort_pixel` / `undistort_pixel` for the camera model.
- **Breaking:** `contour::Node::degree`; read `incident_edges.len()`. `Node::idx` is
  crate-private (`p` is the node's position).
- `GraphEdge::compute_geometry` and `ContourGraph::compute_all_geometry`; set
  `ContourBuildConfig::record_geometry` to get `tangents`, `curvatures` and `arc_params`.
- `Pixel`'s accumulator items (`Acc`, `to_acc`, `acc_to_f32`, `as_f32_slice`) move to its
  sealed supertrait, and the unused `acc_mean4` is gone. `Pixel` keeps `ZERO`, `to_f32`
  and `from_f32_sat`.

### Fixed

- **Behaviour change: `Edge2DDetector` edgels no longer stay on pixel centres.**
  `Subpix2D::ParabolicAlongNormal` fitted its parabola to the non-maximum-suppressed
  magnitude, whose two neighbours on a one-pixel ridge are zero, so it never moved an
  edgel. It now fits the pre-suppression gradient magnitude at the pixel and its two
  neighbours along the image axis closest to the normal, and suppression compares along
  the same axis. Edgel positions move by up to ½ px, and an edge at 30° or 45° gives one
  edgel per row instead of two, so detection returns fewer edgels on oblique edges.
  Shape models built from edgels change with them. The phase-sweep and real-data numbers
  are in `docs/performance.md` (issue #99). `Edge2DConfig::border` is documented as
  having no effect, which was already the case.
- **Contour graphs shattered into fragments** because the edgel mask was not thinned.
  `ContourBuildConfig::thin` (on by default) thins it before tracing.
- **A reused `Edge1DDetector` could differ from a fresh one**, because it kept its kernel
  for any σ within `f32::EPSILON` of the cached one, and it allocated a new kernel for any
  other σ. It now compares σ exactly and refills its kernel buffers in place, so a caliper
  moved across strips of slightly different spacing no longer allocates, and a reused
  detector matches a fresh one bit for bit.
- **`cargo test --release` failed the caliper bit pin** on aarch64 macOS. An optimised
  build lowers a `sin`/`cos` pair to one `sincos` call, which differs from separate
  `sinf`/`cosf` by an ulp on some inputs. The pin now holds one hash per build profile and
  documents what it guarantees. A new CI job runs the workspace tests in release on Linux,
  Windows and macOS.

### Lab

- **An interactive workbench** in `lab/`. The browser build runs over a FastAPI server and
  the Python bindings; the desktop build (Tauri) calls the Rust library directly. Contract
  fixtures keep the two in agreement.
- **Screens:** Library, Teach, Find, Verify, Measure (pixels, or millimetres with a
  calibration loaded), Align, Motion and Mosaic (browser only).
- **Teach** is a workbench: an editable region; a contour inventory linked to the canvas
  (sort, filter, keep or drop, keyboard stepping, sweep-select); a datum handle for the
  origin and 0° direction; and per-level model point counts.
- **Find** lists its matches in an inventory linked to the canvas: score, position, angle,
  scale and support, sortable by column. Hovering a row or a match highlights both, a
  click selects on either side, `↑` / `↓` step, `F` frames the match and `Esc` clears it.
  The selected match is the instance Verify compares. On the desktop, Find runs the same
  search across every frame; each frame shows its own matches, and the frame strip and
  menu mark the frames where the model was not found.
- **Measure** lists every caliper of every object, linked to the canvas the same way: hit
  or rejection reason, edge position, residual against the fit and amplitude, filtered to
  all, hits or rejected. The selected caliper's intensity profile is drawn along the
  caliper, with the nominal and found edges marked.
- **The canvas** keeps image and overlays registered at any window size and zoom, with
  pixel-centre-correct overlays, standard zoom controls and layer toggles. Frames open at
  fit. The focused region moves with the arrow keys (Shift ×10) and resizes with Alt +
  arrows. Contours are drawn batched and picked through a spatial index, and a sweep
  selects every contour it touches.
- **Frames and files.** A strip of thumbnails in the header switches frames on every
  screen, and `[` / `]` step through them. The Library takes images dropped anywhere on
  the window, or picked; the desktop app opens them by path, and the browser build uploads
  them.
- **The inspector** is resizable and remembers its width. Region, datum origin and crop
  rectangle are edited as vectors with units, and a numeric field keeps what is typed
  while it is focused.
- **The desktop app** opens folders by path, caches image tiers on disk, runs heavy work
  off the UI thread, batch-finds across a set, and shows a crash screen instead of a blank
  window.

## [0.1.0] – 2026-02-08

### Added

- Initial release:
  - a 2×2 mean image pyramid;
  - 1-D derivative-of-Gaussian edges and opposite-polarity edge pairs;
  - laser stripe centrelines with coarse-to-fine ROI and continuity;
  - 2-D subpixel edgels;
  - a junction-aware contour graph;
  - 3×3 binary morphology (erode, dilate, open, close).
