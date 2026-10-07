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
  `tangents`, `curvatures` and `arc_params` (`GraphEdge::compute_geometry`),
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
- **Breaking:** `Point2f` / `Vec2f` are nalgebra 0.35 aliases (`Point2<f32>` /
  `Vector2<f32>`), so nalgebra is a dependency. `Vec2fExt` adds `perp`, `cross` and
  `normalized_or_zero`, which returns the zero vector for a zero input; nalgebra's
  `normalize` returns `NaN` there.
- **Breaking:** `Error` is `#[non_exhaustive]` and adds `InsufficientData`, `Degenerate`
  and `InvalidConfig`.
- The crates declare `rust-version = "1.91"`.

### Removed

- The unused `rayon` feature and the `vm-gallery` crate.

### Fixed

- **Contour graphs shattered into fragments** because the edgel mask was not thinned.
  `ContourBuildConfig::thin` (on by default) thins it before tracing.
- **A reused `Edge1DDetector` could differ from a fresh one**, because it kept its kernel
  for any σ within `f32::EPSILON` of the cached one, and it allocated a new kernel for any
  other σ. It now compares σ exactly and refills its kernel buffers in place, so a caliper
  moved across strips of slightly different spacing no longer allocates, and a reused
  detector matches a fresh one bit for bit.

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
