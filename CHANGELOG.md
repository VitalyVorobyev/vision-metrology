# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

The workspace was consolidated from twelve crates into three: `vm-primitives`,
`vision-metrology` and the `vm-python` wheel. The changes below are relative to 0.1.0.

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
  - `ShapeMatch::{model_frame_map, model_frame_pose}` with `CropSpec` produce
    canonical-pose crops.
  - `matching::diagnostics::match_point_scores`.
- **Model persistence** (`serde` feature): `ShapeModel::{save, load, to_bytes, from_bytes}`.
  It is an opaque, versioned format; the current format is 5, and models from format 3 on
  load.
- **Scale invariance** (`scale`): `ShapeModel::resample_at`, `estimate_scale_moments`,
  `estimate_scale_logpolar` and `find_scale_invariant`.
- **1-D edge operators** (`vm-primitives::edge`): `Derivative1D::SmoothThenCentral`
  (Gaussian, then central differences, both in `f64`), `SubpixRefine::Gaussian3`
  (log-parabola peak fit), `Edge1DDetector::{response, smooth_in_ref}`, and
  `DoGKernel1D::with_radius`.
- **Level crossings** (`vm-primitives::edge`): `LevelCrossing1D` with `end_levels`
  (`np.median` of each end), `crossings` (linear interpolation, CaliperBench's equality
  rules) and `half_contrast` (`HalfContrastConfig`, `LevelEdge`, `LevelOutcome`).
- **Calipers and metrology models** (`measure`).
  - `Caliper` with `MeasureRect` / `MeasureArc` / `MeasureRadial` placements, a typed
    `RejectReason`, an optional obliquity gate, and `measure_pairs`.
  - `MeasureConfig::locate` (`Locate::GradientPeak { refine }`) and
    `ProfileConfig::derivative` (`Derivative`) select the subpixel refinement and the
    derivative operator; Python: `vm.Locate.gradient_peak(refine=...)`,
    `MeasureConfig(derivative=..., kernel_radius_px=...)`.
  - `MeasureStrip` (`Caliper::strip`, `Caliper::set_strip`): a straight scan with exact
    endpoints, optional explicit `samples` and `across` counts, and `t` measured from
    `start`. `ProfileConfig::off_image` (`OffImage::{Fill, Reject}`) rejects a placement
    that leaves the image. Python: `Caliper.strip`, `Caliper.move_to_{rect, arc, radial,
    strip}`, `MeasureConfig(off_image=...)`.
  - `EdgeSelect::StrongestInOrder(EdgeSequence)`: one or two edges in scan order, each
    the strongest of its polarity after the previous one, with
    `RejectReason::IncompleteSequence` when a later one is missing. Python:
    `MeasureConfig(select="in_order", sequence=[...])`.
  - `Locate::MidpointCrossing` (one edge at the mean of the profile's end levels,
    checked in CaliperBench's order) and `Locate::HalfContrast` (gradient edges moved to
    their local half-contrast crossing), with `RejectReason::{LowContrast, NoCrossing}`
    and `Caliper::levels`. Python: `vm.Locate.midpoint_crossing(...)`,
    `vm.Locate.half_contrast(...)`, `Caliper.levels()`, `vm.LevelEdge`.
  - `MetrologyModel` applies line and circle objects at a fixture pose and fits them.
  - `measure::diagnostics::layout` gives caliper placement without an image.
  - `measure::diagnostics::explain` traces one measurement (`CaliperTrace`: the profile,
    its smoothed version and derivative, the candidates before `select`, the level
    crossings, and the edges or rejection `measure` returns), and `Caliper::spacing`
    gives the distance between profile samples. With the `serde` feature `CaliperTrace`
    serializes, and `MeasureEdge`, `RejectReason`, `EdgePolarity` and `LevelEdge`
    (de)serialize. Python: `Caliper.explain(img)`, `Caliper.spacing()`,
    `vm.CaliperTrace`.
- **Robust fitting** (`fit`): `fit_line`, `fit_circle` (Taubin then Gauss–Newton) and
  `fit_ellipse`, with `RobustLoss::{Huber, Tukey}` (annealed) and `RansacConfig`. Every fit
  reports `rms`, `max_dev` and `n_used`.
- **Image warping** (`warp`): `Map::{affine, projective, polar, log_polar, from_fn}` and
  `apply` / `apply_with_mask` with a validity mask.
- **Calibration bridge** (`metric`).
  - `CameraModel`, `PinholeIntrinsics`, `BrownConrady5`, `Pose3`, `Plane3`, `PlaneGrid`.
  - `pixel_to_plane`, `plane_grid_map`, `undistort_map`.
  - Importers for calibration-rs rig exports and `table_calibration` JSON.
- **Cross-correlation** (`corr`): `CorrTemplate`, `find`, `find_topk` over corrmatch, and
  `displacement` with optional Lucas–Kanade refinement.
- **LSD line segments** (`lsd`), now downsampling through `pyr`.
- **Primitives.**
  - `Pixel` trait over `u8`/`u16`/`f32` with one generic entry point per algorithm.
  - `Pyramid` generic over `Pixel` with optional binomial pre-smoothing, and
    `level_to_base` / `base_to_level`.
  - `DirectionField` with lazy tiling (`TiledField`).
  - `Circle2f` / `Ellipse2f` / `Conic2f`, transform helpers, and a `prelude` in both crates.
  - `ContourBuildConfig::thin`.
- **Python bindings** for every module above except `laser`, plus `uint8`/`uint16`/`float32`
  dispatch, `.pyi` stubs and `py.typed`.
- **Examples:** `shape_matching`, `inspect_canend`, `measure_circles`, `align_crops`,
  `pose_audit`, `birdseye_mosaic`, and `caliperbench_run`, which runs strip calipers over
  a CaliperBench requests file through its JSONL protocol. Its `gradient_parabolic`,
  `gradient_integer` and `midpoint_crossing` methods return the same rows as
  CaliperBench's baselines (a golden cross-check pins it).
- **Docs:** guides for shape matching and measurement, and a performance and accuracy page.
- An accuracy regression suite with pinned envelopes, and benches for matching, measure,
  warp, corr, morph and edge1d.

### Changed

- **Breaking:** each domain module of `vision-metrology` is now a default-on Cargo feature.
  Names live at their module path; the flat crate-root re-export of domain types is gone,
  and `prelude` replaces it.
- **Breaking:** `_u8`/`_u16`/`_f32` variants of one operation were merged into one generic
  function (Rust) or one dtype-dispatching function (Python).
- **Breaking:** configs use `Option` and enums instead of sentinel values.
  - `Hysteresis` replaces the paired threshold fields.
  - `ShapeSearchConfig` and `LaserExtractConfig` move their effort fields into a nested
    `tuning`.
  - `min_contrast` is a `Contrast::{Raw, FractionOfRange}`.
- **Breaking:** `Edge1DConfig` gained a `derivative` field; the parabolic peak offset
  is clamped to ±0.5 samples (the most a strict local maximum can produce).
- **Breaking:** `MeasureConfig`'s `sigma`, `step` and `border` moved into
  `MeasureConfig::profile` (`ProfileConfig`). Python's `MeasureConfig` keyword
  arguments are unchanged.
- **Breaking:** `Caliper::measure` returns `Result<&[MeasureEdge], RejectReason>`.
  `MetrologyModel::apply` returns one `Result` per object, in object order.
- **Breaking:** `Point2f` / `Vec2f` are nalgebra aliases; `Vec2fExt` adds `perp`, `cross`
  and `normalized_or_zero`.
- **Breaking:** the `shape` module and feature are now `lsd`.
- **Breaking:** `LaserExtractor::extract_line` returns `Result` instead of panicking on a
  missing transposed view.
- The MSRV is 1.91, and nalgebra is 0.35.
- The Python extension's Rust lib target is `vm_python`; the import name is still
  `vision_metrology`.

### Removed

- **Breaking:** the chamfer-distance matcher (`EdgeModel`, `RigidEdgeMatcher`,
  `match_rigid_model` and its Python bindings), replaced by `matching`.
  `morph::chamfer_distance_u8` remains.
- **Breaking:** `MultiScaleEdgeDetector`, which mapped coarse levels with a biased
  coordinate transform.
- The unused `rayon` feature and the `vm-gallery` crate.

### Fixed

- **Pyramid coordinate mapping in multi-scale paths.** It was missing the `(2^l − 1)/2`
  term, which biased fitted circle centres by 0.07–0.10 px.
- **LSD endpoints** were off by up to −0.95 px; LSD now downsamples through `pyr`.
- **A release-mode out-of-bounds write in the 2×2 downsample.** It was guarded only by a
  `debug_assert!`.
- **`watershed` was exponential on flat regions.** It now completes, partitions plateaus
  correctly, and draws 1 px boundaries.
- **Contour graphs shattered into fragments** because the edgel mask was not thinned.
- **Fitters panicked on non-finite input** instead of reporting a degenerate fit.
- **Python `MetrologyModel.apply` ignored the model origin.** The fixture now matches
  `ShapeMatch::pose`.

### Lab

- Contours on the canvas are `@vitavision/stage2d`'s `PolylineSet`: drawn batched and picked
  through a spatial index, with hover shared with the inventory. Sweeps start from bare
  image, the region or a contour through stage2d's `StageSurface` and `useStageDrag`, and a
  sweep now catches a contour that crosses the band between two of its points. Overlays
  use stage2d's role colours (`feature`, `model`, `structure`, `selection`) on a halo, and
  the datum is drawn as an origin ring with its i and j axes.
- The header's frame switcher is `@vitavision/workbench`'s `SequenceNavigator`, a strip of
  thumbnails with `[` / `]`. The Library opens files through workbench's `FileDrop`:
  dropped anywhere on the window, or picked. The desktop app opens them by path, and the
  browser build uploads them.
- The canvas runs on `@vitavision/stage2d` 0.8: its image layer, its region editor and its
  stage handle. Frames open at fit whatever their size, and the focused region moves with
  the arrow keys (Shift ×10) and resizes with Alt + arrows. The workspace rail sits in
  `@vitavision/workbench` 0.2's shell rail.
- The frontend builds on TypeScript 6, React 19.3, vitest 5 and ESLint 10 with the shared
  `@vitavision/config-ts` and `@vitavision/config-eslint` presets; CI lints it.
- The frontend uses the `@vitavision/ui`, `stage2d`, `charts` and `workbench` packages
  directly. The shell is workbench's `AppShell` with a resizable, remembered inspector;
  Align and Bird's-eye zoom views use `ImageStage`.
- Region, datum origin and crop rectangle are edited as vectors with units, and numeric
  fields keep what is typed while focused, so clearing a field to retype it no longer reads
  as zero.
- The browser build no longer reports an error after building a model (reading model
  geometry is desktop-only).
- A new interactive workbench in `lab/`: a browser build over FastAPI and the Python
  bindings, and a Tauri desktop build calling the Rust library directly. Contract fixtures
  keep the two in agreement.
- **Screens:** Library, Teach, Find, Verify, Measure (pixels or millimetres), Align,
  Motion and Mosaic.
- **Teach** is a workbench:
  - an editable region and a contour inventory (sort, filter, keep or drop, keyboard
    stepping, sweep-select);
  - a datum handle and per-level model point counts;
  - a frame switcher on every screen.
- **The canvas** keeps image and overlays registered at any window size and zoom. It has
  standard zoom controls, layer toggles, and pixel-centre-correct overlays.
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
  - binary morphology;
  - thresholding and connected components;
  - first Python bindings.
