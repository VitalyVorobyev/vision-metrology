# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `serde` feature: versioned `ShapeModel` persistence (`to_json` / `from_json`
  with an explicit `format_version` gate) and `Serialize`/`Deserialize` on the
  model and geometry types. Python: `ShapeModel.save(path)` / `ShapeModel.load(path)`.
- Tests: greedy-bound safety (T13 — greediness 0 is bit-identical to an
  exhaustive reference at every qualifying pose), fine-toothed-contour pyramid
  aliasing probe (R3), edge1d Centroid/threshold/typed-entry coverage,
  contour C4 and `min_component_size` behavior, multiscale config defaults,
  serialization round-trips (Rust + Python).
- Benches: `morph` (chamfer distance, Zhang-Suen thinning, open3x3), `edge1d`,
  `contour_smooth_polyline_5k_sigma2`, and a seeded cluttered scene for
  `match_shape` (`shape_find_1280x1024_360deg_clutter` — 10.4 ms vs 7.8 ms
  clean on M4 Pro, the honest baseline for the <5 ms work).
- Persistent-context documentation: `docs/system-design.md` (architecture,
  invariants, decision record), `docs/roadmap.md` (tracks and acceptance
  criteria), `docs/backlog.md` (known debt). `AGENTS.md`/`CLAUDE.md` now point
  to them, and the commit checklist requires keeping them current.

- **Shape-based object detection** (`vision_metrology::matching`). `ShapeModel`
  plus `ShapeMatcher` locate a modelled contour under translation, rotation and
  uniform scale, using the mean dot product of gradient directions (Steger,
  DAGM 2001). Invariant to any monotonic illumination change; the score reads as
  `1 - occluded_fraction` because low-contrast points contribute zero while the
  sum is still divided by the full point count.
  - `Polarity::{Match, IgnoreGlobal, IgnoreLocal}` controls which contrast
    reversals still count as the object.
  - `Refinement::{None, Interpolate, LeastSquares}`; the least-squares mode is
    correspondence-free and degrades to fewer degrees of freedom on symmetric
    parts rather than solving a singular system.
  - Models can also be built from edgels, from directed points, or from contour
    polylines (`ShapeModel::from_edgels` / `from_directed_points` /
    `from_polylines`).
- `vm_primitives::DirectionField`: a dense unit gradient direction field with a
  magnitude gate, in the layout an orientation-based scoring loop wants.
- `vm_primitives` transform helpers: `transform_point`, `transform_vec`,
  `transform_point_iso`, `similarity_from_parts`, `similarity_parts`,
  `Vec2f::perp`, `Vec2f::cross`, and `parabolic_peak_offset`.
- Python bindings for the new API: `ShapeModel`, `ShapeMatcher`, `ShapeMatch`,
  `ShapeModelConfig`, `ShapeSearchConfig`, and `find_shape_model`.
- `examples/shape_matching.rs` (synthetic self-asserting mode plus a real-data
  mode that writes overlay PNGs) and `benches/match_shape.rs`.

- `ContourBuildConfig::thin` (default `true`), which skeletonises the edgel
  occupancy mask before tracing.
- CI jobs for documentation warnings, the declared MSRV, the self-asserting
  examples, and the Python extension module.
- `deny.toml` and `.github/dependabot.yml`.
- `LICENSE-MIT` and `LICENSE-APACHE`, and a `CONTRIBUTING.md`.

### Removed

- **Breaking:** the chamfer-distance matcher — `EdgeModel`, `RigidEdgeMatcher`,
  `RigidMatchConfig`, `MatchConfig`, `MatchResult`, `RigidMatchResult`,
  `chamfer_score`, `normal_score`, `build_scene_chamfer`, `transform_points`
  and `icp_refine`, together with the `RigidMatcher` / `match_rigid_model`
  Python bindings. Its coarse metric ignored gradient orientation entirely, it
  had no pyramid and no early termination, and its angle range could not cross
  ±π. `morph::chamfer_distance_u8` is unaffected and remains public.

### Changed

- `laser/extractor.rs` (1717 lines) split into 8 focused modules; the
  u8/u16/f32 triplication collapsed into one generic implementation over a
  private `ScanPixel` trait. Public API unchanged; extract benches within
  noise of the previous numbers.
- `contour/build.rs` chain-tracing helpers take a borrowed `GridCtx`;
  all three `too_many_arguments` allows removed.
- Workspace MSRV raised from 1.89 to 1.91, ahead of the planned `corrmatch` and
  `box-image-pyramid` dev-dependencies (both declare 1.91). nalgebra 0.35 needs
  only 1.89; the crates were unpublished, so no compatibility promise changed.

- **Breaking:** `LaserExtractor::extract_line_u8`, `extract_line_u16` and
  `extract_line_f32` return `Result<LaserLine, Error>`. They previously
  asserted when `ColAccess::Transposed` was selected without a transposed view.
- Bumped nalgebra to 0.35, criterion to 0.8, and pyo3/numpy to 0.29. The
  workspace MSRV is now 1.89, set by nalgebra.
- `image` is a dev-dependency with only the `png` and `bmp` features, cutting
  the dev tree from 158 to 100 crates.
- The `vm-python` Rust lib target is named `vm_python` rather than
  `vision_metrology`, removing an output-filename collision. The Python module
  is still imported as `vision_metrology`.

### Fixed

- `watershed` was exponential in image size on flat regions and never completed
  at 1280x1024; it also mis-partitioned plateaus and produced 2 px boundaries
  biased toward the lowest-numbered seed.
- Contour graph construction shattered contours into fragments because the
  edgel mask was not thin: a ring produced 288 junctions and 432 edges instead
  of two closed loops.
- `ConicFitter::fit` panicked on non-finite input points instead of reporting a
  degenerate fit.

### Removed

- The `rayon` feature, which was declared but never used.
- The `crates/vm-gallery` crate and the duplicated figure-generation tooling.
