# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `ContourBuildConfig::thin` (default `true`), which skeletonises the edgel
  occupancy mask before tracing.
- CI jobs for documentation warnings, the declared MSRV, the self-asserting
  examples, and the Python extension module.
- `deny.toml` and `.github/dependabot.yml`.
- `LICENSE-MIT` and `LICENSE-APACHE`, and a `CONTRIBUTING.md`.

### Changed

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
