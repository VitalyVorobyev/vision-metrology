# Claude Code — vision-metrology

Please read and follow **@AGENTS.md** (repo-wide conventions and invariants).

## Quick repo map

Three published crates, two layers:

* `crates/vm-primitives`: low-level building blocks
  * `::core` — image views, sampling, border modes, geometry + nalgebra type aliases
  * `::pyr`  — ultra-fast 2×2 mean pyramid
  * `::edge` — 1D/2D subpixel edges (DoG), edgels, edge-pairs
  * `::morph` — binary morphology (parameterized SE), chamfer distance, Zhang-Suen thinning
* `crates/vision-metrology`: high-level domain modules (depends on `vm-primitives`)
  * `::contour`   — contour graph, junctions, per-edge tangent/curvature, polyline smoothing
  * `::laser`     — stripe extraction using opposite-polarity edge pairs
  * `::matching`  — `ShapeModel` + `ShapeMatcher`, gradient-orientation shape-based object detection
  * `::multiscale`— multi-scale edge detection across pyramid levels
  * `::segment`   — Otsu/adaptive thresholding, CCL, watershed, edgel region growing
  * `::shape`     — LSD, Bookstein/Fitzgibbon conic fitting, RANSAC ellipse fitting
* `crates/vm-python`: PyO3 extension module exposing detectors with numpy array I/O

Both `vm-primitives` and `vision-metrology` provide flat crate-root re-exports in addition to module paths.

## Key decisions

* `nalgebra 0.35` is a workspace dependency; use type aliases `Isometry2f / Similarity2f / Affine2f / Projective2f` from `vm_primitives` — do **not** re-implement linear algebra.
* Error type: `vm_primitives::Error` across all crates.
* All public output types must be `'static` / lifetime-free (PyO3 compatibility).
* Config-struct + reusable-detector API pattern throughout.

## What “good” looks like

* No per-scan allocations in extraction loops
* Tests are deterministic and explain the expected geometry
* Benches exist for the real hot functions

## When unsure

* Ask for the missing constraint (pixel format, expected ranges, thresholds, tolerances).
* Prefer a simple baseline API first; we can optimize once behavior is locked.
