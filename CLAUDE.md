# Claude Code — vision-metrology

Please read and follow **@AGENTS.md** (repo-wide conventions and invariants).

## Quick repo map

* `crates/vm-core`: image views, sampling, border modes, geometry + nalgebra type aliases
* `crates/vm-pyr`: ultra-fast 2×2 mean pyramid
* `crates/vm-edge`: 1D/2D subpixel edges (DoG), edgels
* `crates/vm-laser`: stripe extraction using opposite-polarity edge pairs
* `crates/vm-contour`: contour graph, junctions, per-edge tangent/curvature geometry, polyline smoothing
* `crates/vm-morph`: binary morphology (parameterized SE), chamfer distance, Zhang-Suen thinning
* `crates/vm-multiscale`: multi-scale edge detection across pyramid levels
* `crates/vm-shape`: LSD, Bookstein/Fitzgibbon conic fitting, RANSAC ellipse fitting
* `crates/vm-segment`: Otsu/adaptive thresholding, CCL, watershed, edgel region growing
* `crates/vm-match`: `EdgeModel` + chamfer map, rigid/similarity grid search, IoU NMS, ICP refinement
* `crates/vm-python`: PyO3 extension module exposing detectors with numpy array I/O
* `crates/vision-metrology`: umbrella re-export crate

## Key decisions

* `nalgebra 0.33` is a workspace dependency; use type aliases `Isometry2f / Similarity2f / Affine2f / Projective2f` from `vm_core` — do **not** re-implement linear algebra.
* Error type: `vm_core::Error` across all crates.
* All public output types must be `'static` / lifetime-free (PyO3 compatibility).
* Config-struct + reusable-detector API pattern throughout.

## What “good” looks like

* No per-scan allocations in extraction loops
* Tests are deterministic and explain the expected geometry
* Benches exist for the real hot functions

## When unsure

* Ask for the missing constraint (pixel format, expected ranges, thresholds, tolerances).
* Prefer a simple baseline API first; we can optimize once behavior is locked.
