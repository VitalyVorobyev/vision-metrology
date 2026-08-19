# Claude Code — vision-metrology

Please read and follow **@AGENTS.md** (repo-wide conventions and invariants).

Persistent context lives in `docs/system-design.md` (architecture + decisions),
`docs/roadmap.md` (current tracks), and `docs/backlog.md` (known debt) — read them at the
start of a session and keep them updated when scope or decisions change.

## Quick repo map

Three published crates, two layers:

* `crates/vm-primitives`: low-level building blocks
  * `::core` — image views, `Pixel` trait, sampling, border modes, geometry (nalgebra aliases)
  * `::pyr`  — `Pyramid`, 2×2 mean, generic over `Pixel`, optional pre-smooth
  * `::edge` — 1D/2D subpixel edges (DoG), edgels, edge-pairs
  * `::morph` — binary morphology (parameterized SE), chamfer distance, Zhang-Suen thinning
* `crates/vision-metrology`: high-level domain modules (depends on `vm-primitives`)
  * `::contour`   — contour graph, junctions, per-edge tangent/curvature, polyline smoothing
  * `::laser`     — stripe extraction using opposite-polarity edge pairs
  * `::matching`  — `ShapeModel` + `ShapeMatcher`, gradient-orientation shape-based object detection
  * `::segment`   — Otsu/adaptive thresholding, CCL, watershed, edgel region growing
  * `::shape`     — LSD, Bookstein/Fitzgibbon conic fitting, RANSAC ellipse fitting
* `crates/vm-python`: PyO3 extension module exposing detectors with numpy array I/O

Names live at their module path. Both crates ship a `prelude`; crate-root re-exports are explicit lists, never globs. Every `vision-metrology` module is a default-on feature.

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
