# Claude Code — vision-metrology

Please read and follow **@AGENTS.md** (repo-wide conventions and invariants).

## Quick repo map

* `crates/vm-core`: image views, sampling, border modes, geometry
* `crates/vm-pyr`: ultra-fast 2×2 mean pyramid
* `crates/vm-edge`: 1D/2D subpixel edges (DoG), edgels
* `crates/vm-laser`: stripe extraction using opposite-polarity edge pairs
* `crates/vm-contour`: contour graph + junctions (later)

## What “good” looks like

* No per-scan allocations in extraction loops
* Tests are deterministic and explain the expected geometry
* Benches exist for the real hot functions

## When unsure

* Ask for the missing constraint (pixel format, expected ranges, thresholds, tolerances).
* Prefer a simple baseline API first; we can optimize once behavior is locked.
