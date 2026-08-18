# AGENTS.md

Guidance for coding agents working in `vision-metrology`. This repo implements high-precision, high-performance image processing for industrial metrology:

- Morphology
- 1D/2D subpixel edges
- Laser stripe extraction (edge-pair method)
- Subpixel contours with junctions (T/Y)

## Project layout

Three publishable crates with clear layering:

- `crates/vm-primitives`: low-level building blocks.
  - `core`: image views, sampling, border modes, geometry + nalgebra type aliases.
  - `pyr`: 2×2 mean pyramid (no Gaussian downsample).
  - `edge`: 1D/2D subpixel edges (DoG), edgels, edge-pairs.
  - `morph`: binary morphology (parameterized SE), chamfer distance, Zhang-Suen thinning.
- `crates/vision-metrology`: high-level algorithms; depends on `vm-primitives`; re-exports it entirely.
  - `contour`: contour graph, junctions, per-edge tangent/curvature geometry, polyline smoothing.
  - `laser`: laser stripe extraction (rows/cols, ROI+prior tracking).
  - `matching`: `EdgeModel` + chamfer map, rigid/similarity grid search, IoU NMS, ICP refinement.
  - `multiscale`: multi-scale edge detection across pyramid levels.
  - `segment`: Otsu/adaptive thresholding, CCL, watershed, edgel region growing.
  - `shape`: LSD, Bookstein/Fitzgibbon conic fitting, RANSAC ellipse fitting.
- `crates/vm-python`: PyO3 extension module; depends on both above crates.

## Invariants and conventions
- Pixel coordinate convention: **pixel centers** (`i` means coordinate `i as f32`).
- Rust-native only; no OpenCV/FFI.
- Keep hot paths allocation-free per scan/row when possible.
- Unsafe is allowed only for small, justified performance-critical blocks.
- Default border behavior in core/edge is `Clamp` unless explicitly configured otherwise.

## Performance expectations
- Rows scanning should be the fastest path.
- Column scanning should use reusable gather buffers (or transposed mode if provided).
- Reuse detector/extractor scratch buffers across calls.

## Style (minimal)

- Keep public APIs small and explicit.
- Document coordinate conventions and border/ROI rules in crate docs.
- Prefer deterministic tests (synthetic fixtures) over “random noise” unless seeded.

## Typical tasks

### 1) Add/modify APIs
- Update crate-level docs.
- Add unit tests for behavior and edge cases.
- Keep umbrella re-exports (`crates/vision-metrology`) up to date.

### 2) Add fast path
- Implement safe fallback first.
- Add narrow unsafe path with clear safety comments.
- Validate equivalent output with tests.

### 3) Tracking/extraction changes
- Preserve bright-on-dark edge-pair selection unless explicitly changed.
- Keep continuity/gap logic deterministic.
- Ensure invalid samples are still emitted in `LaserLine.samples`.

## Required quality checks before commit
Run from workspace root:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

The workspace MSRV is `1.89` (declared in the root `Cargo.toml`, set by
nalgebra). Clippy enforces it via `incompatible_msrv`. To check it directly:

```bash
cargo +1.89.0 check --workspace --all-targets --all-features
```

If performance-sensitive code changed, also run benchmarks:

```bash
cargo bench --workspace
```

At minimum, run affected bench crate(s):

```bash
cargo bench -p vm-primitives
cargo bench -p vision-metrology
```

## Commit checklist
- Keep commits scoped and descriptive.
- Do not revert unrelated user changes.
- Update `README.md` when crate scope, commands, or benchmark reporting changes.
- If behavior changes, include/adjust tests in the same commit.

## Quick command reference
```bash
cargo test -p vm-primitives
cargo test -p vision-metrology
cargo bench -p vm-primitives --bench downsample
cargo bench -p vm-primitives --bench edge2d
cargo bench -p vision-metrology --bench extract
cargo bench -p vision-metrology --bench detect_multiscale
cargo bench -p vision-metrology --bench detect_shape
cargo bench -p vision-metrology --bench segment
cargo bench -p vision-metrology --bench build_graph
cargo bench -p vision-metrology --bench match_
```
