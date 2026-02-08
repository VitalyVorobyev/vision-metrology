# AGENTS.md

Guidance for coding agents working in `vision-metrology`. This repo implements high-precision, high-performance image processing for industrial metrology:

- Morphology
- 1D/2D subpixel edges
- Laser stripe extraction (edge-pair method)
- Subpixel contours with junctions (T/Y)

## Project layout
- `crates/vm-core`: core image/geometry/sampling primitives.
- `crates/vm-pyr`: 2x2 mean pyramid (no Gaussian downsample).
- `crates/vm-edge`: 1D DoG kernels/convolution/edge + edge-pair primitives.
- `crates/vm-laser`: laser line extraction (rows/cols, ROI+prior tracking).
- `crates/vision-metrology`: umbrella re-export crate.

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
cargo fmt
cargo clippy --all-targets
cargo test
```

If performance-sensitive code changed, also run benchmarks:

```bash
cargo bench --workspace
```

At minimum, run affected bench crate(s):

```bash
cargo bench -p vm-pyr
cargo bench -p vm-laser
```

## Commit checklist
- Keep commits scoped and descriptive.
- Do not revert unrelated user changes.
- Update `README.md` when crate scope, commands, or benchmark reporting changes.
- If behavior changes, include/adjust tests in the same commit.

## Quick command reference
```bash
cargo test -p vm-core
cargo test -p vm-edge
cargo test -p vm-laser
cargo test -p vm-pyr
cargo bench -p vm-pyr --bench downsample
cargo bench -p vm-laser --bench extract
```
