# AGENTS.md

Guidance for coding agents working in `vision-metrology`. This repo implements high-precision, high-performance image processing for industrial metrology:

**Before changing anything, read the three persistent-context documents:**
[`docs/system-design.md`](docs/system-design.md) (architecture, invariants, decisions and
why), [`docs/roadmap.md`](docs/roadmap.md) (current tracks and acceptance criteria), and
[`docs/backlog.md`](docs/backlog.md) (known debt). They are the project's long-term memory
across sessions — trust them over reconstructing state from git history.

- Morphology
- 1D/2D subpixel edges
- Laser stripe extraction (edge-pair method)
- Subpixel contours with junctions (T/Y)

## Project layout

Three publishable crates, strict one-way dependencies:

```
vm-primitives  ──►  vision-metrology  ──►  vm-python
(low-level)         (domain algorithms)    (PyO3 bindings)
```

**The module table in [`docs/system-design.md`](docs/system-design.md#layering) is the
canonical map** — what each module contains, and which crate it lives in. It is kept in one
place on purpose: four separate copies of it drifted, and each still named a `shape` module
two waves after it was renamed to `lsd`. Read it there rather than trusting a summary here.

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

The gate commands, the CI job table, and the MSRV rationale live in
[`CONTRIBUTING.md`](CONTRIBUTING.md#quality-gates) — one copy, so they cannot disagree. The
short version, run from the workspace root:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
python3 tools/check-invariants.py
```

If performance-sensitive code changed, also run the affected bench crate(s)
(`cargo bench -p vm-primitives`, `cargo bench -p vision-metrology`); see CONTRIBUTING for the
per-bench list.

## Commit checklist
- Keep commits scoped and descriptive.
- Do not revert unrelated user changes.
- Update `README.md` when crate scope, commands, or benchmark reporting changes.
- If behavior changes, include/adjust tests in the same commit.
- If the change alters scope, decisions, or invariants, update `docs/roadmap.md`,
  `docs/backlog.md`, and/or `docs/system-design.md` in the same commit — **rewriting** the
  affected entry, not appending a second one that contradicts it.
- Completed work moves out of `docs/roadmap.md` into `CHANGELOG.md`'s `[Unreleased]`
  section. The roadmap describes what is *ahead*.
- Invariant numbers in `docs/system-design.md` are append-only and are cited by number from
  source files; `tools/check-invariants.py` enforces that.
- If the change adds public Rust API, update `vm-python` bindings and a Python test in
  the same PR.

## Quick command reference
```bash
cargo test -p vm-primitives
cargo test -p vision-metrology
cargo bench -p vision-metrology --bench match_shape
python3 tools/check-invariants.py
```

The full bench list is in [`CONTRIBUTING.md`](CONTRIBUTING.md#benchmarks).
