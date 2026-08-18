# Contributing

Development workflow for the `vision-metrology` workspace. User-facing documentation
lives in [README.md](README.md) and the per-crate READMEs; this file is for people
working *on* the library.

## Layout

Three publishable crates, two layers:

```
crates/vm-primitives     core · pyr · edge · morph
crates/vision-metrology  contour · laser · matching · multiscale · segment · shape
crates/vm-python         PyO3 bindings (depends on both)
```

Both `vm-primitives` and `vision-metrology` provide flat crate-root re-exports in
addition to module paths.

## Quality gates

Run from the workspace root before every commit:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
```

CI runs exactly these, plus a cross-platform build/test on Windows and macOS.
Documentation is built with `RUSTDOCFLAGS="-D warnings"`, so check intra-doc links
locally too:

```bash
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

## Conventions

- **Pixel-center coordinates.** Integer `i` means coordinate `i as f32`.
- **Rust-native only.** No OpenCV, no FFI.
- **Config struct + reusable detector.** Public algorithms take an `XConfig`
  (`Debug + Clone + PartialEq + Default`) and live on an `XDetector` that owns
  reusable scratch across calls.
- **Error type.** `vm_primitives::Error` throughout, with `&'static str` payloads
  only — no owned strings, no `format!`. Most APIs are infallible and return
  `Vec`/`Option`; `Result` is for constructors and validation.
- **Lifetime-free public output types**, for PyO3 compatibility.
- **Default border mode is `Clamp`** unless explicitly configured otherwise.
- **No per-scan allocations** in extraction loops; reuse detector scratch buffers.
- `unsafe` is allowed only for small, performance-critical blocks, and every block
  carries a `// SAFETY:` comment stating the invariant it relies on.

## Tests

Every test is an inline `#[cfg(test)] mod tests` in the file it tests. Fixtures are
deterministic synthetic images with known geometry — no unseeded RNG. Assertions carry
a message stating the expected geometry and the tolerance:

```rust
assert!(err < 0.01, "sub-pixel residual expected, got err={err}");
```

Doctests double as API smoke tests. `crates/vm-python` sets `doctest = false` because
its lib name deliberately collides with the `vision-metrology` package.

Python binding tests:

```bash
cd crates/vm-python && maturin develop && pytest tests/
```

## Benchmarks

Criterion, `harness = false`, one `[[bench]]` stanza per file. Benchmark IDs follow
`operation_size`; the representative image size is 1280×1024.

```bash
cargo bench --workspace

# vm-primitives
cargo bench -p vm-primitives --bench downsample
cargo bench -p vm-primitives --bench edge2d

# vision-metrology
cargo bench -p vision-metrology --bench build_graph
cargo bench -p vision-metrology --bench detect_multiscale
cargo bench -p vision-metrology --bench detect_shape
cargo bench -p vision-metrology --bench extract
cargo bench -p vision-metrology --bench match_
cargo bench -p vision-metrology --bench segment

# a single benchmark function
cargo bench -p vm-primitives --bench downsample -- downsample2x2_mean_u8_to_f32_1280x1024
```

Add a benchmark whenever you add or change a hot path. Benchmark numbers are
machine-specific; record them in the PR description rather than in tracked files.

## Commits and PRs

- Keep commits scoped and descriptive.
- If behavior changes, adjust tests in the same commit.
- Update the affected README when crate scope or public API changes.
- Do not revert unrelated changes.
