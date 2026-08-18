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
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
```

CI runs all four, and additionally:

| Job | Command |
|-----|---------|
| MSRV | `cargo +1.89.0 check --workspace --all-targets --all-features` |
| Examples | every self-asserting example under `crates/vision-metrology/examples/` |
| Python bindings | `pip install crates/vm-python` then `pytest crates/vm-python/tests` |
| Cross-platform | build and test on Windows and macOS |

The weekly security workflow additionally runs `cargo audit` and
`cargo deny check` (licences, duplicate versions, source registries). Both are
expected to pass with no ignores; if a new dependency introduces a licence that
is not in `deny.toml`'s allow-list, that is a deliberate review, not a config
oversight.

### MSRV

The workspace declares `rust-version = "1.89"` in the root `Cargo.toml`. It is
currently set by nalgebra 0.35, not by anything in this repository. `cargo
clippy` enforces it through `incompatible_msrv`, so a `std` item stabilised
later than 1.89 fails the lint rather than surfacing as a user's build error.

`rust-toolchain.toml` pins day-to-day work to stable; the MSRV job overrides it
with `cargo +1.89.0`, which takes precedence over the file.

### Python bindings

The extension module is built by maturin and imported as `vision_metrology`.
Note that the Rust lib target is deliberately named `vm_python` instead: naming
it `vision_metrology` collides with the `vision-metrology` crate's own lib
target. The Python-visible name comes from the `#[pymodule]` function name and
from `module-name` in `pyproject.toml`.

```bash
pip install crates/vm-python
pytest crates/vm-python/tests
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
