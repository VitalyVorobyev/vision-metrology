# Contributing

Development workflow for the `vision-metrology` workspace. User-facing documentation
lives in [README.md](README.md) and the per-crate READMEs; this file is for people
working *on* the library.

## Layout

Three publishable crates, two layers:

```
vm-primitives    low-level building blocks
vision-metrology domain algorithms (depends on vm-primitives)
vm-python        PyO3 bindings (depends on both)
```

The per-module breakdown — what lives in each, in both library crates — is the table in
[`docs/system-design.md`](docs/system-design.md#layering), which is the single place it is
maintained.

`vision-metrology` re-exports the curated set of `vm_primitives` names most callers need at
its own crate root, plus the `vm_primitives` crate itself and a `prelude`. Every other name —
including everything inside `vision-metrology`'s own domain modules — lives at its module path
only; there is no flat re-export block (invariant 17 in `docs/system-design.md`).

## Quality gates

Run from the workspace root before every commit:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
python3 tools/check-invariants.py
```

CI runs all five, and additionally:

| Job | Command |
|-----|---------|
| MSRV | `cargo +1.91.0 check --workspace --all-targets --all-features` |
| Invariant numbering | `python3 tools/check-invariants.py` |
| Examples | every self-asserting example under `crates/vision-metrology/examples/` |
| Python bindings | `pip install crates/vm-python` then `pytest crates/vm-python/tests` |
| Cross-platform | build and test on Windows and macOS |

The weekly security workflow additionally runs `cargo audit` and
`cargo deny check` (licences, duplicate versions, source registries). Both are
expected to pass with no ignores; if a new dependency introduces a licence that
is not in `deny.toml`'s allow-list, that is a deliberate review, not a config
oversight.

### MSRV

The workspace declares `rust-version = "1.91"` in the root `Cargo.toml`. **Why 1.91 and not
1.89** is recorded once, in [`docs/system-design.md`](docs/system-design.md) ("MSRV 1.89 →
1.91"). Operationally: `cargo clippy` enforces the floor through `incompatible_msrv`, so a
`std` item stabilised later than 1.91 fails the lint rather than surfacing as a user's build
error. `rust-toolchain.toml` pins day-to-day work to stable; the MSRV job overrides it with
`cargo +1.91.0`, which takes precedence over the file.

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
cargo bench -p vm-primitives --bench edge1d
cargo bench -p vm-primitives --bench edge2d
cargo bench -p vm-primitives --bench morph

# vision-metrology
cargo bench -p vision-metrology --bench build_graph
cargo bench -p vision-metrology --bench detect_shape
cargo bench -p vision-metrology --bench extract
cargo bench -p vision-metrology --bench match_shape
cargo bench -p vision-metrology --bench segment

# a single benchmark function
cargo bench -p vm-primitives --bench downsample -- downsample2x2_mean_u8_to_f32_1280x1024
```

Add a benchmark whenever you add or change a hot path. Benchmark numbers are
machine-specific; record them in the PR description rather than in tracked files.

## Documentation illustrations

The PNGs under `docs/assets/` (embedded in the README and the `docs/*.md` guides)
are rendered deterministically from synthetic fixtures by one example — never
committed from a private dataset frame:

```bash
cargo run --release --example gen_illustrations --all-features
```

Re-run it and commit the results whenever a change alters what one of the six
illustrations shows (shape matching, caliper anatomy, laser stripe extraction,
robust circle fit, contour graph, pyramid levels). The renderer asserts its own
fixtures (found match count, junction count, fit radius, …), so a silent behavior
change there fails the run instead of quietly changing the picture.

`docs/assets/birdseye-mosaic.png` is the one exception: it comes from a **real**
2-camera table calibration (`examples/birdseye_mosaic.rs`), not `gen_illustrations`'
synthetic fixtures — the dataset itself lives outside this repo (same private-data policy
as canend/glue-rig), so only the derived PNG is committed:

```bash
WRITE_ASSETS=1 cargo run --release -p vision-metrology --example birdseye_mosaic
```

That run estimates the shared target plane from the two frames (the calibration records no
target pose) and **fails instead of writing the asset** if the two rectified views do not
agree — overlap ZNCC below 0.75. So it self-checks the way `gen_illustrations` does; a
regression that loses the plane cannot quietly republish a broken picture.

## Persistent-context documents

`docs/system-design.md`, `docs/roadmap.md` and `docs/backlog.md` are the project's long-term
memory. Three rules keep them from silting up:

- **Rewrite, don't append.** A superseded decision gets its entry rewritten (saying what
  replaced it), never a second entry contradicting the first.
- **Finished work leaves the roadmap** for `CHANGELOG.md`'s `[Unreleased]` section. The
  roadmap describes what is ahead; the changelog records what landed, with its numbers.
- **Say a thing once.** If a number, rule or decision is already written down somewhere,
  link there. `docs/system-design.md`'s module table, invariant list, and performance table
  are each the single authority for their content.

Invariant numbers are cited by number from source files and are therefore **append-only**;
`tools/check-invariants.py` (also a CI job) verifies contiguity and that every citation
resolves.

## Commits and PRs

- Keep commits scoped and descriptive.
- If behavior changes, adjust tests in the same commit.
- Update the affected README when crate scope or public API changes.
- Do not revert unrelated changes.
