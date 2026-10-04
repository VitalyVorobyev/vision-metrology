# Contributing

Development workflow for the `vision-metrology` workspace. User documentation is in
[README.md](README.md), the crate READMEs and [`docs/`](docs). Design context for
contributors is in [`docs/dev/`](docs/dev):
- [system design](docs/dev/system-design.md): layering, invariants, the ADR index;
- [roadmap](docs/dev/roadmap.md);
- [backlog](docs/dev/backlog.md).

## Quality gates

Run from the workspace root before every commit:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace --all-features
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --all-features --no-deps
python3 tools/check-invariants.py
```

CI runs these and more:

| Job | What it runs |
|---|---|
| Rust quality | fmt, clippy, test, doc |
| Feature matrix | `cargo hack clippy` over each `vision-metrology` feature and the `vm-primitives` feature powerset |
| MSRV | `cargo check --workspace --all-targets --all-features` on the `rust-version` toolchain of the root `Cargo.toml` |
| Invariant numbering | `python3 tools/check-invariants.py` |
| Examples | the self-asserting examples listed in `ci.yml` |
| Python bindings | `pip install crates/vm-python`, then `pytest crates/vm-python/tests` |
| Cross-platform | build and test on Windows and macOS |
| Lab | frontend typecheck, lint, test and build; desktop crate fmt, clippy and test, which includes the contract replay |

CI does not run the lab backend's pytest; run it locally when touching `lab/backend`.

The other workflows:
- `audit.yml` runs weekly: `cargo audit` and `cargo deny check`, with no ignores. A licence
  missing from `deny.toml`'s allow-list is a deliberate review, not a config fix.
- `bench.yml` runs on demand and writes a criterion table to the job summary. Shared
  runners make its numbers indicative only.
- `python-wheels.yml` runs on a published release, or by hand: it builds the wheels and the
  sdist, and on a release publishes them to PyPI.
- `publish-docs.yml` publishes the rustdoc to GitHub Pages on every push to `main`.

**MSRV.** It is `rust-version` in the root `Cargo.toml`, explained in
[ADR-0002](docs/dev/adr/0002-dependency-and-toolchain-policy.md). Clippy's
`incompatible_msrv` lint catches `std` items newer than the floor. `rust-toolchain.toml`
pins day-to-day work to stable, and the MSRV job overrides it.

## Python bindings

```bash
cd crates/vm-python
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release
pytest tests/
```

- **Names.** The Rust lib target is named `vm_python`, not `vision_metrology`, so it
  cannot collide with the `vision-metrology` crate's lib target. The importable name
  `vision_metrology` comes from the `#[pymodule]` function and from `module-name` in
  `pyproject.toml`. The crate sets `doctest = false`, because its examples are Python.
- **Stubs.** `python/vision_metrology/__init__.pyi` is maintained by hand. Update it in the
  same change as the `#[pymodule]` registration list. A test checks that every stubbed name
  exists at runtime.
- **Parity** is [invariant 15](docs/dev/system-design.md#invariants).

## Tests

- **Placement.** Unit tests are inline (`#[cfg(test)] mod tests`). Integration tests live
  in `crates/*/tests/`, including the accuracy suite (`tests/accuracy.rs`) and the
  cross-algorithm checks.
- **Fixtures are deterministic synthetic images with known geometry**, anti-aliased when
  subpixel accuracy is asserted. There is no unseeded RNG.
- **Assertions state the expected geometry and the tolerance:**

  ```rust
  assert!(err < 0.01, "sub-pixel residual expected, got err={err}");
  ```

- **The accuracy suite pins envelopes** at about 1.5× the measured worst case. A new
  operator adds a row, and its numbers go into
  [`docs/performance.md`](docs/performance.md).
- **The CaliperBench golden fixture** (`tests/fixtures/caliperbench_golden.json`) holds
  what CaliperBench's own baselines return on small inline images, and
  `tests/caliperbench_protocol.rs` checks `examples/caliperbench_run.rs` against it. It is
  generated, not edited. Regenerate it when CaliperBench's baselines change, with
  CaliperBench's environment:

  ```bash
  uv run --directory /path/to/caliperbench python "$PWD/tools/gen_caliperbench_golden.py"
  ```

  The script refuses a case whose outcome hangs on a near-tie, because `f32` and
  `float64` could decide it differently.
- **Doctests are API smoke tests.** The two crate READMEs are the crate-level rustdoc
  (`#![doc = include_str!("../README.md")]`), so their examples compile and run.
- **Private datasets.** Tests that need one skip with a message when it is absent. The
  datasets are not distributed:
  - the can-end frames used by `inspect_canend`, `pose_audit` and the lab's folder test;
  - the glue-rig sequence.

## Benchmarks

Criterion, `harness = false`, one `[[bench]]` per file. IDs follow `operation_size`, and
the representative image size is 1280×1024.

```bash
cargo bench -p vm-primitives --bench downsample   # also: edge1d, edge2d, morph
cargo bench -p vision-metrology --bench match_shape
# also: build_graph, detect_shape, extract, segment, measure, warp, corr
cargo bench -p vm-primitives --bench downsample -- downsample2x2_to_f32_u8_1280x1024
```

Add a benchmark when you add or change a hot path, and put before/after numbers in the PR
description. Published numbers live in [`docs/performance.md`](docs/performance.md);
update it when a change moves them.

## Documentation

- **Audience decides location.**
  - User-facing: `README.md`, crate READMEs, rustdoc (`//!`, `///`), `docs/*.md`,
    `CHANGELOG.md`, `lab/README.md`.
  - Contributor-facing: this file, `AGENTS.md`, `CLAUDE.md`, `docs/dev/`,
    `lab/ARCHITECTURE.md`, `lab/contract/README.md`, `.claude/skills/`, plain `//`
    comments, and test-file headers.

  User-facing text never links into `docs/dev/` and never mentions plan labels, PR numbers
  or history. Invariant citations go in `//` comments.
- **State what is true now.** History belongs in `CHANGELOG.md` and nowhere else.
- **Say it once.** Each fact has one home, and other places link to it:
  - module tables: the crate READMEs;
  - invariants: `docs/dev/system-design.md`;
  - numbers: `docs/performance.md`.
- **Decisions are ADRs**, written to the rules in system design's
  [Decisions](docs/dev/system-design.md#decisions) section.
- **Finished roadmap items leave `docs/dev/roadmap.md`** for `CHANGELOG.md` `[Unreleased]`.
- **Invariant numbering** is append-only
  ([system design](docs/dev/system-design.md#invariants)). `tools/check-invariants.py`
  checks that every citation resolves, and rejects plan labels outside the roadmap and the
  changelog, links into `docs/dev/` from user-facing files, and one external project name.
  It also enforces the size cap of invariant 14 against the list in `backlog.md`.

### Illustrations

The PNGs in `docs/assets/` are rendered deterministically from synthetic fixtures:

```bash
cargo run --release --example gen_illustrations --all-features
```

Re-run it and commit the results when a change alters what an illustration shows. The
renderer asserts its own fixtures, so a behaviour change fails the run instead of quietly
changing a picture.

`docs/assets/birdseye-mosaic.png` comes from a real two-camera calibration, which is not
distributed:

```bash
WRITE_ASSETS=1 cargo run --release -p vision-metrology --example birdseye_mosaic -- \
  --data-dir <dataset>
```

It estimates the target plane from the two frames and refuses to write the asset unless
the rectified views agree (overlap ZNCC at least 0.75).

## Commits and pull requests

- Write descriptive commit messages, and adjust tests in the same commit as a behaviour
  change.
- Docs move with the code: [invariant 16](docs/dev/system-design.md#invariants).
- User-visible changes get a line in `CHANGELOG.md` under `[Unreleased]`.
