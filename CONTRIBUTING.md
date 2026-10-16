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
| Lowest nalgebra | clippy, test and doc after `cargo update -p nalgebra --precise 0.34.0`, the bottom of the supported range |
| Downstream with calibration-rs | `tools/downstream-smoke/check.sh`: a standalone crate on `vision-metrology` and `vision-calibration-core` must resolve to one nalgebra and run |
| Invariant numbering | `python3 tools/check-invariants.py` |
| Examples | the self-asserting examples listed in `ci.yml` |
| Python bindings | `pip install crates/vm-python`, then `pytest crates/vm-python/tests` and `pytest tools/registration_eval` |
| Cross-platform | build and test on Windows and macOS |
| Lab | frontend typecheck, lint, test and build; desktop crate fmt, clippy and test, which includes the contract replay |

CI does not run the lab backend's pytest; run it locally when touching `lab/backend`.

The other workflows:
- `audit.yml` runs weekly: `cargo audit` and `cargo deny check`, with no ignores. A licence
  missing from `deny.toml`'s allow-list is a deliberate review, not a config fix.
- `bench.yml` runs on demand and writes a criterion table to the job summary. Shared
  runners make its numbers indicative only.
- `python-wheels.yml` runs on every PR and push to `main`: it builds the abi3 wheels for
  Linux x86_64 and aarch64, macOS (universal2) and Windows, and the sdist. It installs each
  wheel into a clean venv on its own platform, under Python 3.10 and 3.13, and runs
  `crates/vm-python/tests/test_readme_quickstart.py`, which executes the READMEs'
  self-contained Python snippets as written. Only a published release uploads to PyPI, over
  a trusted publisher.
- `release-crates.yml` runs on every PR and push to `main`: `cargo package` for
  `vm-primitives` and `vision-metrology` together, which builds each from its packaged
  sources, and `cargo publish --dry-run -p vm-primitives`. A version tag publishes both to
  crates.io over trusted publishing, once the repository variable
  `CRATES_IO_TRUSTED_PUBLISHING` is `true`.
- `publish-docs.yml` publishes the rustdoc to GitHub Pages on every push to `main`.

**MSRV.** It is `rust-version` in the root `Cargo.toml`, explained in
[ADR-0002](docs/dev/adr/0002-dependency-and-toolchain-policy.md). Clippy's
`incompatible_msrv` lint catches `std` items newer than the floor. `rust-toolchain.toml`
pins day-to-day work to stable, and the MSRV job overrides it.

**What a crate ships.** Each library crate's `include` list in its `Cargo.toml` is the
allow-list for crates.io: sources, README, licence texts, benches and, for
`vision-metrology`, the integration tests with fixtures under 64 KiB. The examples stay in
the repository. Check a change with `cargo package -p vm-primitives -p vision-metrology --list`.

**nalgebra.** The workspace accepts `>=0.34, <0.36` (ADR-0002), and `Cargo.lock` stays on
the newest version. Code must compile against both ends: use only nalgebra API that 0.34
has, and run the lowest-nalgebra job's `cargo update` locally when unsure. Do not commit
that lock change.

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
- **Version.** `pyproject.toml` declares `dynamic = ["version"]`, so the wheel takes the
  workspace version from `Cargo.toml`.
- **README snippets run.** `tests/test_readme_quickstart.py` executes every Python block in
  the top-level and the bindings README that starts with an `import`, and a
  `print(...)  # 1.0 2.0` line must print those numbers. Fragments with placeholder names
  start with something else.

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
  - the glue-rig sequence, read from `data/glue-rig` or from `VM_GLUE_RIG_DATA`.

## Benchmarks

Criterion, `harness = false`, one `[[bench]]` per file. IDs follow `operation_size`, and
the representative image size is 1280×1024.

```bash
cargo bench -p vm-primitives --bench downsample   # also: edge1d, edge2d, morph
cargo bench -p vision-metrology --bench match_shape
# also: build_graph, detect_shape, extract, segment, measure, bead, measure_numeric, warp, corr
cargo bench -p vm-primitives --bench downsample -- downsample2x2_to_f32_u8_1280x1024
```

The experimental similarity benches in `corr` need `--features corr-experimental`.

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
- **Private datasets only test the workspace.** Their adapters live in the git-ignored
  `tools/private/` and write a neutral manifest that the tracked tools read
  (`tools/registration_eval/README.md`). Tracked files report private runs only under
  opaque labels and as aggregates. `tools/check-invariants.py` fails if anything under
  `tools/private/` is tracked, and, when the local, git-ignored
  `tools/private/forbidden-terms.txt` exists, if a tracked file contains one of its terms
  (one per line). CI has no such list and skips that check.
- **Public datasets go through the registry**, [`tools/datasets/`](tools/datasets/README.md),
  before anything is evaluated on them: the licence as stated at the source, a fetch with
  checksums into the git-ignored `data/`, and a dev/test split of IDs committed before the
  first run, with settings frozen on dev. Its README holds the intake checklist.
  `tools/check-invariants.py` fails when `docs/performance.md` names a dataset that the
  registry lacks or has not marked `verified`.

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
