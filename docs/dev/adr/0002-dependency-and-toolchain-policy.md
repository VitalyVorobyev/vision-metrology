# ADR-0002: Dependency and toolchain policy

- Status: Accepted
- Date: 2026-10-16

## Context

The library depends on other repositories by the same maintainer (`corrmatch`,
`calibration-rs`), on nalgebra, and on a small set of crates.io dependencies. It has to stay
publishable and its CI reproducible, and its users build it on their own toolchains.

## Decision

- **Cross-repo dependencies are crates.io releases only.** A missing feature is implemented
  upstream, released, and bumped here. Committed code never uses a git or path dependency
  on another repository.
- **nalgebra `>=0.34, <0.36` is a workspace dependency, and linear algebra that nalgebra
  provides is never re-implemented.** The geometry aliases (`Point2f`, `Similarity2f`, …)
  are nalgebra types (ADR-0004), so a consumer must resolve the nalgebra we use. The range
  reaches down to 0.34 because calibration-rs's solver backends pin it, and a crate that
  uses both libraries needs one nalgebra. Code uses only API common to both versions.
- **Both ends of the nalgebra range are tested.** `Cargo.lock` stays on the newest version,
  which every CI job builds. The `nalgebra-min` job runs
  `cargo update -p nalgebra --precise 0.34.0`, then clippy, the tests and the docs.
  The `downstream-smoke` job resolves `tools/downstream-smoke`, a standalone crate that
  depends on `vision-metrology` and on calibration-rs's `vision-calibration-core` from
  crates.io, and fails unless the graph holds one nalgebra and a point passes through
  both public APIs.
- **Cargo does not unify the range by itself.** Semver-incompatible versions may coexist,
  so a fresh lock takes 0.35 for us next to calibration-rs's 0.34. A downstream crate runs
  `cargo update -p nalgebra@0.35.0 --precise 0.34.<n>` once, and its lockfile keeps it;
  the smoke job applies exactly that step.
- **A structured solve nalgebra lacks gets a small routine beside its callers,** in the
  module that owns them. It runs in `f64`, and its tests check it against nalgebra's dense
  solver. The pentadiagonal smoother that `measure` shares (ADR-0018, ADR-0020) is the
  case: nalgebra has no banded solver, and a dense Cholesky costs O(N³) where the band
  costs O(N).
- **The MSRV follows the strictest dependency.** `rust-version` in the root `Cargo.toml`
  is the highest `rust-version` of any dependency, dev-dependencies included, because the
  MSRV CI job builds `--all-targets`; `corrmatch` sets it. Clippy's `incompatible_msrv`
  enforces it for `std` items. Raising it is a deliberate change.
- **The release profile is tuned and benches inherit it:** `lto = "thin"`,
  `codegen-units = 1`, `[profile.bench] inherits = "release"`. It costs only release build
  time, and benches then measure what users ship.

## Alternatives

- **Git or path dependencies on sibling repositories.** Faster to iterate, but they make the
  crates unpublishable and CI non-reproducible.
- **Vendoring an upstream fix.** It forks code that the maintainer owns anyway.
- **A sparse-matrix crate for one banded system.** It adds a dependency, and its
  factorisation builds compressed structures on every call.
- **nalgebra 0.35 only.** Simpler, but a crate that also uses calibration-rs cannot depend
  on us until its solver backends move, and copies code instead.
- **The Python wheel plus versioned JSON as the only bridge to calibration-rs users.** It
  works without a shared nalgebra, but loses the Rust types at the boundary.
- **Dense nalgebra solves for structured systems.** They are correct but cost O(N³) per
  solve, which a per-frame loop cannot afford.

## Consequences

- An upstream gap waits for an upstream release. Examples are corrmatch's `u8`-only API
  (backlog) and calibration-rs's nalgebra version (roadmap, Later).
- An API that exists only in nalgebra 0.35 cannot be used until the range moves up. The
  `nalgebra-min` job catches it. When calibration-rs moves to 0.35, the range drops 0.34
  and both CI jobs go with it.
- A dependency bump can raise the MSRV. The MSRV job catches it, and `rust-version` and the
  job's `MSRV` move together.
