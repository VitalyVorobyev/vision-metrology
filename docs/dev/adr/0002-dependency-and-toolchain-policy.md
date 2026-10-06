# ADR-0002: Dependency and toolchain policy

- Status: Accepted
- Date: 2026-10-06

## Context

The library depends on other repositories by the same maintainer (`corrmatch`,
`calibration-rs`), on nalgebra, and on a small set of crates.io dependencies. It has to stay
publishable and its CI reproducible, and its users build it on their own toolchains.

## Decision

- **Cross-repo dependencies are crates.io releases only.** A missing feature is implemented
  upstream, released, and bumped here. Committed code never uses a git or path dependency
  on another repository.
- **nalgebra 0.35 is a workspace dependency, and linear algebra that nalgebra provides is
  never re-implemented.** The geometry aliases (`Point2f`, `Similarity2f`, …) are nalgebra
  types (ADR-0004).
- **A structured solve nalgebra lacks gets a small routine beside its only caller.** It runs
  in `f64`, and its tests check it against nalgebra's dense solver. The bead tracker's
  pentadiagonal system (ADR-0018) is the case: nalgebra has no banded solver, and a dense
  Cholesky costs O(N³) where the band costs O(N).
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
- **Dense nalgebra solves for structured systems.** They are correct but cost O(N³) per
  solve, which a per-frame loop cannot afford.

## Consequences

- An upstream gap waits for an upstream release. Examples are corrmatch's `u8`-only API
  (backlog) and calibration-rs's nalgebra version (roadmap, Later).
- A dependency bump can raise the MSRV. The MSRV job catches it, and `rust-version` and the
  job's `MSRV` move together.
