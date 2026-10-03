# ADR-0002: Dependency and toolchain policy

- Status: Accepted
- Date: 2026-08-20

## Context

The library depends on other repositories by the same maintainer (`corrmatch`,
`calibration-rs`), on nalgebra, and on a small set of crates.io dependencies. It has to stay
publishable and its CI reproducible, and its users build it on their own toolchains.

## Decision

- **Cross-repo dependencies are crates.io releases only.** A missing feature is implemented
  upstream, released, and bumped here. Committed code never uses a git or path dependency.
- **nalgebra 0.35 is a workspace dependency, and linear algebra is never re-implemented.**
  The geometry aliases (`Point2f`, `Similarity2f`, …) are nalgebra types (ADR-0004).
- **The MSRV follows the strictest dependency.** It is the highest `rust-version` of any
  dependency, dev-dependencies included, because the MSRV CI job builds `--all-targets`.
  It is currently 1.91, set by `corrmatch`. Clippy's `incompatible_msrv` enforces it for `std`
  items. Raising it is a deliberate change recorded here.
- **The release profile is tuned and benches inherit it:** `lto = "thin"`,
  `codegen-units = 1`, `[profile.bench] inherits = "release"`. Measured gain is about 2% on
  shape search, at no cost beyond release build time, and benches then measure what users
  ship.

## Alternatives

- **Git or path dependencies on sibling repositories.** Faster to iterate, but they make the
  crates unpublishable and CI non-reproducible.
- **Vendoring an upstream fix.** It forks code that the maintainer owns anyway.

## Consequences

- An upstream gap waits for an upstream release. Examples are corrmatch's `u8`-only API and
  calibration-rs's nalgebra version, both in the backlog.
- A dependency bump can raise the MSRV. The MSRV job catches it, and this ADR is updated
  with the new floor.
