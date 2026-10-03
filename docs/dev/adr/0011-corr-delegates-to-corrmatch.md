# ADR-0011: Cross-correlation delegates to corrmatch

- Status: Accepted
- Date: 2026-08-20

## Context

Some tasks are photometric rather than geometric: tracking a textured window between frames,
or locating a patch with no reliable edges. `corrmatch` (same maintainer, on crates.io)
already implements SIMD ZNCC/SSD with pyramids, rotation and top-k search.

## Decision

- `corrmatch` is a regular, optional dependency behind the default-on `corr` feature, and
  the standard cross-correlation engine of this crate.
- **The wrapper is thin.** It translates coordinates and errors, adapts `ImageView` (copying
  only when the view is strided), and defines its own `CorrConfig`/`CorrTemplateConfig`
  (invariant 10) instead of re-exporting corrmatch's `#[non_exhaustive]` types. `CorrMatch`
  is its own type: its score is a correlation coefficient, not `1 − occluded_fraction`, and
  there is no scale search.
- **`u8` only**, as corrmatch is. `u16`/`f32` support belongs upstream, not in a quantizing
  cast here.
- **`displacement` refines in this crate.**
  - Stage 1 is a bounded corrmatch search around the previous position.
  - Stage 2 is translation-only inverse-compositional Lucas–Kanade on the template's own
    gradient. A parabola fitted to a discrete correlation surface is biased towards integer
    positions, and only a second, differently biased estimator removes that bias.

## Alternatives

- **A native ZNCC port**, to keep the shape matcher's validator independent of what it
  validates. Rejected: it would duplicate mature SIMD code with a worse copy. The cross-check
  (`tests/corrmatch_bridge.rs`) is about two different algorithms agreeing, not about
  build-graph independence.
- **Dev-dependency only.** That denies users the tool.

## Consequences

- A corrmatch release can raise this crate's MSRV (ADR-0002).
- Scale search and non-`u8` data wait on upstream (backlog).
