# System design

The architecture of `vision-metrology` as it is: layering, invariants, and the index of
design decisions. With [`roadmap.md`](roadmap.md) (what is next) and
[`backlog.md`](backlog.md) (known debt) it is the starting context for any work session.

## Layering

```
vm-primitives  ──►  vision-metrology  ──►  vm-python
(low-level)         (domain algorithms)    (PyO3 bindings, wheel only)

lab/  (FastAPI + React + Tauri, outside the Cargo workspace; uses vm-python and vision-metrology)
```

- **`vm-primitives`** holds image containers, sampling, geometry, pyramids, 1-D/2-D edges
  and binary morphology. Its module table is in
  [`crates/vm-primitives/README.md`](../../crates/vm-primitives/README.md).
- **`vision-metrology`** holds the domain modules, each a default-on Cargo feature. Its
  module and feature tables are in
  [`crates/vision-metrology/README.md`](../../crates/vision-metrology/README.md). These two
  READMEs are the crate-level rustdoc, and they are the only module tables in the repo.
- **`vm-python`** is numpy in, numpy out. Its coverage table is in
  [`crates/vm-python/README.md`](../../crates/vm-python/README.md).
- **`lab/`** is described in [`lab/ARCHITECTURE.md`](../../lab/ARCHITECTURE.md) and
  [ADR-0015](adr/0015-the-lab.md).

Dependencies point one way. A name lives at its module path; each library crate has a
curated `prelude`, and crate-root re-exports are explicit lists.

## The measurement chain

```
acquire → [metric: undistort / rectify] → matching: locate → pose as fixture
        → measure: calipers → fit: primitives + residuals → metric: millimetres → pass / fail
```

`warp` does the resampling; `corr` and `scale` assist the locator. `contour`, `segment`,
`lsd` and `laser` are standalone extractors. `examples/inspect_canend.rs` runs the chain end
to end on real frames; its reference numbers are in
[`docs/performance.md`](../performance.md).

## Invariants

Breaking an invariant is a design change: it needs an ADR, or an update to an existing one.

**The numbering is an API.** Source files cite invariants by number, so numbers are
append-only: never renumber, never reuse. A retired invariant keeps its number with a
`(retired)` note naming its replacement. `tools/check-invariants.py` (a CI job) checks that
the list is contiguous and that every citation resolves.

1. **Pixel centres.** Integer coordinate `i` means position `i as f32`. Edgels, ROIs,
   poses and the pyramid mapping all assume it.
2. **Pyramid coordinate mapping.** The level-`l` coordinate of a level-0 point is
   `L_l(p) = (p − (2^l − 1)/2) / 2^l`; candidate propagation is `q_l = 2·q_{l+1} + 0.5`.
   `pyr::level_to_base` / `base_to_level` are the only implementation; nothing re-derives
   it inline.
3. **Model and scene alias the same way.** A shape model's level-`l` points come from
   running the edge detector on level `l` of the reference ROI's own pyramid, never from
   decimating level-0 points. Model build and scene search use the same downsample kernel.
4. **Score semantics.** The shape-matching score divides by the full model point count
   `n`, never by the contributing count, so `score ≈ 1 − occluded_fraction`. Model points
   are decimated spatially uniformly for the same reason.
5. **Rust-native.** No OpenCV and no FFI in the library crates.
6. **Hot paths are allocation-free per call.** Detectors own reusable scratch; the only
   per-call allocation allowed is the output container.
7. **`unsafe` policy.** Only small, justified blocks, each with a `// SAFETY:` comment.
   A guarding `assert!` and the block it protects move together in any refactor.
   `unsafe_op_in_unsafe_fn` is denied workspace-wide.
8. **Error type.** `vm_primitives::Error` everywhere, with `&'static str` payloads only.
9. **`'static` public outputs.** No lifetimes in public result types (PyO3 compatibility).
10. **Config struct plus reusable detector, and no sentinel values.** A config is a plain
    `pub` struct with `Default`, built with `..Default::default()`. A detector owns its
    scratch and is reused across calls. "Absent", "automatic" and "unlimited" are written in
    the type (`Option<T>`, `Option<NonZeroUsize>`, a named enum), never as `0`, `0.0` or an
    empty range. A config past about 8 fields splits its effort fields into a nested
    `tuning` struct, and thresholds that depend on the pixel type carry a unit type
    ([ADR-0007](adr/0007-configs-say-what-they-mean.md)).
11. **The default border mode is `Clamp`** in core and edge, unless configured otherwise.
12. **Determinism.** No RNG in library code. Tests use synthetic fixtures, seeded if
    randomness is unavoidable. f32 sort ties are broken explicitly, e.g. `(−score, x, y)`.
13. **Toolchain.** Edition 2024; the MSRV is `rust-version` in the root `Cargo.toml`;
    nalgebra 0.35 is the workspace dependency. Linear algebra is never re-implemented
    ([ADR-0002](adr/0002-dependency-and-toolchain-policy.md)).
14. **File size.** Soft cap of about 600 code lines per source file (tests excluded).
    Crossing it means splitting in the same change. Known offenders are listed in
    `backlog.md`, and `tools/check-invariants.py` fails on any other file over 600.
15. **vm-python parity.** A change that adds public Rust API updates the bindings, the
    `.pyi` stubs and a Python test in the same PR, unless the vm-python README lists the
    item as not bound.
16. **Docs as memory.** A change to scope, decisions or invariants updates
    `docs/dev/system-design.md` (and the ADR it touches), `roadmap.md` or `backlog.md` in
    the same PR, rewriting the affected entry rather than appending to it.
17. **One canonical path per name.** No glob re-exports across crate boundaries. `prelude`
    is a curated convenience, and any crate-root re-export is an explicit list.
    `vision-metrology` re-exports the `vm_primitives` crate, not its contents.
18. **Every domain module is feature-gated**, default-on. A new module ships with its
    feature, `required-features` on every example, bench and test that uses it, and a row
    in the crate's feature table. CI checks each feature on its own.
19. **One entry point per algorithm**, generic over `Pixel`. No `_u8`/`_u16`/`_f32` variants
    of the same operation. A `_f32` suffix marks something that only takes `f32`.
20. **Storage in f32, accumulation in f64.** Pixel coordinates are stored as `f32`; every
    normal-equation, moment or residual sum runs in `f64`.
21. **Every measurement reports its residual.** A fitting or measuring operation returns the
    statistics that qualify it (`rms`, `max_dev`, points used), not just the parameters.

## Decisions

One file per decision in [`adr/`](adr/). Each has a Status and a Date, then the sections
Context, Decision, Alternatives and Consequences. An ADR states the decision as it stands:
no change log, and no measured numbers, which live in
[`docs/performance.md`](../performance.md). When a decision changes, rewrite its ADR and
update its status and date. Do not append a contradicting one.

| ADR | Decision |
|---|---|
| [0001](adr/0001-shape-matching-by-gradient-orientation.md) | Locate parts by gradient-orientation shape matching |
| [0002](adr/0002-dependency-and-toolchain-policy.md) | Dependency and toolchain policy (crates.io only, MSRV, nalgebra, release profile) |
| [0003](adr/0003-metric-calibration-bridge.md) | `metric`: the calibration bridge from pixels to millimetres |
| [0004](adr/0004-core-raster-and-geometry.md) | `core` is a nalgebra-free raster half plus a nalgebra geometry half |
| [0005](adr/0005-stored-model-format.md) | The stored shape model is opaque, read-only and versioned |
| [0006](adr/0006-absence-is-a-result.md) | A measurement that found nothing is a result |
| [0007](adr/0007-configs-say-what-they-mean.md) | Configs say what they mean |
| [0008](adr/0008-calipers.md) | Calipers and their placements |
| [0009](adr/0009-robust-fitting.md) | Robust fitting: algebraic start, geometric refinement, reported residuals |
| [0010](adr/0010-warp-and-rectified-crops.md) | `warp` gathers from the source; rectified crops are sized by their spec |
| [0011](adr/0011-corr-delegates-to-corrmatch.md) | Cross-correlation delegates to corrmatch |
| [0012](adr/0012-mosaic-compositing.md) | Mosaics pick one camera per pixel and never blend |
| [0013](adr/0013-scale-invariance.md) | Scale invariance is estimate-then-verify |
| [0014](adr/0014-masked-teaching-and-reference-angle.md) | Masked teaching and the model's reference angle |
| [0015](adr/0015-the-lab.md) | The lab: one frontend, two transports, shared UI packages |
| [0016](adr/0016-scope-what-we-do-not-build.md) | Scope: what this library deliberately does not build |
| [0017](adr/0017-textbook-edge-location.md) | Textbook edge location and CaliperBench compatibility |
