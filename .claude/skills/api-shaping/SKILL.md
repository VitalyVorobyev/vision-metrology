---
name: api-shaping
description: Use this when designing or refactoring public APIs across the workspace crates. Keeps APIs small, explicit, and stable while allowing fast internals.
---

# API shaping (lightweight)

## Aim

* Small public surface
* Clear data ownership (views vs owned)
* Fast internals without leaking complexity

## Binding rules

* Invariants 9, 10, 17 and 19 in `docs/dev/system-design.md`: `'static` public outputs,
  config struct plus reusable detector with no sentinels, one canonical path per name, one
  entry point per algorithm.
* ADR-0006 (a measurement that found nothing is a result) and ADR-0007 (configs say what
  they mean), in `docs/dev/adr/`.

## Prefer

* `ImageView<T>` / `ImageViewMut<T>` in APIs; keep crates buffer-agnostic.
* Separate “core algorithm” from “pipeline convenience wrapper”.

## Avoid

* Generic abstractions that obscure hot loops.
* Exposing internal scratch buffers in public API.
* “One mega function” that does everything.

## Patterns that work here

* `detect_*(&mut self, img: &ImageView<_>, cfg: &Config) -> Output`
* `Output` types that can be iterated cheaply (`Vec<Edgel>`, `Vec<LaserSample>`)
* Optional features for parallelism/SIMD later, not in baseline.
