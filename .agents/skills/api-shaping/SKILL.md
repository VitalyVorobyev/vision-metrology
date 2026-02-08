---

name: api-shaping
description: Use this when designing or refactoring public APIs across vm-* crates. Keeps APIs small, explicit, and stable while allowing fast internals.
---------------------------------------------------------------------------------------------------------------------------------------------------------

# API shaping (lightweight)

## Aim

* Small public surface
* Clear data ownership (views vs owned)
* Fast internals without leaking complexity

## Prefer

* `Detector` structs that own scratch buffers → avoid allocations per call.
* `Config` structs with safe defaults, but don’t hide “magic” thresholds.
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
