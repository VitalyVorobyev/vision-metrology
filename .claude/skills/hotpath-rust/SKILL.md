---

name: hotpath-rust
description: Use this when writing performance-critical inner loops (downsample, convolution, row/col scanning). Keeps changes safe, fast, and benchmarked.
-----------------------------------------------------------------------------------------------------------------------------------------------------------

# Hotpath Rust checklist (lightweight)

## Goals

* predictable performance
* minimal unsafe
* no per-row/per-col allocations

## Do

* Reuse scratch buffers stored on a struct (`Vec<T>` reused via `clear()` / `resize()`).
* Separate:

  * **safe reference implementation**
  * **fast path** (contiguous / common border mode) with minimal branching
* If using `unsafe`, document invariants right above the block.

## Don’t

* Allocate inside the scan loop.
* Add new heavy deps “for convenience”.

## Always add

* 1–2 unit tests that pin behavior (edge cases included).
* A Criterion bench for the hot function (representative size).
