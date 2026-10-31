# ADR-0023: The `filter` module: exact linear filters, built in-house

- Status: Accepted
- Date: 2026-10-31

## Context

The module needs separable Gaussian, box and binomial smoothing that feeds the pyramid
pre-smooth and the derived filters. Two questions come before code:

- Is there a Rust crate to build on? The strongest pure-Rust option is libblur (0.24.0). It
  has SIMD box and Gaussian paths, and an approximate "fast" Gaussian. It targets 8-bit and
  RGB image processing, has no accuracy contract for its approximate paths, limits the
  radius, and brings its own image types.
- Can we beat it? Speed alone is a weak claim where libblur is already fast. The metrology
  case needs exact behaviour: a stated accuracy, integer results that match a reference bit
  for bit, and the same output on every platform.

Invariant 5 rules out OpenCV and other FFI. [ADR-0016](0016-scope-what-we-do-not-build.md)
limits the module to Gaussian, box, binomial, median and grayscale morphology.

## Decision

1. **Build in-house. No blur crate is a dependency of any library crate.** libblur and
   OpenCV are comparators only. `tools/filter_bench` pins libblur in its own workspace.
2. **Accuracy is stated per path and checked against an independent oracle.**
   - Integer paths (box and binomial on `u8`) are bit-exact against a naive integer
     reference, with round-half-up of `sum / (2r+1)²`.
   - Float paths are checked against an f64 naive separable reference with the same
     truncation and `Clamp` borders. The bound is a few ulp of the storage type, not
     bit-identity, because summation order differs.
   - The oracle is checked against scipy (`tools/filter_bench/reference.py` writes golden
     files that the Rust tests read).
3. **Gaussian by σ, with the method chosen by measurement.** An exact folded FIR is the
   baseline. Above a crossover σ, a cascade of three extended boxes is used, with a stated
   error budget. The crossover is σ = 8, measured on the M4 Pro.
4. **Runtime dispatch with `multiversion`, no hand-written SIMD.** The kernels are slice
   loops that the compiler vectorises; `multiversion` compiles them for the CPU levels of the
   target and selects at run time. `pulp` would need explicit generic SIMD code and more
   per-ISA tests. No `unsafe` (invariant 7).
5. **The exact float Gaussian is not a speed claim.** Its parity with libblur is measured,
   and on x86_64 it lags (a stated lag). The FIR is kept as the reference, and it is not
   public beyond `gaussian_f32`.
6. **Speed claims need measurement on both targets, single-threaded, with load recorded.**
   Numbers are in [performance](../../performance.md#linear-filters), not here.

## Alternatives

- **Depend on libblur.** Fast, but it brings its own image and edge types, an approximate
  Gaussian with no error contract, a box whose `u8` output differs from the exact rounded
  mean by up to 1 LSB (so it cannot be the bit-exact reference), and a dependency in every
  library consumer.
- **OpenCV via FFI.** Excluded by invariant 5.
- **Naive loops only.** Exact, but about 10× slower at the pyramid's σ, which fails the
  hot-path rule (invariant 6).
- **Hand-written intrinsics.** Fastest, but one `unsafe` block per ISA and a per-ISA test
  matrix (invariant 7). Deferred until a measured need exists.
- **Recursive Gaussian (Deriche, van Vliet).** Not built. The measured three-box cascade
  meets the large-σ need within its error budget, and a recursive filter adds a second
  Gaussian method with its own border initialisation.

## Consequences

- The integer box divides with a float reciprocal and an integer correction, exact for
  windows up to `(2·1023+1)²`. Binomial divides by `4^order` with a shift.
- The cascade has a 5 grey-level budget against the untruncated Gaussian on the stress
  fixture (step 160, noise ±64), for σ from 2 to 8. Below σ = 2 the three-box approximation is
  coarse, so `Auto` does not use it there. The budget is a decision for review.
- The cascade is latency-bound (a serial running sum per row) and costs the same at every σ.
  Vectorising across rows is the next step if large-σ speed matters.
- `tools/filter_bench` stays unpublished as the comparator. `filter-bench.yml` runs it on
  x86_64 on demand.
- The public API takes and returns images, each with a `BorderMode` and a `Result`. Scratch
  is internal; a reusable-scratch object is open work (see the backlog).
- Consumers: the pyramid pre-smooth uses the FIR. `gradient::smooth_binomial3`, `edge1d` and
  `conv1d` keep their own paths, because their locked expression trees and numbers depend on
  them. Rewiring one changes locked results and needs a measured benefit first.
- The roadmap's acceptance criterion for B3 is amended to match: bit-exact integer paths, and
  a stated few-ulp bound for float paths.
