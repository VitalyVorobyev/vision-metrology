# ADR-0023: The `filter` module: exact linear filters, built in-house, gated on measured headroom

- Status: Accepted; decisions 3 and 4 amended in Phase 2 (see below)
- Date: 2026-10-31

## Context

Roadmap item B3 asks for separable and recursive Gaussian, a sliding-window box mean, an
O(1)-per-radius median and grayscale morphology, and folds in `edge/conv1d.rs`. The module
feeds the pyramid pre-smooth and illumination correction. Two questions come before code:

- Is there a Rust crate to build on? The strongest pure-Rust option is libblur
  (0.24.0). It has SIMD paths for box, Gaussian and an approximate "fast" Gaussian. It is
  built for 8-bit and RGB image processing. It has no accuracy contract for the
  approximate paths, a radius limit, and its own image types.
- Can we beat it? Speed alone is a weak claim on a machine where libblur is already fast.
  The metrology case needs the exact behaviour: a stated accuracy, integer results that
  match a reference bit for bit, and the same output on every platform.

Invariant 5 (no OpenCV, no FFI in library crates) rules out the common C++ options. The
scope ADR ([ADR-0016](0016-scope-what-we-do-not-build.md)) limits the module to Gaussian,
box, median and grayscale morphology.

## Decision

1. **Build in-house. No blur crate is a dependency of any library crate.** libblur and
   OpenCV are comparators only: `tools/filter_bench` pins libblur 0.24.0 in its own
   workspace, outside the root workspace and its lockfile.
2. **Accuracy is stated per path, and checked against an independent oracle.**
   - Integer paths (box, binomial on `u8`/`u16`) are bit-exact against the naive integer
     reference, which uses round-half-up of `(sum)/(2r+1)^2`.
   - Float paths are checked against an f64 naive separable reference with the same
     truncation and `Clamp` borders. The bound is a few ulp of the storage type plus the
     accumulation order, not bit-identical: the roadmap's "bit-for-bit" wording for float
     paths is amended in this PR.
   - The oracle itself is checked against scipy (`tools/filter_bench/reference.py`
     writes golden files; the Rust tests compare against them).
3. *(Amended in Phase 2, below.)* **Gaussian by sigma, with the method chosen by measurement.** The exact folded FIR,
   register-blocked over chunks of output columns, is the baseline for every sigma. A
   large-sigma method (a recursive filter or a cascade of boxes) is added only if it beats
   the FIR at that sigma and stays inside a measured error budget against the oracle. Its
   threshold is set by measurement in Phase 3, not guessed here.
4. **SIMD: safe code shaped for autovectorisation, register blocking, runtime dispatch.**
   The dispatch crate is now chosen (Phase 2, below).
   No `unsafe` (invariant 7). The choice between `pulp` and `multiversion` for dispatch is
   deferred to Phase 2 and recorded there with a measurement.
5. **Speed claims require measurement on both targets.** No claim over libblur is made
   until it is measured on macOS (aarch64) and Linux (x86_64), single-threaded and
   multi-threaded, with the machine load recorded. The Linux measurement is in the section
   below; multi-threaded runs are not measured yet.

## Phase 0 gate: what was measured

Apple M4 Pro, `tools/filter_bench`, a 1280x1024 image, best of 25 runs after 3 warm-up,
single thread unless noted. Load average was about 40 during the runs (shared machine).
Two independent runs agreed within about 4% on the rows below. Full output:
[`tools/filter_bench/README.md`](../../../tools/filter_bench/README.md).

| case | prototype | libblur (1 thread) | libblur / prototype | error vs f64 oracle |
|---|---|---|---|---|
| box u8, r = 1, 2, 4, 8 (exact) | 1.40 to 1.42 ms | 0.87 to 0.91 ms | 0.62 to 0.65x | ours 0; libblur up to 1 LSB |
| Gaussian f32, sigma = 1 | 0.85 ms | 0.69 ms | 0.81x | 5.7e-5 vs 4.6e-5 |
| Gaussian f32, sigma = 2 | 1.33 ms | 1.21 ms | 0.92x | 5.1e-5 vs 6.5e-5 |
| Gaussian f32, sigma = 4 | 2.13 ms | 2.43 ms | 1.14x | 5.9e-5 vs 6.0e-5 |
| Gaussian f32, sigma = 6 | 3.52 ms | 3.89 ms | 1.10x | 7.6e-5 vs 5.5e-5 |
| libblur `fast_gaussian` u8, approximate (context) | n/a | 2.05 to 2.12 ms | n/a | not measured |

What the numbers say:

- **The exact Gaussian FIR has no speed headroom over libblur.** It is at parity (0.81x to
  1.14x). The first, non-blocked prototype was 2.9x slower at sigma = 6 (10.1 ms), so the
  register blocking is what closes the gap. The FIR is not where a speed win lies.
- **The box path is slower (0.62x to 0.65x) and this is open.** The likely cause is the
  64-bit reciprocal in the final division, which does not vectorise on NEON, and the
  per-element clamp when the row is padded. Phase 2 measures this before it claims anything.
- **libblur's box is not exact.** Its `u8` output differs from the rounded integer mean by
  up to 1 LSB. This is the one place where exactness is a real, measured difference.
- **Float accuracy is a tie.** Both implementations are within 1e-4 of the f64 oracle on
  0 to 255 data. f32 storage sets that limit. Accumulation in f64 was not measured here.
- **The large-sigma target is set by the approximation, not the exact FIR.** libblur's
  approximate Gaussian is about 2 ms regardless of radius (O(1) per pixel). An exact
  large-sigma method has to be within a stated error of the oracle and beat the exact
  FIR, which costs 3.5 ms at sigma = 6.

**Gate result:** proceed to Phase 1 on the reframed goal. The module is justified by
exactness, stated accuracy and a single engine for derived uses. It is not justified by a
speed lead over libblur on the exact Gaussian, which the measurements do not support.
Speed targets for Phase 2 and 3 are: box at parity with libblur on both targets (NEON is
0.62x; x86_64 is already ahead), and a large-sigma path that beats the exact FIR at its crossover sigma while
staying within its error budget.

## Linux x86_64 measurement

Run in CI on `ubuntu-latest` (rustc 1.99.1), with the same bench, single thread, best of 25.
Two matrix entries: the default `x86_64` target (SSE2 only), and `target-cpu=x86-64-v3`
(AVX2 and FMA). The fused variant uses `f32::mul_add` in the tap update. Each cell is
libblur's time divided by the prototype's, so above 1 means the prototype is faster. The
full logs are in the `filter-bench` workflow artifacts.

| case | default target | x86-64-v3 | x86-64-v3, fused |
|---|---|---|---|
| box u8, r = 4 (exact) | 1.89x | 1.30x | not run |
| Gaussian f32, sigma = 1 | 0.47x | 0.57x | 0.60x |
| Gaussian f32, sigma = 2 | 0.52x | 0.74x | 0.78x |
| Gaussian f32, sigma = 4 | 0.56x | 0.84x | 0.91x |
| Gaussian f32, sigma = 6 | 0.67x | 0.90x | 0.99x |

What the Linux numbers change:

- **The Gaussian is not at parity on x86_64.** It is behind by 0.47x to 0.67x on the default
  target and by 0.57x to 0.99x with AVX2 and FMA. On the M4 it was at parity. The gap is
  wider at small sigma, and its cause is not identified.
- **Fusing the tap update helps only a little** (at most 10%) once FMA is enabled. Without
  hardware FMA, `mul_add` becomes a library call and is 10 to 20 times slower (22 to
  130 ms). So a fused kernel may only run where the target has FMA, which is a dispatch
  requirement, not a flag.
- **The box goes the other way.** On x86_64 the prototype is faster than libblur (1.3x to
  1.9x). On the M4 it was slower (0.62x). libblur's box is target-dependent; the prototype's
  is not, so the box gap is specific to NEON.
- **The runner is noisy.** The prototype's box time moved from 1.77 ms to 2.48 ms across three
  runs of the same code. The ratios within one run are the usable figures, and even they
  moved by up to 0.05 between runs.

## Phase 2 amendment

Phase 2 covers the exact integer paths: box mean and binomial smoothing on `u8`, in
`vm-primitives::filter::integer`. Two decisions change, both on the Phase 0 numbers.

- **The float Gaussian is out of the in-house scope for now.** The Phase 0 gate found no
  speed headroom over libblur on the exact Gaussian, and the x86_64 default target lags it
  (0.47x to 0.67x). Accepted as a stated lag. The crate-internal FIR from Phase 1 stays as
  the reference for the `f32` tests and is not public. Where a consumer needs a float
  Gaussian, the choice (a libblur backend, or a narrower in-house method) is made in
  Phase 4 with its own decision record.
- **Dispatch uses `multiversion` 0.9, not `pulp`.** The kernels are slice loops that the
  compiler vectorises. `multiversion` compiles each one for the CPU levels of the target and
  picks at run time; it needs no hand-written SIMD. `pulp` would need explicit generic SIMD
  code, which is more `unsafe`-adjacent surface and more code to test per ISA.

Phase 2 measurements, Apple M4 Pro (NEON), 1280x1024 `u8`, best of 25, load about 33. The
libblur column was measured in the same session as the box:

| case | integer kernel | libblur (1 thread) | libblur / kernel |
|---|---|---|---|
| box r = 1 | 0.63 ms | 0.83 ms | 1.3x |
| box r = 4 | 0.97 ms | 0.91 ms | 0.93x |
| box r = 8 | 1.06 ms | 0.84 ms | 0.79x |
| binomial order 2, 4, 6 | 0.63, 0.87, 1.11 ms | n/a | n/a |

What the numbers say:

- **Box at small radius now beats libblur on NEON.** Direct tap sums in the horizontal pass
  vectorise; the Phase 0 prototype was 0.62x. Wider windows switch to a running sum, which is
  sequential, and that path is the open gap: 0.79x at r = 8. Parity on both targets is not
  met yet for r of 8 or more.
- **x86_64 (GitHub `ubuntu-latest`, same run as libblur, single thread, best of 25).** The
  integer kernel against libblur's box, as libblur / kernel:

  | case | default target: kernel / libblur | x86-64-v3: kernel / libblur |
  |---|---|---|
  | box r = 1 | 2.05 ms / 4.70 ms = 2.3x | 1.79 ms / 2.02 ms = 1.13x |
  | box r = 4 | 2.60 ms / 4.80 ms = 1.8x | 2.27 ms / 2.09 ms = 0.92x |
  | box r = 8 | 2.61 ms / 4.71 ms = 1.8x | 2.25 ms / 2.05 ms = 0.91x |
  | binomial order 2, 4, 6 | 0.85, 1.22, 1.58 ms | 0.66, 0.99, 1.30 ms |

  The default-target build is 1.4x slower without dispatch (portable 2.81 ms at r = 1), so the
  AVX2 clone is what matters there. Under `x86-64-v3` the portable build is already AVX2, so the
  two match. On x86 the box is ahead of libblur on the default target and level or slightly
  behind it on AVX2 at r ≥ 4. The shared runner moved the prototype's box by about 40% between
  runs, so only the ratios within one run count.
- **Dispatch makes no difference on aarch64.** NEON is in the baseline, so the cloned
  functions are identical.
- **Binomial has no libblur comparator.** libblur has no binomial; the integer kernel is
  compared with the box only, and its exactness is checked against a naive integer
  reference on every border and size in the tests.

Exactness: the box divides with a float reciprocal and an integer correction, which is exact
for windows up to `(2·1023+1)²`. Binomial divides by `4^order` with a shift. Both are
bit-identical to the naive integer reference (tests in `filter::integer`).

## Phase 3 amendment: large-sigma Gaussian

Phase 3 adds the large-sigma path that the Gaussian section above calls for. The method is chosen by measurement:

- **Exact FIR below the crossover, cascade of three extended boxes above it.** The three passes
  are identical. Each has weight 1 on `|j| ≤ r` and a fractional weight `α` on `|j| = r+1`,
  with `r` and `α` set so that each pass has variance exactly `σ²/3`. Integer-width boxes
  could not match `σ²` (a first version was off by up to 3% of σ² at σ = 12), so the ends are
  fractional. The cascade's cost per pixel does not depend on σ.
- **Crossover σ = 8**, measured on the M4 Pro (1280×1024 `f32`, best of 25): FIR 4.9 ms, cascade
  4.7 ms at σ = 8, and the cascade is faster above it. The FIR cost grows with σ, so the
  cascade wins from there on.
- **Error budget (decision recorded for review):** the cascade is within **5 grey levels**
  (about 2% of 0..255) of the untruncated Gaussian on the test fixture for σ from 2 to 8. The
  fixture has a step of 160 and noise of ±64, so the figure is a stress case. The worst
  measured cases are `Constant(7)` at σ = 3 (3.2) and at σ = 16 (4.4). Below σ = 2 the three-box
  approximation is coarse (about 2 grey levels at σ = 1.5), so the `Auto` path does not use it.
- **Cost.** The horizontal pass is a serial running sum, so it is latency-bound. The cascade is
  about 4.8 ms at every σ. Vectorising across rows would lower that, and it is the next
  optimisation if large-σ speed matters. No libblur comparison is made here; the libblur
  `fast_gaussian` is approximate and has no error contract, so the comparison is not like for
  like.

Exactness of the border handling is checked separately from the approximation: the cascade equals
the separable convolution with its effective kernel, to `1e-3`, for every border mode.

## Phase 4 amendment: public API, consumers

The `filter` module is public. The API breaks no earlier contract: the module did not exist
publicly before. Three functions take images and return images, each with a `BorderMode`
and a `Result`:

- `box_mean_u8(src, radius, border)`: exact box mean, rounded half up.
- `binomial_u8(src, order, border)`: exact binomial of even order 2 to 12.
- `gaussian_f32(src, sigma, border)`: exact FIR below σ = 8, three-box cascade from σ = 8.

Scratch is internal. A reusable-scratch API is deferred to the hot-path work, once a consumer
measures the allocation. The Python bindings are `box_mean`, `binomial` and `gaussian`, with
the same names and border strings (`"clamp"`, `"reflect101"`, `"constant"` with `value`).

Consumers, decided by what each one computes:

- **Pyramid pre-smooth (`PreSmooth::Binomial121`) now uses the FIR.** It is the same `[1 2 1]/4`
  filter with `Clamp` borders. The output changes only in `f32` rounding order, and the
  pyramid tests pass unchanged.
- **Not rewired: `gradient::smooth_binomial3`, `edge1d` and `conv1d`.** `smooth_binomial3` is
  part of an expression tree that the tiled gradient path must reproduce bit for bit, and its
  tests lock that tree. `edge1d` smooths 1-D `f64` signals with its own kernel and the locked
  laser results depend on it. `conv1d` is a 1-D `f32` convolution, and no caller needs the
  2-D filter. Rewiring any of them changes locked numbers, so each stays until a measured
  benefit justifies it.
- **The integer kernels have no consumer yet.** They are public for callers that work on `u8`
  images, and the Python bindings expose them.

Not measured: the pyramid pre-smooth before and after. The FIR does the same work as the
hand-written loop, so no speed claim is made for it.

## Alternatives

- **Depend on libblur.** Fast, but it brings its own image and edge types, an approximate
  Gaussian with no error contract, a box whose `u8` output differs from the exact rounded
  mean by up to 1 LSB (so it cannot serve as the bit-exact reference for integer paths),
  and a dependency in every library consumer.
- **OpenCV via FFI.** Excluded by invariant 5.
- **Naive loops only.** Simple and exact, but about 10x slower at the sigmas used by
  the pyramid, so it fails the hot-path rule (invariant 6 and `hotpath-rust`).
- **Hand-written intrinsics.** Fastest possible, but needs an `unsafe` block per ISA and a
  per-ISA test matrix (invariant 7). Deferred until a measured need exists.

## Consequences

- Phases 1 to 5 each land as one PR: Phase 1 is the kernel and exact FIR, Phase 2 is box
  and binomial with the dispatch decision, Phase 3 is large-sigma Gaussian, Phase 4 wires
  the consumers and Python bindings, and Phase 5 updates the docs.
- Roadmap B3's acceptance criterion is amended in the same PR: bit-exact for integer
  paths, a stated few-ulp bound for float paths. The roadmap entry names this ADR.
- `tools/filter_bench` stays as the comparator. It is not published. Its golden files are
  the first cross-implementation check of the oracle, and the library tests reuse them. A
  manual-plus-branch CI job (`filter-bench.yml`) runs it on x86_64.
- **The speed target is now target-dependent, and runtime dispatch is needed to meet it.**
  The default x86_64 build cannot reach parity on the Gaussian, and only FMA-enabled code
  may use fused taps. Phase 2 therefore has to decide the dispatch mechanism on measured
  evidence for both targets. Whether parity on x86_64 is required, or a stated lag on the
  default target is accepted, is a gate decision for the user.
- Open: the f64-accumulation comparison against libblur's `IeeeBinaryConvolutionMode::High`;
  the box gap on NEON; the small-sigma gap on x86_64.
