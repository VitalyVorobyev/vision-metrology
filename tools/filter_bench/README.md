# Filter headroom bench

Phase 0 of the `filter` module ([ADR-0023](../../docs/dev/adr/0023-linear-filter-module.md)).
It answers one question before any library code is written: how much headroom is there
over libblur, the strongest pure-Rust blur crate, on the filters the module needs?

This is **development tooling**, not library code and not part of CI. The prototype kernels
in `src/proto.rs` are the first cut that the measurement was made on. The library
implementation is designed after these numbers, not copied from them.

## What is here

| File | Purpose |
|---|---|
| `src/proto.rs` | Prototype box (exact integer) and Gaussian (register-blocked FIR), the f64 oracle, and a deterministic fixture. Unit tests check them against the scipy golden files. |
| `src/main.rs` | The headroom harness: timing, accuracy against the oracle, and the libblur comparison. |
| `reference.py` | Independent scipy implementation. It writes `golden/`, and `--check` verifies the files are current. |
| `golden/` | Golden outputs for a 24x17 fixture. |

The crate is standalone (an empty `[workspace]` table), so libblur never enters the root
workspace or its lockfile. libblur is pinned to `=0.24.0`.

## Run

```sh
cd tools/filter_bench
cargo test --release                  # oracle and prototype against the scipy golden files
cargo run --release                   # the headroom table (about a minute)
python3 reference.py --check          # golden files are current (needs numpy, scipy)
```

To regenerate the golden files: `python3 reference.py` with numpy and scipy installed.

## Method

- Fixture: a 1280x1024 `u8` image from a 64-bit LCG, so it is reproducible without an RNG
  crate. Both sides read the same pixels.
- Timing: 3 warm-up calls, then 25 timed calls. The table reports the **best** (minimum)
  time. On a shared machine interference only adds time, so the minimum is the closest
  estimate of the kernel's own cost. Median was used first and gave a 6x spread between
  runs under load, so it was dropped.
- Fairness: the headline libblur column is `ThreadingPolicy::Single`. The prototype is
  single-threaded. libblur's adaptive threading is shown separately.
- Accuracy: the max absolute error against the f64 naive separable reference, with the
  same `ceil(3 sigma)` truncation and `Clamp` borders. Box errors are integer LSB counts
  against the naive integer mean.

## Results

Apple M4 Pro (aarch64, NEON), macOS. Load average about 40 during the runs (other work on
the machine). Two runs agreed within about 4%. Times in milliseconds.

```
image 1280x1024, 25 timed runs after 3 warm-up, best-of ms, single thread unless noted
| case | prototype ms | libblur ms (1 thread) | libblur / prototype | libblur ms (adaptive) | prototype max err | libblur max err |
|---|---|---|---|---|---|---|
| box u8 r=1 (exact) | 1.398 | 0.902 | 0.65x | 0.350 | 0 (exact) | 1 |
| box u8 r=2 (exact) | 1.415 | 0.871 | 0.62x | 0.439 | 0 (exact) | 1 |
| box u8 r=4 (exact) | 1.418 | 0.905 | 0.64x | 0.455 | 0 (exact) | 1 |
| box u8 r=8 (exact) | 1.421 | 0.875 | 0.62x | 0.361 | 0 (exact) | 1 |
| (saxpy baseline) gauss sigma=1 | 1.934 | | | | 5.7e-5 | |
| gauss f32 sigma=1 (R=3) | 0.853 | 0.693 | 0.81x | 0.361 | 5.7e-5 | 4.6e-5 |
| (saxpy baseline) gauss sigma=2 | 3.497 | | | | 5.1e-5 | |
| gauss f32 sigma=2 (R=6) | 1.326 | 1.214 | 0.92x | 0.546 | 5.1e-5 | 6.5e-5 |
| (saxpy baseline) gauss sigma=4 | 6.863 | | | | 5.9e-5 | |
| gauss f32 sigma=4 (R=12) | 2.132 | 2.429 | 1.14x | 0.812 | 5.9e-5 | 6.0e-5 |
| (saxpy baseline) gauss sigma=6 | 10.102 | | | | 7.6e-5 | |
| gauss f32 sigma=6 (R=18) | 3.519 | 3.887 | 1.10x | 1.500 | 7.6e-5 | 5.5e-5 |
| (context) libblur fast_gaussian u8 radius=6, approximate | | 2.047 | | | | |
| (context) libblur fast_gaussian u8 radius=18, approximate | | 2.117 | | | | |
```

The `saxpy baseline` rows are the first-cut Gaussian (one row-saxpy per tap). They show
that register blocking, which keeps each output chunk in registers across all taps, is
what brings the FIR from 2.9x slower to parity at sigma = 6.

## Limits of this measurement

- **One machine.** Linux x86_64 has not been measured. ADR-0023 makes both targets a
  condition of any speed claim.
- **Load.** The runs were made on a loaded machine. Best-of-25 reduces the effect; a quiet
  re-run should move the ratios by a few percent, not change their order. Re-run before
  quoting a number outside ADR-0023.
- **Prototype only.** The box path has a known gap (0.62x to 0.65x) that the prototype
  has not closed. The likely cause is the 64-bit reciprocal in the final division, which
  does not vectorise on NEON. This is measured as open, not explained away.
- **Float accumulation.** Both sides accumulate in f32. libblur's `High` accumulation mode
  (f64) was not measured.
- **Context rows are not comparisons.** `fast_gaussian` is an approximation with no sigma
  and no error column, and its timings include a 1.3 MB copy.
