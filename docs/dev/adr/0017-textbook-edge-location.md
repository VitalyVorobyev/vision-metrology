# ADR-0017: Textbook edge location and CaliperBench compatibility

- Status: Accepted
- Date: 2026-10-03

## Context

CaliperBench is a benchmark for 1-D edge measurement on synthetic tiles. Its reference
baselines are textbook operators over strips: a profile fixed sample for sample, a Gaussian
then central differences, level crossings, a half-contrast edge definition and a greedy
selection rule. Comparing this library's caliper with them, operator for operator,
needs the caliper to reproduce those profiles and definitions exactly.

The library's own defaults (derivative of Gaussian, a three-point parabola) are what the
accuracy envelopes and the can-end baseline are pinned to, so they cannot move to match.

## Decision

- **An edge is located one of three ways (`MeasureConfig::locate`, `Locate`):**
  - `GradientPeak { refine }`, the default: extrema of `Edge1DDetector`'s derivative,
    filtered by threshold and `PolaritySelect` and narrowed by `EdgeSelect`. The
    derivative operator (`ProfileConfig::derivative`) and the refinement (`refine`, which
    includes a log-parabola) are configurable, so a textbook operator can stand in for the
    default.
  - `MidpointCrossing`: CaliperBench's `midpoint_crossing` baseline. The smoothed
    profile's end levels (medians of `endpoint_samples` at each end) give one level and
    one polarity, and the crossing nearest the middle is the edge. Its checks run in
    CaliperBench's order (contrast, then polarity, then the crossing), so a rejection
    names the same gate the reference would.
  - `HalfContrast`: CaliperBench's reference edge definition, the local half-contrast
    crossing. Gradient edges are the seeds; each moves to the crossing of the mean of the
    flank medians either side of it, re-centred until it settles. It refines a selected
    edge rather than selecting one, so it keeps `select` and the gradient's seed-finding.

  The level primitives (`LevelCrossing1D`: end levels, interpolated crossings, the
  half-contrast iteration) live in `vm-primitives` next to `Edge1DDetector`, in samples;
  `measure` converts pixel distances by the profile's spacing.
- **`Derivative::SmoothThenCentral` is computed in `f64`.** It smooths with a normalised
  Gaussian and takes central differences (`Derivative1D::SmoothThenCentral` in
  `vm-primitives`), both in `f64`, like the strip's geometry and weights (ADR-0008). The
  parabola divides the response's rounding by a broad peak's small curvature, so `f32`
  rounding alone would move the peak away from a numpy reference.
- **`EdgeSelect::StrongestInOrder` is CaliperBench's greedy rule:** per entry, the strongest
  edge strictly after the previous choice, ties to the earlier edge. It orders by subpixel
  profile position rather than `t`, because `t` decreases along an arc with a negative
  extent.
- **Derivative of Gaussian stays the default.** Gradient peaks need no flat material either
  side of the edge, find several edges in one window, and are what the accuracy envelopes
  and the can-end baseline are pinned to.
- **CaliperBench runs the caliper through an example, not library API.**
  `examples/caliperbench_run.rs` speaks its JSONL protocol: a request's strip, its
  parameters in samples, its polarities (`StrongestInOrder`, or `Strongest` for a negative
  task) and its failure reasons, in its order. The protocol and Pillow's image conversion
  are CaliperBench's conventions, not the caliper's, so they stay out of `measure`.
- **A golden fixture pins the mapping, with no runtime dependency on CaliperBench.**
  `tests/fixtures/caliperbench_golden.json` holds what CaliperBench's own baselines return
  on small inline images, intermediates included. CaliperBench generates it
  (`tools/gen_caliperbench_golden.py`), and `tests/caliperbench_protocol.rs` checks the
  runner against it. CaliperBench is needed only to regenerate the fixture
  ([CONTRIBUTING](../../../CONTRIBUTING.md#tests)).

## Alternatives

- **Extending `MeasureRect` instead of adding a strip placement.** A rect is centred, and
  its ends and samples follow from `half_len` and `step`. Endpoint and sample-count
  overrides would give one type two conflicting ways to say where it samples.
- **Making `SmoothThenCentral` the default.** It would move every pinned envelope and the
  can-end baseline, with no accuracy gain on the library's own fixtures.
- **Non-textbook logic in the runner.** Edge definitions written inside the example would be
  unavailable to library users, and the golden check would test the runner rather than the
  caliper. Only CaliperBench's protocol and image conventions belong there.
- **Calling CaliperBench from the test suite.** It would make `cargo test` depend on a
  Python environment and an external checkout.

## Consequences

- The caliper can be compared with CaliperBench's baselines operator for operator, and
  `docs/performance.md` reports the strip and caliper rows on its image model.
- The textbook operators are public API with Python parity, beside defaults that stay
  pinned.
- A change to CaliperBench's baselines means regenerating the golden fixture with its
  environment.
