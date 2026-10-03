# ADR-0008: Calipers and their placements

- Status: Accepted
- Date: 2026-08-19

## Context

A caliper turns a 2-D region into a 1-D profile and finds subpixel edges along it.
`MetrologyModel` distributes calipers over nominal primitives (lines, circles) held in the
part's frame, applies them at a fixture pose, and fits the hits with `fit`. Two things decide
whether the result is unbiased:

- how the profile is averaged across the caliper's width;
- whether the scan crosses the edge along its normal.

## Decision

- **Placements are geometry types:**
  - `MeasureRect`: a straight scan, averaged across.
  - `MeasureArc`: a scan along an arc, averaged radially.
  - `MeasureRadial`: a radial scan, averaged **along the arc**.
  - `MeasureStrip`: a straight scan from `start` to `end`, averaged across, with
    optional explicit `samples` along (endpoints included) and `across` lines.
- **A strip exists for exact endpoints and explicit sample counts.** A rect is centred,
  so its ends and its sample positions follow from `half_len` and `step`. Reproducing a
  reference implementation (CaliperBench's strips) needs the profile fixed sample for
  sample: the last sample exactly on `end`, the spacing `length / (samples − 1)`, and
  σ converted with that true spacing. Its geometry and bilinear weights are computed in
  `f64`, and `SmoothThenCentral` smooths and differences in `f64`, so results agree with
  a numpy reference to well under 1e-4 px. In `f32`, the smoothing alone moved a broad
  derivative peak by up to 1.7e-4 px: the parabola divides the response's rounding by
  the peak's small curvature. `t` is the distance from `start`, the reference's
  convention. `OffImage::Reject`
  (`ProfileConfig::off_image`) gives the reference's strict bounds; it applies to every
  placement; the default, `Fill`, samples the outside with `ProfileConfig::border`.
- **A curved edge gets its own placement.** A rectangle averages along a straight chord. On a
  circle of radius 40, a sample 5 px to the side sits at radius 40.31, on the wrong side of
  the edge. A 32-caliper fit reads 39.88 px; `MeasureRadial` reads 39.990 px, and its bias
  no longer grows with width.
- **Edges along the profile** are located one of three ways (`Locate`):
  - `GradientPeak { refine }`, the default: extrema of `Edge1DDetector`'s derivative,
    filtered by threshold and `PolaritySelect` and narrowed by `EdgeSelect`. The
    derivative operator and the refinement are configurable
    (`ProfileConfig::derivative`, `refine`). The defaults, derivative of Gaussian and a
    three-point parabola, are what the accuracy envelopes and the can-end baseline are
    pinned to; the textbook alternatives (Gaussian then central differences, a
    log-parabola) exist so results can be compared with reference implementations
    operator for operator.
  - `MidpointCrossing`: CaliperBench's `midpoint_crossing` baseline. The smoothed
    profile's end levels (medians of `endpoint_samples` at each end) give one level and
    one polarity, and the crossing nearest the middle is the edge. Its checks run in
    CaliperBench's order (contrast, then polarity, then the crossing), so a rejection
    names the same gate the reference would.
  - `HalfContrast`: CaliperBench's reference edge definition, the local half-contrast
    crossing. Gradient edges are the seeds; each moves to the crossing of the mean of the
    flank medians either side of it, re-centred until it settles. It is a refinement of
    a selected edge, not a selector, so it keeps `select` and the gradient's
    seed-finding.

  The level primitives (`LevelCrossing1D`: end levels, interpolated crossings, the
  half-contrast iteration) live in `vm-primitives` next to `Edge1DDetector`, in
  samples, with `measure` converting pixel distances by the profile's spacing.
  Gradient peaks stay the default because they need no flat material either side of
  the edge, find several edges in one window, and are what existing results are pinned
  to.
- **`EdgeSelect::StrongestInOrder`** is CaliperBench's greedy rule (per entry, the
  strongest edge strictly after the previous choice, ties to the earlier edge). It
  orders by subpixel profile position rather than `t`, because `t` decreases along an
  arc with a negative extent.
- **An optional obliquity gate** rejects an edge whose image gradient is too oblique to the
  scan. A glancing crossing reports a position along the scan rather than along the edge
  normal, and the two differ by `1/cos θ`.
- **A rejection is typed** (`RejectReason`, ADR-0006). The level methods add
  `LowContrast` and `NoCrossing`.
- **The config separates what is looked for from how the profile is built.**
  `MeasureConfig` holds threshold, polarity, selection and the obliquity gate;
  `MeasureConfig::profile` (`ProfileConfig`) holds smoothing, sampling step and
  border handling.
- **CaliperBench runs the caliper through an example, not library API.**
  `examples/caliperbench_run.rs` speaks its JSONL protocol: a request's strip, its
  parameters in samples, its polarities (`StrongestInOrder`, or `Strongest` for a
  negative task) and its failure reasons, in its order. The protocol and Pillow's image
  conversion are CaliperBench's conventions, not the caliper's, so they stay out of
  `measure`. A golden fixture generated by CaliperBench itself
  (`tools/gen_caliperbench_golden.py`) pins the mapping, intermediates included.
- **Explaining is a separate call.** `measure` keeps only what the next call reuses, so
  its hot path stays allocation-free. `measure::diagnostics::explain` measures through
  the same `Caliper` and then copies out the intermediates (profile, smoothed profile,
  derivative, candidates before `select`, level crossings), so the trace cannot disagree
  with the result it explains. A whole model is explained in one pass:
  `diagnostics::explain_model` places each caliper with `apply`'s placement code,
  measures it once through `explain`, and fits its first edges through the same loop
  `apply` runs (`model::measure_placed`, with the measurement as a closure). Each object's
  result is therefore `apply`'s, with every caliper's placement and trace beside it, and
  a tool needs no second measurement per caliper.
- **Placement is computed once.** `MetrologyModel::apply` and `measure::diagnostics::layout`
  call the same placement code, so a drawn caliper is the one that measures. `layout` needs
  no image. `MeasureRadial::center` is the circle's centre, not the caliper's boundary point.

## Alternatives

- **Rectangles for every primitive.** They bias circle radii low, by an amount that grows with
  caliper width.
- **Hard-thresholded test fixtures.** A binary disc puts its edge 0–0.5 px inside the nominal
  radius depending on phase, so it cannot assert subpixel accuracy. Fixtures are anti-aliased.
- **Background-padding and centreline-refinement gates.** These are properties of a tracked
  contour, not of a caliper. They belong in a future bead/stripe tool built on `measure`.

## Consequences

- Each new placement needs its own profile sampler, a layout entry, Python parity, and an
  accuracy row.
- A reusable `Caliper` owns its profile and detector scratch, so a model with many calipers
  measures without allocating per caliper.
