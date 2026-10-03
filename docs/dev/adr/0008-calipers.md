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
  `f64` so results agree with a numpy reference to well under 1e-4 px. `t` is the
  distance from `start`, the reference's convention. `OffImage::Reject`
  (`ProfileConfig::off_image`) gives the reference's strict bounds; it applies to every
  placement; the default, `Fill`, samples the outside with `ProfileConfig::border`.
- **A curved edge gets its own placement.** A rectangle averages along a straight chord. On a
  circle of radius 40, a sample 5 px to the side sits at radius 40.31, on the wrong side of
  the edge. A 32-caliper fit reads 39.88 px; `MeasureRadial` reads 39.990 px, and its bias
  no longer grows with width.
- **Edges along the profile** come from `Edge1DDetector`, filtered by threshold and
  `PolaritySelect`, and narrowed by `EdgeSelect`. The derivative operator and the
  subpixel refinement are configurable (`ProfileConfig::derivative`,
  `Locate::GradientPeak { refine }`). The defaults, derivative of Gaussian and a
  three-point parabola, are what the accuracy envelopes and the can-end baseline
  are pinned to. The textbook alternatives (Gaussian then central differences, a
  log-parabola) exist so results can be compared with reference implementations
  operator for operator.
- **An optional obliquity gate** rejects an edge whose image gradient is too oblique to the
  scan. A glancing crossing reports a position along the scan rather than along the edge
  normal, and the two differ by `1/cos θ`.
- **A rejection is typed** (`RejectReason`, ADR-0006).
- **The config separates what is looked for from how the profile is built.**
  `MeasureConfig` holds threshold, polarity, selection and the obliquity gate;
  `MeasureConfig::profile` (`ProfileConfig`) holds smoothing, sampling step and
  border handling.
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
