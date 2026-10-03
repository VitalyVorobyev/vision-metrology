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
- **A curved edge gets its own placement.** A rectangle averages along a straight chord. On a
  circle of radius 40, a sample 5 px to the side sits at radius 40.31, on the wrong side of
  the edge. A 32-caliper fit reads 39.88 px; `MeasureRadial` reads 39.990 px, and its bias
  no longer grows with width.
- **Edges along the profile** come from `Edge1DDetector` (derivative of Gaussian, parabolic
  subpixel refinement), filtered by threshold and `PolaritySelect`, and narrowed by
  `EdgeSelect`.
- **An optional obliquity gate** rejects an edge whose image gradient is too oblique to the
  scan. A glancing crossing reports a position along the scan rather than along the edge
  normal, and the two differ by `1/cos θ`.
- **A rejection is typed** (`RejectReason`, ADR-0006).
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
