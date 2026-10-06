# ADR-0008: Calipers and their placements

- Status: Accepted
- Date: 2026-10-03

## Context

A caliper turns a 2-D region into a 1-D profile and finds subpixel edges along it.
`MetrologyModel` distributes calipers over nominal primitives (lines, circles) held in the
part's frame, applies them at a fixture pose, and fits the hits with `fit`. Two things decide
whether the result is unbiased:

- how the profile is averaged across the caliper's width;
- whether the scan crosses the edge along its normal.

## Decision

- **Placements are geometry types:**
  - `MeasureRect`: a straight scan, centred, averaged across.
  - `MeasureArc`: a scan along an arc, averaged radially.
  - `MeasureRadial`: a radial scan, averaged **along the arc**.
  - `MeasureStrip`: a straight scan from `start` to `end`, averaged across, with
    optional explicit `samples` along (endpoints included) and `across` lines.
- **A curved edge gets its own placement.** A rectangle averages along a straight chord. On a
  circle of radius 40, a sample 5 px to the side sits at radius 40.31, on the wrong side of
  the edge, so rectangles bias a fitted radius low by an amount that grows with their
  width. `MeasureRadial` averages along the arc, and its bias does not grow with width.
- **A strip has exact endpoints and an explicit sample count.** A rect is centred, so its
  ends and its sample positions follow from `half_len` and `step`. A strip's first sample is
  on `start` and its last exactly on `end`, spaced `length / (samples − 1)`, and σ is
  converted with that true spacing. Its geometry and bilinear weights are computed in
  `f64`. `t` is the distance from `start`. Why the strip exists is
  [ADR-0017](0017-textbook-edge-location.md).
- **Placement is computed once.** `MetrologyModel::apply` and `measure::diagnostics::layout`
  call the same placement code, so a drawn caliper is the one that measures. `layout` needs
  no image. `MeasureRadial::center` is the circle's centre, not the caliper's boundary point.
- **`EdgeSelect` decides which candidates are reported:** `All`, `First`, `Last`,
  `Strongest`, or `StrongestInOrder` ([ADR-0017](0017-textbook-edge-location.md)).
- **`OffImage` decides what a profile that leaves the image means**, for every placement
  (`ProfileConfig::off_image`). `Fill`, the default, samples the outside with
  `ProfileConfig::border`. `Reject` rejects the caliper with `RejectReason::OffImage`, the
  strict bounds a reference implementation applies.
- **An optional obliquity gate** rejects an edge whose image gradient is too oblique to the
  scan. A glancing crossing reports a position along the scan rather than along the edge
  normal, and the two differ by `1/cos θ`.
- **A rejection is typed** (`RejectReason`, ADR-0006): `ProfileTooShort`, `NoEdge`,
  `WrongPolarity`, `TooOblique`, `OffImage`, `IncompleteSequence`, and, for the level
  methods of ADR-0017, `LowContrast` and `NoCrossing`.
- **The config separates what is looked for from how the profile is built.**
  `MeasureConfig` holds `threshold`, `polarity`, `select`, `locate` (ADR-0017) and the
  obliquity gate. `MeasureConfig::profile` (`ProfileConfig`) holds the smoothing `sigma`,
  the `derivative` operator, the sampling `step`, the `border` and `off_image`.

## Alternatives

- **Rectangles for every primitive.** They bias circle radii low, by an amount that grows with
  caliper width.
- **Hard-thresholded test fixtures.** A binary disc puts its edge 0–0.5 px inside the nominal
  radius depending on phase, so it cannot assert subpixel accuracy. Fixtures are anti-aliased.
- **Background-padding and centreline-refinement gates.** These are properties of a tracked
  contour, not of a caliper: refining centres from a rough polyline and re-measuring from the
  refined one, and requiring clean background beyond each edge. They belong in the bead
  tracker built on `measure` ([ADR-0018](0018-tracked-curves.md)).

## Consequences

- Each new placement needs its own profile sampler, a layout entry, Python parity, and an
  accuracy row.
- A reusable `Caliper` owns its profile and detector scratch, so a model with many calipers
  measures without allocating per caliper.
- Explaining a caliper or a whole model goes through this placement and measurement code
  ([ADR-0006](0006-absence-is-a-result.md)).
