# ADR-0018: Tracked curves: a prior, two caliper stages and a regularised normal solve

- Status: Proposed
- Date: 2026-10-06

## Context

An elongated bead or stripe usually has a path that is already roughly known: from a
reference part, the previous frame, a fixture pose or a robot program. Typical cases are an
adhesive or sealant bead, a seam, or a stripe seen at an angle. The task is not to find
edges anywhere in the image. It is to place reliable measurement poses along the bead, and
to report its position and width there.

A caliper's pose is a point, a tangent and a normal; the normal is the measurement
direction. Calipers placed independently along a prior measure on the wrong normal wherever
the prior is off. Their raw midpoints also carry wrong edge pairs (reflections, a
neighbouring edge), gaps and misses. ADR-0008 already keeps centreline refinement and the
bead-specific gates out of `Caliper`.

## Decision

- **A bead is tracked from a prior, not segmented.** `measure::BeadTracker` takes an open
  polyline and returns three things: the refined centreline, a final measurement per
  station, and quality statistics. The work scales with the number and length of the search
  profiles, not with the image area.
- **The loop is an Active Shape Model search without a statistical shape model.** One pass
  runs these steps:
  - stations uniform in arc length;
  - a tangent and a normal at each station;
  - a strip caliper along each normal, which yields the bead's edge pair;
  - one global, regularised solve for the normal offsets;
  - the curve moves along its normals and is resampled.

  A fixed number of passes repeats this.
- **Stations.**
  - The prior is resampled uniformly in arc length, in `f64`. The station count is fixed
    for the call, so station `i` keeps its identity across passes.
  - Tangents are chords over a ± window. The chord is exact on a circle and smooths a
    coarse prior's corners.
  - The normal is `t.perp()`, and offsets are signed along it.
- **Edge evidence reuses the caliper.** Each station measures with `Caliper::measure` on a
  `MeasureStrip`, keeping every candidate edge of either polarity. The tool then picks the
  pair itself:
  - the polarity order, from the bead's appearance;
  - a width range;
  - the admissible offset window;
  - an optional clearance: no other edge just outside the bead;
  - an optional ambiguity margin over the best competing pair.

  The score is the weaker edge's amplitude, discounted by the offset.
  - `measure_pairs` is not used: its greedy pairing has no width or position gate.
  - Edge location (the derivative, refinement, half contrast) stays the caliper's, so the
    tool adds no second edge detector.
- **The global update is a robust, regularised least-squares problem on the normal
  offsets.**
  - It minimises
    `Σ wᵢ ρ(dᵢ − d̂ᵢ) + λ0 Σ dᵢ² + (ℓ1/h)² Σ (Δd)ᵢ² + (ℓ2/h)⁴ Σ (Δ²d)ᵢ²`.
    This is the discretisation of a continuous energy, so the result does not depend on
    the station spacing `h`.
  - `λ0` is trust in the prior. `ℓ1` and `ℓ2` are lengths: corrections much shorter than
    `ℓ2` are suppressed.
  - Rejected stations have zero weight. `ρ` is `fit::RobustLoss`, applied by IRLS.
  - The system is symmetric pentadiagonal and is solved by a banded LDLᵀ in O(N)
    (ADR-0002).
  - A curvature guard clips the offset window on the concave side and scales the step, so
    the curve cannot fold.
- **Tracking and measuring are separate stages with separate settings.**
  - Tracking evidence establishes where the bead is, with a wide, permissive search.
  - The reported dimensions come from a second set of calipers, placed on the refined curve
    with stricter settings. That stage has no regularisation and no feedback into the curve.
  - Its midpoints' residual against the refined curve is reported (invariant 21).
- **Absence is a result** (ADR-0006).
  - Each station carries a hit or a typed rejection.
  - A bead that is not there is still a result, with every station rejected. An error
    means an invalid configuration or prior.
  - Quality is reported as the support fraction, the longest unsupported gap, each pass's
    correction and residual, the reason the loop stopped, width statistics, and rejections
    counted by reason.
- **The result is the next prior.** The refined centreline is a polyline in image
  coordinates, and passing it back as the next call's prior is temporal tracking. The caller
  applies a known motion to the prior: a similarity or affine transform, or a
  `corr::displacement` shift. There is no tracking framework.
- **Explaining runs the same code once.** The algorithm runs over a probe.
  - The quiet probe measures.
  - The tracing probe measures through `diagnostics::explain` and records every station.

  `diagnostics::explain_bead` therefore returns `track`'s result to the bit, with each
  station's evidence beside it.
- **Scope.** The tool handles open, non-branching curves that have a prior. It does not
  cover:
  - closed or branching curves;
  - curved strip placements;
  - acquisition without a prior, meaning candidate centrelines found from the image alone
    (e.g. a Steger ridge detector). Acquisition is a separate mode, evaluated offline
    before any decision, and it must not slow the prior-driven path.

## Alternatives

- **A caliper option.** Rejected for the reasons in ADR-0008.
- **Connecting raw midpoints.** Wrong pairs, gaps and misses would go straight into the
  curve.
- **Snakes with an internal energy.** The elasticity term shrinks and straightens the curve
  by itself, which biases a metrology result. Here the penalties act on the offsets, and the
  data term is explicit edge evidence.
- **An Active Shape Model with a PCA shape model.** It needs training sets. The regulariser
  stands in for the shape prior without one.
- **Fitting a spline.** It adds knot choices and gives no advantage over the station model
  at this sampling density.
- **Extending `laser`.** It scans the rows or columns of a stripe that stays roughly
  parallel to the image axis; a bead follows an arbitrary curve.
- **A dense solve.** It costs O(N³) per reweighting iteration.
- **Global acquisition on every frame.** It spends full-image work on an answer the prior
  already gives.

## Consequences

- New `measure` API, with Python parity, accuracy rows and a bench.
- The banded solve is the library's first structured solver. ADR-0002 states the rule it
  follows.
- A width measured along the refined curve's normal is only as good as that curve. The
  residuals and support statistics show when it is not.
- The penalties act on each pass's increment, so high-frequency error in the prior decays
  over passes rather than in one.
- Strips are straight, and on a curved bead their averaging biases the centre towards the
  concave side. The bias grows with the strip's half width and falls with the radius.
