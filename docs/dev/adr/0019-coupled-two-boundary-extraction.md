# ADR-0019: Coupled two-boundary extraction: one exact path over both walls

- Status: Accepted
- Date: 2026-10-16

## Context

`BeadTracker` (ADR-0018) chooses each station's edge pair on its own, in
`pairing::choose`, and only then regularises the offsets. A station sees its own strip and
nothing else, so a wrong pair that is as strong as the right one, such as another line in
the window, is measured as well as the bead would be, and no per-station signal can tell.
The backlog records these silent wrong locks on textured surfaces.

Both walls of a bead are continuous along it. A choice of both walls at every station
together, under that continuity, can prefer a weaker pair that continues the curve over a
stronger one that does not. A shortest path over the stations also has max-marginals: what
the best answer that differs at a station costs. Those give a margin per station and for
the whole call, which no per-station signal provides.

`tools/bead_eval/coupled.py` prototyped two variants, a sparse one over the caliper's
candidate pairs and a dense one over the profile itself, on real cracks and on synthetic
beads with known walls. The library's operator reproduces the dense variant.
`docs/performance.md` holds the numbers.

## Decision

- **A pairing mode of `BeadTracker`.** `BeadConfig::pairing` is
  `BeadPairing::PerStation` (the default, ADR-0018's choice) or
  `BeadPairing::Coupled(BeadCoupling)`. The stations, strips, stages and solve of ADR-0018
  stay. In each stage the coupled mode measures every station's strip first, then chooses
  the walls of all stations as one path, then reports each station.
- **A dense path over (centre, width) bins, on `measure::numeric::LayeredPath`
  (ADR-0020).** Per station the centre runs over the stage's reach on a grid of
  `bin_px`, rounded to whole profile samples, and the width over the width range on twice
  that grid, so both walls fall on bins. Four planes share the grid:
  - **both walls**;
  - **the `−n` wall alone** and **the `+n` wall alone**, which exist only where the other
    wall's bin has no valid response, off the image. Where one side leaves the image the
    visible wall is still measured, and on the image a one-wall state cannot stand in for
    a weak wall;
  - **absent**, with the same geometry, so a gap is bridged smoothly.

  A state outside the station's offset window (ADR-0018's curvature clip) does not exist.
  The path is solved exactly by a forward and a backward pass.
- **The evidence is the stage's caliper response.** Each strip's derivative response, per
  pixel, is the path's evidence, whichever `Derivative` the stage uses: the default
  bilinear profile, or `Derivative::Continuous`, which samples `ProfileSampler` at the
  strip's points (ADR-0021). Samples are max-pooled into bins; a bin is valid only when
  every sample in it lies on the image (for the continuous profile, when its whole
  support does). No second profile is sampled.
- **Costs per unit length.** A wall's evidence is its response signed to the bead's
  polarity and softly saturated, `e(g) = (g/s)/(1 + |g|/s)`, so one strong edge cannot carry
  a pair. Per pixel of curve, both walls cost `−e_l − e_r + λ0 (c/reach)²`, one wall
  `−e − e(absent_level) + λ0 (c/reach)²`, and absence `−2 e(absent_level)`. Between stations
  `h` apart, quadratic penalties on both walls' steps and on the width change, divided by
  `h`, with bounded slopes (at least one bin), and a cost per switch between present and
  absent. A switch between both walls and one is free: the image decides it. Data per unit
  length and smoothness per unit slope make the result independent of the station
  spacing, as ADR-0018's length scales do.
- **`clearance` becomes a cost.** With coupled pairing, `BeadConfig::clearance` charges
  each wall the evidence of the strongest other edge within that distance outside it,
  beyond the snap radius. `min_margin`, a per-station gate, must be `None`.
- **Separable transitions.** `Δl² + Δr² = 2Δc² + Δw²/2`, so the step is a centre term
  plus a width term, two banded passes per plane. A plane with no finite state in a layer
  is skipped, which is exact, so on the image's interior the one-wall planes cost
  nothing.
- **Edge location stays the caliper's.** Each chosen wall moves to the nearest caliper
  edge of its polarity within a pixel, or one and a half bins on a coarser grid. A wall
  with no such edge is not measured: the station reports the other wall alone, or none.
  The accuracy rows chose the snap over refining on the continuous profile: with the
  default derivative the walls' σ is lower at the same blur (`docs/performance.md`), and
  `Derivative::Continuous` remains a stage setting for whoever needs it.
- **Typed outcomes (ADR-0006).** A station with both walls is a `BeadHit`. Otherwise it is
  `BeadReject::OneSided`, with the measured wall in `BeadSample::wall`;
  `BeadReject::Absent`; the caliper's own reason when its strip found no edge or lay off
  the image; or `NoPair` when no caliper edge lies at the chosen walls.
- **Max-marginals are reported, as a competitor flag.** Per station with a wall, the margin
  is the cost of the best path with a wall more than `margin_tol_px` away there, minus the
  best path's (`BeadSample::margin`, and the traces of `explain_bead`). Each stage's
  `BeadPathScore` holds a low quantile of the margins and the path's mean evidence per
  pixel. The first tracking pass's window is the prior's, so its margin is the call's
  flag; the passes between the first and the final stage choose the path without its
  backward pass and report no margins.
- **One tracking pass is the recommended setting.** A later pass re-stations on the
  chosen line and extracts again inside a window centred on it; on the development split
  it locked less often and accepted more wrong locks. `BeadTuning::passes` keeps its
  default, which per-station pairing needs.

## Alternatives

- **Today's per-station choice.** It has no view of the neighbouring stations, and no
  margin wider than one strip. It remains the default.
- **A Viterbi over the caliper's candidate pairs.** It is cheaper, but its states exist only
  where the thresholded caliper found both edges, so it inherits the per-station gates.
  It did not lower wrong locks at equal coverage.
- **Only both-walls and absent states.** Where a wall leaves the image, the state space
  then has only absence or a wrong pair for the visible wall, and loses it (issue #73's
  case).
- **One-wall states everywhere.** They would let a station drop a weak but observable
  wall, and change the choice on every textured image.
- **Sampling `ProfileSampler` separately for the path.** It doubles the profile work per
  station, while the stage's caliper already computes a response along the same strip,
  continuous when the stage asks for it.
- **The max-marginal margin as the call's wrong-lock test.** On real cracks most wrong
  locks are not close calls: another structure has at least as much evidence. The margin
  flags a nearly equal competitor; it does not certify a lock.
- **A width prior on by default.** It lowered the lock rate on beads whose width varies.
- **An unbounded quadratic distance transform.** Exact in linear time, but the wall steps
  must be bounded, and a banded minimum with a short band is as cheap.

## Consequences

- The coupled mode costs more per call than per-station pairing; `benches/bead.rs` pins it
  at 300 stations within the tracker's 3-pass budget.
- The mode has its own config, `BeadCoupling`, in units that do not depend on the spacing
  (ADR-0007), with Python parity.
- Accuracy rows pin the walls' bias and σ, a varying width, a gap, a highlight and a 2 px
  distractor; tests pin the parallel line's margin and the one-sided walls.
- An edge that merges with a nearby one in the derivative still biases its wall: the path
  chooses between edges, and cannot separate them.
- The tracker takes no mask, so occlusion inside the image is not a one-wall case for it;
  the path itself handles any invalid samples, as its unit tests on issue #73's profiles
  show.
