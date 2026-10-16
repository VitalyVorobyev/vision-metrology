# ADR-0019: Coupled two-boundary extraction: one exact path over both walls

- Status: Proposed
- Date: 2026-10-15

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
beads with known walls. `docs/performance.md` holds the numbers.

## Decision

- **A dense dynamic programme over (centre, width) bins, plus absence.** Per station, the
  states are the bead's centre on the profile's sample grid within the reach and its width
  on twice that grid, so both walls fall on profile samples; and an absent layer with the
  same geometry, so a gap is bridged smoothly. It is solved exactly by a forward and a
  backward pass.
- **Costs per unit length.** A present state costs the soft-saturated, polarity-signed
  derivative at each wall, `e(g) = (g/g0)/(1 + |g|/g0)`, so one strong edge cannot carry a
  pair; an absent one a fixed level. Optional terms: a robust prior on log width, and
  trust in the prior's offset. Between stations, quadratic penalties on both walls' steps
  and on the width change, divided by the spacing, with bounded slopes, and a cost per
  switch between present and absent. Data per unit length and smoothness per unit slope
  make the result independent of the station spacing, as ADR-0018's length scales do.
- **Separable transitions.** `Δl² + Δr² = 2Δc² + Δw²/2`, so in (centre, width) the
  transition is a sum of a centre term and a width term. The minimum over it is two banded
  passes per station instead of a pass over every pair of states.
- **Max-marginals are reported, as a competitor flag.** Per present station, the margin is
  the cost of the best path with a wall more than a tolerance away there, minus the best
  path's; the call reports a low quantile of the margins. A low value means another line
  in the window was nearly as good. They come from the extraction whose window is the
  prior's. The best path's mean evidence is reported beside them.
- **Edge location stays the caliper's.** Each chosen wall moves to the nearest caliper edge
  of its polarity within a pixel, through the existing `Caliper` and `Edge1D`; a wall with
  no such edge keeps its grid position and says so. There is no second edge detector.
- **It plugs into `BeadTracker` as an alternative pairing mode.** The stations, strips,
  stages and solve of ADR-0018 stay; the mode replaces the per-station choice with the
  path, the absent states are typed rejections (ADR-0006), and `explain_bead` reports the
  margins through the existing probes.

## Alternatives

- **Today's per-station choice.** It has no view of the neighbouring stations, and no
  margin wider than one strip.
- **A Viterbi over the caliper's candidate pairs.** It is cheaper, but its states exist only
  where the thresholded caliper found both edges, so it inherits the per-station gates.
  It did not lower wrong locks at equal coverage.
- **The max-marginal margin as the call's wrong-lock test.** On real cracks most wrong
  locks are not close calls: another structure has at least as much evidence. The margin
  flags a nearly equal competitor; it does not certify a lock.
- **Re-stationing on the chosen curve and extracting again.** The second window is centred
  on the chosen line, so a competitor can leave it and the margin no longer sees it. It
  did not lock more often.
- **A width prior on by default.** It lowered the lock rate on beads whose width varies.
- **An unbounded quadratic distance transform.** Exact in linear time, but the wall steps
  must be bounded, and a banded minimum with a short band is as cheap.

## Consequences

- The work per station is the number of states times the two bands, far more than one
  pair search; a bench must show the tracker's budget holds before this is accepted.
- The mode needs its own config: the saturation, absence level, stiffnesses, slopes and
  switch cost, in units that do not depend on the spacing (ADR-0007).
- Accuracy rows for the boundaries' bias and σ, Python parity, and the synthetic fixtures
  with asymmetric walls, a distractor, a parallel line, gaps, occlusion and highlights.
- An edge that merges with a nearby one in the derivative still biases its wall: the path
  chooses between edges, and cannot separate them.
