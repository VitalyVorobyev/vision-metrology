# ADR-0001: Locate parts by gradient-orientation shape matching

- Status: Accepted
- Date: 2026-08-18

## Context

Every measurement starts from a part located at subpixel pose under rotation, uniform
scale, partial occlusion, clutter and illumination change. The locator's pose becomes the
fixture that calipers are placed in, so its bias propagates into every measured dimension.

## Decision

`matching` implements Steger's similarity measure (the algorithm behind HALCON
`find_shape_model`): `S = (1/n) Σ (R·tᵢ)·ĝ(pᵢ)` over the model's edge points, with three
polarity modes.

- Coarse-to-fine search over a box-mean pyramid with 3-D (x, y, angle) local maxima, greedy
  early termination checked once per 8-point chunk, and correspondence-free least-squares
  pose refinement (4 → 3 → 2 DOF Cholesky fallback for symmetric parts).
- The score divides by the **full** model point count (invariant 4), so
  `score ≈ 1 − occluded_fraction` and `min_score` has a physical meaning.
- Model and scene are built from the same pyramid kernel (invariant 3).
- The dense, gated unit-gradient field per pyramid level, `DirectionField`, lives in
  `vm-primitives::edge`. It is an image primitive with no matching semantics, and the search
  fills it lazily in tiles around the candidates it actually visits.

## Alternatives

- **Chamfer distance matching.** It ignores gradient orientation, so it latches onto any
  edge of the right shape. Rejected.
- **Correlation (ZNCC) as the locator.** Photometric, so it tracks illumination and surface
  texture rather than geometry, and it has no scale search. Kept as a separate tool (`corr`,
  ADR-0011) and as an independent cross-check of the matcher, not as its replacement.
- **Holding `GradientBuffers` per pyramid level.** They borrow the 2-D edge detector, so they
  cannot be stored across levels, and running the full edge pipeline (NMS, hysteresis, edgel
  build) per level is about twice the necessary work.

## Consequences

- The model's `min_contrast` decides real-world performance. On low-relief parts, faint
  surface shading admits model points that never repeat, and because of invariant 4 they
  dilute every score rather than merely not helping. The tuning guide is `docs/shape-matching.md`.
- Search time is dominated by preprocessing (the direction-field pyramid), not by scoring.
  This is why the fields are tiled lazily.
- `morph::chamfer_distance_u8` remains as an independent primitive.
