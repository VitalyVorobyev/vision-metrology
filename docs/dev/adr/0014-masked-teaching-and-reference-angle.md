# ADR-0014: Masked teaching and the model's reference angle

- Status: Accepted
- Date: 2026-08-21

## Context

A model taught from a bare rectangle learns whatever background the rectangle contains, and
because of invariant 4 those points dilute every later score. The found angle is also
relative to however the reference image happened to be shot, not to a canonical orientation
of the part.

## Decision

- **`ShapeModelBuilder::build_with_mask`** takes an optional inclusion mask. A point is kept
  if the level-0 position it was aggregated from lies inside the mask, **dilated**: a coarse
  level's point can sit up to half its own pixel from the fine edge that produced it, and a
  tight mask silently deletes coarse levels.
- **`ShapeModelConfig::reference_angle`** rotates the model **frame** onto a caller-chosen
  canonical orientation at build time. A found pose then reads as "how far from canonical".
  It does not filter points. `reference_geometry` reports the taught geometry in the reference
  image's frame, and `model_geometry` reports it in the rotated model frame.

## Alternatives

- **Arbitrary-polygon ROIs.** A mask is more general, and a UI can build one from picked
  contours.
- **Rotating the reference image before teaching.** Resampling changes the edges being taught.

## Consequences

- The model format gained `reference_angle` (format 5, ADR-0005). Older models read 0.
- The lab's curated teaching (pick contours → mask) depends on this; ADR-0015 covers the lab.
