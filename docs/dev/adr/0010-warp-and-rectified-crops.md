# ADR-0010: `warp` gathers from the source, and rectified crops are sized by their spec

- Status: Accepted
- Date: 2026-08-20

## Context

Undistortion, bird's-eye views, polar unwrapping and canonical-pose crops are all resampling
problems. A resampled pixel that came from outside the source image is not measurement data.
A canonical crop that feeds anomaly learning needs identical shapes on every frame.

## Decision

- **`dst → src` gather.** A `Map` stores, for every destination pixel, the source coordinate
  to sample. `apply` visits each destination pixel once, with no holes.
  `Map::affine`/`projective` therefore take the destination-to-source transform, which is
  the inverse of a fixture pose. The docs say so next to the one-line fix (`.inverse()`).
- **One construction path.** Every builder (affine, projective, polar, log-polar) goes
  through `Map::from_fn`. The coordinates are computed once, and `apply`'s loop never touches
  a matrix or a trig function. Polar maps sample bin centres, so a full turn never duplicates
  its seam column.
- **The validity mask comes from the fast-path branch.** `apply_with_mask` writes 255/0 from
  the same in-bounds test that selects the fast path, so the mask and the sampling cannot
  disagree.
- **Every call states its `BorderMode`.** Rectified crops recommend `Constant` plus the mask,
  not `Clamp` (the documented exception to invariant 11), because clamping fabricates texture
  that an anomaly model would learn as signal.
- **No prefiltering inside `warp`.** Minifying by more than about 2× aliases. Decimation goes
  through `pyr::Pyramid`.
- **Rectified crops live in `matching::crop`.** `CropSpec` plus
  `ShapeMatch::{model_frame_map, model_frame_pose}` builds the crop. The output size depends
  only on the spec (`rect × px_per_unit`), never on the match. `normalize_scale` reuses the
  pose's own isometry, because rebuilding it from decomposed parts leaves an offset of
  `(scale − 1)·R·origin`.

## Alternatives

- **Forward scatter.** It leaves holes and double-writes wherever the map compresses or
  expands.
- **A separate `align` module.** It would be an abstraction two functions wide over
  `Map::from_fn`.
- **An implicit default border.** It hides the one choice that decides whether a crop
  contains fabricated pixels.

## Consequences

- Building a map costs a precompute and memory per destination pixel. It is amortised over
  every frame warped with the same geometry.
- Python carries the raw pose on its `ShapeMatch` class instead of reconstructing one.
