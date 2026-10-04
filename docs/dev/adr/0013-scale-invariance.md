# ADR-0013: Scale invariance is estimate-then-verify

- Status: Accepted
- Date: 2026-08-20

## Context

The shape matcher can scan a discrete scale range, but:

- its cost grows linearly with the width of the range;
- a model taught with the default `scale_range = (1, 1)` cannot be found away from 1.0 at
  all, however wide the search asks;
- at `scale < 1`, rounding rotated points to pixels can make two model points read the same
  scene pixel, which slightly inflates the score.

## Decision

- **Estimate once, resample the model, verify in a narrow band.**
  `ShapeModel::resample_at(s)` rebuilds every level from the stored teach points, through the
  same assembly code the model build uses. This is exact because pyramid coordinates are
  affine. The resampled model searches `(0.95, 1.05)`.
- `find_scale_invariant` returns the pose in the **original** model's frame, not in the
  resampled model's scaled one.
- **Two estimators, for different scenes:**
  - `estimate_scale_moments` segments an isolated blob and compares its outer radius with the
    model radius. It works on any model.
  - `estimate_scale_logpolar` needs teach points and an approximate centre. It correlates
    log-polar edge-density rasters with `corr::find`: a uniform scale becomes a row shift,
    Fourier–Mellin without an FFT. It compares edge density because a model stores edge
    points, not a reference image.
- **Point-collapse inflation at `scale < 1` is documented, not fixed.** It is small: the
  scale rows of [`docs/performance.md`](../../performance.md) include it. A test pins it as
  known behaviour.

## Alternatives

- **A wider scan.** Linear cost, and useless for models taught at a single scale.
- **A static bank of models pre-resampled at K scales.** Kept only for cluttered scenes with
  no prior on position or size (backlog).
- **Three deduplication designs for the point-collapse inflation, each measured and
  rejected:**
  1. Dedup in the rotation step whenever `scale < 1`. Subpixel refinement probes scales below
     1 even for single-scale models, so this moves the can-end baseline.
  2. Dedup only during the sweep. It slows the scale-range search measurably, for scores
     that only rank candidates.
  3. Dedup only in the reported score. The sweep and the final score then disagree: lowering
     the reported score can drop the true best candidate under `min_score` and promote a
     worse one, with a position error of a large fraction of a pixel.

  A future fix must make the sweep's candidate selection and the reported score agree on
  whether a duplicate counts.
- **Moment of gyration as the moments estimator.** A filled disc and its boundary ring have
  different radii of gyration (`R/√2` versus `R`), which biases a silhouette-to-model
  comparison. Rejected for the outer radius.

## Consequences

- Teach points are stored in the model (ADR-0005). A model without them cannot be resampled
  and says so.
- Scale-invariant search costs about the same at any range width.
