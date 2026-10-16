# ADR-0022: Planar rig substrate pose lives in `metric::rig`

- Status: Accepted
- Date: 2026-10-16

## Context

Several fixed views look at one planar substrate that moves rigidly under them. Each view
has a plane→image homography from a calibration, and measures one masked translation per
frame pair in native pixels. The caller needs one rigid SE(2) pose per frame for the
substrate in the common plane frame, with provenance (#74). A downstream application found
what shapes the operator:
- lifting one translation at several points of a wide view biases the rotation to zero;
- a plain least-squares start defeats Huber when one of three views is gross;
- one view constrains translation only;
- a split chain on every failure fragments the sequence into single frames;
- views that agree do not prove the scale right.

The operator turns pixel measurements into plane millimetres through calibration
homographies, which is ADR-0003's job. It consumes `corr::TranslationEstimator`'s output
but does not register images.

## Decision

- **Home: `metric::rig`, a public submodule of `metric`**, behind the `metric` feature.
  It is a calibration-bridge operator: per-view pixel measurements in, plane-frame poses
  out. The canonical paths are `metric::rig::*`; the prelude does not carry them.
- **Contract.** `RigPoseConfig` plus a reusable `RigPoseEstimator` (invariant 10).
  `begin(views, cfg)` validates and sizes the scratch, `step(shifts, uses)` allocates
  nothing (invariant 6), and `estimate` runs a sequence and owns its output. A step's
  input is one `Option<ViewShift>` per view: `d_px`, its covariance, the acceptance, the
  region centroid and an optional closure. Every frame reports its `RigStep` (`Anchor`,
  `Measured`, `SingleView`, `Bridged`), the pose and increment with covariances, the
  residual in plane units (invariant 21) and the residual scale. Every step and view
  reports a `ViewOutcome`, its weight, its lift point, its lifted displacement and its
  residual. The trajectory carries the sequence statistics.
- **Lift at one point**, the region centroid: the exact plane displacement between
  `H⁻¹(c)` and `H⁻¹(c + d)`, with the covariance through the Jacobian at the midpoint.
  Rotation comes from the lever arms between views.
- **Solve per step** for the increment `p' = R(θ)p + t`, parametrised about the mean lift
  point. Gauss–Newton from `θ = 0`, whose first iteration is the small-angle weighted
  least squares, weighted by each view's lifted information.
- **Robustness in two stages.**
  - Huber IRLS on the Mahalanobis residuals, its weights started from the cross-view
    median displacement.
  - Then, by default, a graduated Tukey biweight (ADR-0009's annealing), from two starts:
    the Huber solution and the constant-velocity prediction. With three views a gross
    view can be traded for a rotation that makes a good view look gross at equal cost.
    The prediction's Mahalanobis distance chooses between the two minima. It never moves
    either: the step stays a measurement.
  - The weights work in units of the stated σ. Registration covariances are optimistic,
    so `shift_sigma_floor_px` (default 0.05 px) states what a shift can claim. A scale
    estimated with the weights is pulled up by the gross view it should expose, so the
    residual scale (median Mahalanobis residual, at least 1) only inflates the reported
    covariance.
- **No usable view: bridge.** The prediction (the last increment plus the motion noise at
  the last lever centre) is the step, marked `Bridged`. The chain is never split.
- **One usable view, or lift points at one place: hold the rotation** at the prior
  (`RotationHold::Previous` or `Zero`, with its σ), and mark the step `SingleView`.
- **Scale is a separate diagnostic.** `path_length_ratio` compares the path of a substrate
  point with an external one. Cross-view agreement cannot check scale: a common
  plane-height error scales every view's lift alike and cancels between views.
- **Smoother.** `RigPoseEstimator::smooth` is a Rauch–Tung–Striebel smoother over the
  increments at the sequence's lever centre, a random walk with the configured motion
  noise. It has a Mahalanobis gate, and bridged steps are unobserved. Smoothed
  increments carry covariances. Smoothed poses do not, because the smoothed increments
  are correlated.
- **Python** gets the batch API, numpy in and out: `estimate`, `smooth` and the
  diagnostics. The streaming `begin`/`step` is not bound.

## Alternatives

- **A `rig` crate root module or a `sequence` module.** The operator's substance is the
  calibration lift; a new top-level module would duplicate `metric`'s units and
  conventions.
- **Lifting each view's translation over its whole region** (several points). It asserts
  zero in-view rotation.
- **Huber only, as the downstream prototype validated.** With three views it bounds a
  gross view's pull without removing it: the good views' residuals grow to balance it.
- **A scale estimated alongside the weights.** One gross view among three inflates the
  median residual and switches the robust weights off.
- **Reusing `measure::numeric`'s Whittaker smoother.** It is not covariance-aware, and
  ADR-0020 keeps it inside `measure`.
- **Starting a new chart on every failure.** It fragments the sequence.

## Consequences

- A gross view is visible per step (`Rejected`, weight 0), and so is a covariance that is
  too optimistic (`residual_scale` > 1).
- Three views cannot always attribute a gross error from one step alone. Under smooth
  motion the prediction decides; under abrupt motion with a gross view, the step can pick
  the wrong view, and the smoother's gate is the second line.
- The lift point is the caller's region centroid. A registration whose effective centroid
  differs (gradient-weighted support) adds rotation × offset to the shift; the caller
  chooses the region.
