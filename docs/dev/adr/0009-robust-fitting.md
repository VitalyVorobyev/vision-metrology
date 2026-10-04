# ADR-0009: Robust fitting: algebraic start, geometric refinement, reported residuals

- Status: Accepted
- Date: 2026-08-19

## Context

Fitted primitives are the measured dimensions. They are fitted to caliper hits that contain
outliers (print marks, highlights, a neighbouring edge), often on short arcs, and the caller
needs to know whether a fit can be trusted.

## Decision

- **Every fitter starts algebraically and then refines geometrically.** It returns
  `Fit<M>` with `rms`, `max_dev` and `n_used` (invariant 21).
- **`fit_circle` starts from Taubin**, which is nearly unbiased on short arcs, then runs
  Gauss–Newton on the true residual `‖p − c‖ − r`. On short arcs this is the difference
  between visible bias and a subpixel fit ([`docs/performance.md`](../../performance.md)).
- **Robust losses (Huber, Tukey) use graduated non-convexity.** `RobustLoss::annealed`
  starts wide and shrinks geometrically. A fixed Tukey radius applied to a contaminated
  start rejects the inliers and keeps whatever the bad start passed through.
- **RANSAC (`FitConfig::ransac`) handles gross outliers that flip the initial model.** For
  example, an outlier can make total least squares return a line orthogonal to the true
  one, and no reweighting recovers from an orthogonal start. Every fitter honours it.

## Alternatives

- **Kåsa circle fit.** It collapses towards the chord on short arcs.
- **Reweighting only (IRLS without annealing or RANSAC).** It fails exactly in the two cases
  above.

## Consequences

- A caller can gate on form tolerance (`max_dev`) directly.
- Robust fitting costs a few microseconds per fit (see `docs/performance.md`), negligible
  next to the image work.
