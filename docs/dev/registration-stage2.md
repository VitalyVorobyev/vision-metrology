# Masked similarity and anchored-sequence design

This finite second stage builds on masked translation. The reusable operator belongs in
`corr`; private sequence adaptation, pose bookkeeping, overlays and interactive review
remain development tools. Two callers are inter-frame substrate registration and local
reference-to-current fixture localization. Ordinary teach/find remains in corrmatch.

## Contract

The three types are behind the non-default `corr-experimental` feature, and in Python
under `vision_metrology.experimental`, until the evidence supports promotion (backlog,
Measurement).

Add `SimilarityEstimator`, `SimilarityConfig`, and `Similarity` using the existing
`Similarity2f`, `TranslationConfig`/`TranslationTuning` and typed registration outcomes.
The map is previous pixel-centre coordinates to current coordinates, `q = s R p + t`.
The caller supplies an optional prediction. Search bounds apply to centre displacement,
rotation and log-scale departure from that prediction. This is local registration;
there is no exhaustive angle/scale bank or FFT search.

Masked sampling of the prediction initializes the existing bounded translation search.
A robust six-parameter solve refines translation, the two similarity coefficients,
gain and offset on native images. Centre-relative, radius-normalized coordinates keep
conditioning comparable between image sizes. Numerical work accumulates in f64; stored
coordinates/results remain f32. Every interpolation and derivative tap must be valid.
Rank, support, search bounds, photometry, convergence and independently fitted regions
qualify a result. Missing regional checks remain explicitly unavailable. The local 4×4
covariance describes centre displacement, angle and log-scale, not manufacturing uncertainty.

Translation's implementation and public contract remain unchanged. Fixed-size solves,
reused planes and warp/search/refinement buffers keep prepared native calls allocation-free.
Python receives matching configuration, prediction, transform, diagnostics and stubs.

## Anchors and visualization

Development tooling composes accepted previous-to-current maps into seed-to-current maps.
It also attempts independent keyframe-to-current registration with an explicit prediction.
A supported anchor replaces the accumulated pose and reports correction/closure at
image corners; a failed anchor remains unavailable, without fabricated constraints.
Camera transitions start a new segment. Direct anchors cannot remove error inherited by
an anchor itself; both chained and corrected trajectories are retained for comparison.
No sequence manager or pose graph becomes public API in this stage.

The tooling reads a neutral sequence manifest; a private dataset's adapter writes it,
from the git-ignored `tools/private/`. The raw multi-strip sequence G has no camera
groups, masks or ribbon annotations: its adapter writes one manifest per strip, with a
caller-selected texture ROI as the only mask, and records that provenance. It does not
invent a seed. For sequences A–F, the caller's camera groups and masks are authoritative.
HTML viewers embed private pixels only into local, untracked output directories; no remote
scripts or hosted assets.

## Acceptance

- Analytic positive/negative rotations and scales, gain/offset, noise, masks, partial
  overlap, stride/odd sizes, rank loss and search-bound failures. Clean corner error
  below 0.1 px, angle/log-scale error with pinned accuracy envelopes.
- Synthetic long sequences compare composed and independently anchored maps against
  known truth; anchor failures and camera handovers remain explicit.
- Python parity and zero-allocation prepared-call tests; representative strip/larger
  window benchmarks and complete repository gates.
- Local reports and overlays for G and for the grouped sequences A–F; coverage,
  failures, closure and latency remain visible. No cached estimate is motion truth.

Similarity may still fail under weak spatial texture, large changes, local minima,
non-similarity deformation or insufficient overlap. Private-data coverage alone cannot
certify accuracy; promotion rests on synthetic known-transform tests and honest diagnostics.
