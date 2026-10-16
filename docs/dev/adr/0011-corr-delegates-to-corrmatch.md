# ADR-0011: Cross-correlation delegates to corrmatch

- Status: Accepted
- Date: 2026-10-12

## Context

Some tasks are photometric rather than geometric: tracking a textured window between frames,
or locating a patch with no reliable edges. `corrmatch` (same maintainer, on crates.io)
already implements SIMD ZNCC/SSD with pyramids, rotation and top-k search.
Registering substrate motion with foreground and stationary structures excluded needs
independent validity masks in both frames, including during refinement and reduction.
That contract is not represented by corrmatch's current high-level matching interface.

## Decision

- `corrmatch` is a regular, optional dependency behind the default-on `corr` feature, and
  the standard cross-correlation engine of this crate.
- **Ordinary matching is a thin wrapper.** It translates coordinates and errors, adapts `ImageView` (copying
  only when the view is strided), and defines its own `CorrConfig`/`CorrTemplateConfig`
  (invariant 10) instead of re-exporting corrmatch's `#[non_exhaustive]` types. `CorrMatch`
  is its own type: its score is a correlation coefficient, not `1 − occluded_fraction`, and
  there is no scale search.
- **Ordinary matching and `displacement` are `u8` only**, as corrmatch is. `u16`/`f32` support belongs upstream, not in a quantizing
  cast here.
- **`displacement` refines in this crate.**
  - Stage 1 is a bounded corrmatch search around the previous position.
  - Stage 2 is translation-only inverse-compositional Lucas–Kanade on the template's own
    gradient. A parabola fitted to a discrete correlation surface is biased towards integer
    positions, and only a second, differently biased estimator removes that bias.
- **Masked local registration is a narrow native exception.** `TranslationEstimator` accepts
  two image views, independently framed validity masks and an optional displacement
  prediction. Its bounded box-mean pyramid search uses jointly valid ZNCC, followed by
  robust Gauss–Newton refinement of translation, gain and offset. It is generic over
  `Pixel`, without quantization. It does not duplicate corrmatch's general matcher,
  rotated bank, beam search or SIMD engine, and uses no FFT.
- **Local similarity shares that exception.** `SimilarityEstimator` prediction-warps
  valid native samples to initialize the bounded translation search, then jointly
  refines uniform scale, rotation, translation and photometry. It uses the existing
  `Similarity2f` map and translation tuning/status types. It has no global angle or
  scale bank, no general matching engine and no FFT. Its independent regional fits
  compare maps over full-image corners; unobservable regions remain unavailable.
- **Excluded pixels contribute no evidence.** Every pyramid cell and interpolation
  footprint requires all source mask pixels to be valid. Exact bilinear derivatives
  use those same valid footprints. No border fill is registration evidence.
- **Acceptance is explicit.** Search ambiguity, support, conditioning, search limits,
  refinement convergence and spatially separated regional shifts qualify a result.
  Regions that lack texture cannot certify the motion model. Photometric residuals
  are reported in previous-image units; covariance is a local linearized approximation,
  not a tolerance or a guarantee of registration accuracy.
- **No sequence framework.** Scratch belongs to the reusable estimator; images, masks,
  filtering, camera choice, motion prediction, annotations and transport belong to the
  caller. Scratch grows during preparation and is reused without allocation at stable
  dimensions and configuration. Failed estimates never silently become accepted priors.

## Alternatives

- **A native ZNCC port**, to keep the shape matcher's validator independent of what it
  validates. Rejected: it would duplicate mature SIMD code with a worse copy. The cross-check
  (`tests/corrmatch_bridge.rs`) is about two different algorithms agreeing, not about
  build-graph independence.
- **Dev-dependency only.** That denies users the tool.
- **Whole-frame correlation with zero-filled exclusions.** Zero fill creates artificial
  stationary edges and allows excluded values into derivatives. Two-mask validity is
  part of the likelihood, not an image intensity convention.
- **Porting an application pipeline.** Foreground inference, frame pairing, ribbon
  labeling and inspection orchestration do not belong in the general-purpose library.

## Consequences

- A corrmatch release can raise this crate's MSRV (ADR-0002).
- General matching scale search and non-`u8` data wait on upstream (backlog).
- Masked translation and local similarity have their own accuracy rows, allocation gate, benchmarks and Python
  parity. Private sequence evaluation is development tooling and never writes source
  annotations. The existing `displacement` implementation and semantics are unchanged.
