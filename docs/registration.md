# Masked translation registration

`corr::TranslationEstimator` estimates the motion of image content between two views.
If a point is `p` in the previous image, its predicted current position is
`p + result.shift`. Coordinates refer to pixel centres; integer coordinates are pixel
centres, with no half-pixel offset added by the caller.
Coordinates are local to the supplied views. If crop origins differ, full-image content
motion is `shift + current_crop_origin - previous_crop_origin`.

```rust
use vision_metrology::corr::{TranslationConfig, TranslationEstimator, TranslationStatus};
use vision_metrology::Image;

let previous = Image::new_fill(64, 48, 0u8);
let current = Image::new_fill(64, 48, 0u8);
let valid = Image::new_fill(64, 48, 1u8);
let mut estimator = TranslationEstimator::new();
let result = estimator.estimate(
    &previous.as_view(), &current.as_view(),
    Some(&valid.as_view()), Some(&valid.as_view()),
    None, &TranslationConfig::default(),
)?;
if result.status == TranslationStatus::Accepted {
    let motion = result.shift.expect("accepted estimates have a shift");
    println!("content motion: {motion:?}, residual {:?}", result.residual_rms);
}
# Ok::<(), vision_metrology::Error>(())
```

The estimator accepts `u8`, `u16` and `f32` without quantization. Two inputs to one call
share a pixel type, but can have different dimensions or strides. Each optional `u8`
mask has its own image's dimensions; zero means excluded and nonzero means valid.
An absent mask makes the entire image valid. Non-finite unmasked f32 pixels are errors;
excluded values are never evidence.

## Search and refinement

The optional prediction centers the search; it is a displacement, not a position.
The default search extends 12 full-resolution pixels in each direction around zero.
Up to three levels of 2× box-mean reduction initialize a bounded integer ZNCC search.
Every reduced cell requires all four source pixels to be valid; odd trailing rows or
columns are dropped. A level without usable evidence falls back to a finer search.
The competing-peak margin is the minimum margin over the searched levels, so an alias
at a coarse level remains visible after local fine-level refinement. This checks
searched alternatives, not every possible translation in the image.

Robust Gauss–Newton estimates translation, gain and offset together. Intensities are
standardized for numerical conditioning; the reported photometric model is restored to
input units:

`previous(x) ≈ gain * current(x + shift) + offset`.

Huber weights limit isolated photometric outliers. All bilinear value and derivative
taps must be valid and in bounds, including zero-weight taps at integer coordinates.
There is no border fill in the evidence. The default iteration budget is 12 and the
convergence tolerance is 0.005 px (also applied to standardized photometric updates).

Illumination filtering and mask construction are caller choices. In a moving-camera
application, exclude stationary foreground structures and their shadows. Smooth
camera-fixed lighting can still bias raw-intensity registration; an appropriately
masked band-pass is useful when substrate texture must be separated from that lighting.
The estimator does not choose or learn such preprocessing.

## Interpreting the result

Use a shift as a fixture only when `status` is `Accepted`. Invalid configuration and
input shapes are errors; inability to register an image is a typed result:

| Status | Meaning |
|---|---|
| `InsufficientSupport` | Too few jointly visible pixels |
| `LowTexture` | Photometry or the translation Hessian cannot constrain the motion |
| `Ambiguous` | Weak ZNCC or a competing searched peak |
| `SearchBoundary` | The optimum reaches the configured search extent |
| `RefinementFailed` | Refinement cannot produce a finite converged solution within its budget |
| `ModelMismatch` | Independently refined regions disagree beyond the configured limit |

Failed results can retain a diagnostic shift. That value is not an accepted measurement.
RMS and maximum photometric residuals are in previous-image pixel units. The covariance
`[xx, xy, yy]` is in px², from the local normal equations with gain/offset eliminated and
scaled by the residual variance. It does not include model bias, mask errors, repeated
patterns or accumulated sequence drift, and is not a manufacturing tolerance.

The regional check refines four image quadrants where sufficient two-dimensional
texture exists. `regions` reports how many succeeded. `region_error` is the largest
pairwise shift disagreement; it is absent with fewer than two usable regions or when
`max_region_error` is `None`. Acceptance with an unavailable regional check does not
certify a translation-only scene. Observable rotation or deformation can be rejected;
there is no estimate of either in this operator.

The estimator owns reusable scratch. Its first call prepares capacity; changing sizes,
pyramid configuration or search capacity can grow it. At unchanged sizes and
configuration, subsequent native calls allocate nothing, even when mask contents change.
Filtering, image I/O, sequence bookkeeping and caller outputs are outside that contract.

## Python

```python
import numpy as np
import vision_metrology as vm

previous = np.zeros((48, 64), dtype=np.float32)
current = previous.copy()
valid = np.ones(previous.shape, dtype=np.uint8)
estimator = vm.TranslationEstimator(vm.TranslationConfig())
result = estimator.estimate(previous, current,
                            previous_mask=valid, current_mask=valid)
if result.status == "accepted":
    dx, dy = result.shift
```

Python accepts matching `uint8`, `uint16` or `float32` images and `uint8` masks, with
C-contiguous storage. Configuration is captured as a copy at construction. The usual
numpy-to-image conversion copies inputs; only the native numerical scratch has the
allocation-free guarantee. Python status names are the Rust names in snake case.

## Local masked similarity (experimental)

> **Experimental.** `SimilarityEstimator`, `SimilarityConfig` and `Similarity` are behind the
> non-default feature `corr-experimental`, and in Python under `vision_metrology.experimental`.
> They are outside the semver promise until the regional consistency check is recalibrated
> for real data. Translation's contract is unaffected.

`corr::SimilarityEstimator` adds uniform scale and rotation near a supplied prediction
(identity by default). The map is previous to current: `q = scale * R(angle) * p + translation`.
It uses the same mask-safe planes and typed outcomes as translation.

```rust
use vision_metrology::corr::{SimilarityEstimator, SimilarityConfig, TranslationStatus};
use vision_metrology::Image;
let a = Image::new_fill(64, 48, 0f32);
let mut estimator = SimilarityEstimator::new();
let result = estimator.estimate(&a.as_view(), &a.as_view(), None, None, None,
                                &SimilarityConfig::default())?;
if result.status == TranslationStatus::Accepted {
    let previous_to_current = result.transform.expect("accepted maps are present");
    println!("{previous_to_current:?}");
}
# Ok::<(), vision_metrology::Error>(())
```

- **Bounds, not a search.** `max_angle` and `max_log_scale` (0.05 each) limit departure from
  the prediction. A large search window can lose the initial capture under a small rotation,
  so give a nearby prediction. A bound is not a convergence guarantee.
- **Outcomes.** Rank loss is `LowTexture`. `region_error` is absent when fewer than two regions
  are supported, and missing regions cannot certify a similarity model. Rejected results may
  keep diagnostic maps.
- **Covariance.** The 4×4 covariance covers centre x/y, angle and log-scale, from a local
  normal-equation approximation. It does not quantify sequence drift or statistical coverage.
- **Composition.** For differing crop origins the full map is
  `T(current_origin) * local_map * T(-previous_origin)`. To compose accepted maps, multiply
  `current_map * previous_seed_to_previous_map`, and transport a normal with the rotation
  before normalizing it: uniform scale changes lengths.
- Do not treat failed maps, transformed predictions or covariance as new boundary evidence.

The measured behaviour, the known-transform accuracy and the private-sequence results are in
[`tools/registration_eval/README.md`](../tools/registration_eval/README.md#results).

## Anchored-sequence development experiment

The anchored-sequence experiment is an evaluation, not an API. Its method and results are in
[`tools/registration_eval/README.md`](../tools/registration_eval/README.md#similarity-reciprocal-sequences-and-review-similarity_runpy).
