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
There is no border fill in the evidence. Each refinement freezes its observations at the
start: the valid previous pixels whose current-image taps stay valid for every shift
within ±½ px of the seed. The set is re-frozen only if the shift leaves that window, so
taps entering or leaving validity cannot make the solve chatter. The Huber scale is
estimated at the seed and once after the first step, then fixed. Convergence tests the
translation step alone, against a tolerance of 0.005 px; the default budget is 30
iterations, which slow pairs need (on a dark, mostly flat substrate the robust scale is
small and the solve creeps) and converging pairs do not spend.

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
texture exists; `regions` reports how many succeeded. With at least two, every pair of
regional shifts is compared by a covariance-normalized test,
`χ² = dᵀ (C_i + C_j + b² I)⁻¹ d`, where `d` is the shift difference, `C_i` and `C_j` the
regional covariances and `b` the configured model-bias floor. The largest χ² is
`region_chi2`; above `RegionCheck::max_chi2` the result is `ModelMismatch`.
`region_error` is the largest pairwise shift distance in px, for diagnosis. Both are
absent with fewer than two usable regions or when `region_check` is `None`.

| `RegionCheck` field | Default | Meaning |
|---|---|---|
| `max_chi2` | 13.8 | largest accepted χ²; the 99.9% quantile of χ² with 2 degrees of freedom |
| `bias_floor_px` | 0.06 | model bias added in quadrature to each component, px |

The covariance assumes independent residuals; the floor carries what it does not:
interpolation bias and the noise correlation a caller's band-pass introduces. With
negligible noise the test tolerates about `bias_floor_px · sqrt(2 · max_chi2)` of
disagreement between two quadrants; noisy pairs are allowed proportionally more. The
defaults were chosen for at most 1% false rejects of true translations on a synthetic
DoG-band texture under read noise and on HPatches development self-pairs, and an
observable rotation of a 320×97 strip is rejected; [performance.md](performance.md) has
the numbers, including the coverage this costs on real strips, whose quadrants disagree
by more than noise. Raising `bias_floor_px` tolerates more model error; that is the
caller's tolerance, not a recalibration. Acceptance with an unavailable regional check does not certify a
translation-only scene, and there is no estimate of rotation or deformation in this
operator.

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

`vm.RegionCheck(max_chi2=..., bias_floor_px=...)` configures the regional test, and
`TranslationConfig(region_check=None)` disables it; results carry `region_chi2`.
Python accepts matching `uint8`, `uint16` or `float32` images and `uint8` masks, with
C-contiguous storage. Configuration is captured as a copy at construction. The usual
numpy-to-image conversion copies inputs; only the native numerical scratch has the
allocation-free guarantee. Python status names are the Rust names in snake case.

## Local masked similarity (experimental)

> **Experimental.** `SimilarityEstimator`, `SimilarityConfig` and `Similarity` are behind
> the non-default Cargo feature `corr-experimental`, and in Python under
> `vision_metrology.experimental`. They are outside the semver promise until the evidence
> supports promotion: they may change or be removed in any release. Translation's
> contract is unaffected.

`corr::SimilarityEstimator` adds uniform scale and rotation near a supplied
`Similarity2f` prediction (identity by default). Its map is previous to current:
`q = scale * R(angle) * p + translation`. It uses the same mask-safe planes and typed
outcomes as translation. Ordinary template matching remains delegated to corrmatch.

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

```python
from vision_metrology.experimental import SimilarityConfig, SimilarityEstimator

estimator = SimilarityEstimator(SimilarityConfig())
result = estimator.estimate(previous, current, previous_mask=valid,
                            current_mask=valid, prediction=(0., 0., 0., 1.))
if result.status == "accepted":
    tx, ty, angle_radians, scale = result.transform
```

`SimilarityConfig.translation` reuses support, initializer score/margin, translation
search extent, robust tuning and the optional regional check; `SimilarityConfig::default()`
sets similarity's own `RegionCheck` (below). Similarity uses 30 iterations by default. `max_angle` defaults to 0.05 radians and `max_log_scale`
to 0.05. These limit departure from the prediction, rather than providing an exhaustive
angle/scale search. Native refinement gates x/y centre displacement in current pixel
units; initialization searches residual translations in the prediction-warped previous
frame. For non-unit predicted scale/rotation those search boxes differ.
A large window can lose initializer capture under a small rotation: use an explicit
nearby prediction. A bound is not a convergence guarantee.

Prediction-warped masked sampling initializes the existing translation search. The
initializer's integer correlation and margin are reported, not a global similarity
correlation bank. The initializer runs the search alone, with no translation
refinement: an integer peak that passes the score, margin and boundary gates seeds the
similarity solve directly. A robust six-parameter forward-additive
solve then refines four motion coefficients and gain/offset against the original
current plane. Coordinates are centred at `((width-1)/2, (height-1)/2)` and normalized
by the radius to the corner, keeping rank checks comparable between sizes. The two
linear coefficients parameterize uniform scale and rotation without introducing shear.

Each quadrant is refitted about its own centroid `c` and RMS radius `ρ` (of its valid
previous pixels), and compared with the global fit there, with no extrapolation to the
image corners: `d = [Δq(c), ρ·Δangle, ρ·Δlog-scale]` in px, normalized by the regional
4×4 covariance (scaled to px) plus the floor, χ² with 4 degrees of freedom. For nested
fits the regional covariance bounds that of the difference, so the test is conservative.
Similarity's defaults are `max_chi2` 18.5 (the 99.9% quantile for 4 degrees of freedom)
and `bias_floor_px` 0.06; the rejection rates of mild anisotropic scales are in
[performance.md](performance.md). `region_error` is `|Δq(c)| + ρ·|(Δangle, Δlog-scale)|`,
a bound on the two maps' disagreement within radius `ρ`; it and `region_chi2` are absent
when fewer than two regions are supported. Rank loss is `LowTexture`; missing
regions cannot certify a similarity model. Rejected results may retain diagnostic maps.
The 4×4 covariance is for centre displacement x/y, angle and log-scale; it is a local
normal-equation approximation with photometry eliminated. It does not quantify global
sequence drift or certify statistical coverage. Photometry retains the same input-unit
contract. Stable-size prepared native similarity calls also allocate nothing.

For differing crop origins, the full map is
`T(current_origin) * local_map * T(-previous_origin)`, where `T` is translation.
To compose accepted maps, multiply `current_map * previous_seed_to_previous_map`.
Transport a normal with the rotation, then normalize it; uniform scale changes lengths.
Do not treat failed maps, transformed predictions or covariance as new boundary evidence.

## Anchored-sequence development experiment

The caller-owned tooling in `tools/registration_eval` retains chained adjacent maps and
independently registered seed-to-current anchors. Both adjacent steps and anchors require
accepted reverse registration and corner closure at most 0.25 px before constraining a
sequence. Loss of evidence starts a new segment. An anchor has no authority without
support, and constraints cannot correct inherited anchor errors or absence of overlap.

The synthetic 64-frame noisy fixture verifies anchors against known transforms, showing
lower mean accumulated error and anchor corner error below 0.1 px. Private-data runs
have no independent motion truth. With the recalibrated regional check, similarity
accepts more of them than before, but its reciprocal closure on accepted pairs is not
yet as good as translation's, so it stays experimental. See the measured accuracy,
coverage, closure, rejection counts and latency in [performance.md](performance.md).
