# Performance and accuracy

Measured numbers for the library: speed, accuracy on synthetic ground truth, and results
on real captures. Speed is measured on an Apple M4 Pro, single thread, release profile.
Every accuracy figure is the worst cell of a deterministic sweep, and the test suite fails
if it is exceeded.

## Speed

Shape matching on a 1280×1024 scene with an 800-point model:

| Operation | Time |
|---|---|
| model creation | 0.49 ms |
| full 360° `find`, clean scene | 3.4 ms |
| full 360° `find`, heavily cluttered scene | 6.5 ms |
| tracked `find` (±60 px ROI, ±10° prior) | 1.5 ms |
| full 360° `find`, `greediness = 0.0` | 5.3 ms |
| `find` over a 0.8–1.25× scale range | 16.8 ms |
| full-frame direction field (not computed during `find`) | 4.0 ms |
| 2-D edge detection, `u8`, full frame | 5.6 ms |

On the cluttered scene the time splits into the top-level sweep (about 2.3 ms) and the
descent of well-scoring candidates (about 4.2 ms).

Scale-invariant search (estimate, resample, verify) is about 2.2–2.4× faster than scanning
the equivalent scale range on the same scene, with the same found-rate and accuracy.

Other operators:

| Operation | Time |
|---|---|
| `fit_circle`, 500 points | 2.6 µs |
| `fit_circle`, 500 points, Tukey loss | 4.2 µs |
| `fit_line`, 500 points | 1.7 µs |
| `fit_ellipse`, 100 points | 1.6 µs |
| `fit_ellipse` with RANSAC, 1000 points | 430 µs |
| `warp::Map::apply`, affine, 640×480, bilinear | 510 µs |
| `warp::Map::apply`, polar, 640×480, bilinear | 494 µs |
| `corr::find`, VGA scene, 64×64 template, rotation off / on | 4.60 / 22.1 ms |
| `corr::displacement`, 320×97 window, quadratic / + Lucas–Kanade | 1.60 / 1.71 ms |

## Accuracy envelopes

Each row sweeps a synthetic fixture with known subpixel ground truth (anti-aliased edges
from an analytic Gaussian-CDF profile, seeded uniform noise of up to 5 LSB quantized to
`u8`). It reports the worst bias and standard deviation found anywhere in the sweep. The
envelope is what `tests/accuracy.rs` enforces, at about 1.5× the measured value.

| Operator | Unit | Worst \|bias\| | Worst σ | Envelope (bias / σ) |
|---|---|---:|---:|---|
| `Edge1DDetector` position | px | 0.042 | 0.095 | 0.07 / 0.15 |
| `Edge2DDetector` position | px | 0.085 | 1.195 | 0.13 / 1.80 |
| `Caliper` (rect) position | px | 0.026 | 0.062 | 0.05 / 0.12 |
| `fit_circle` radius and centre | px | 0.325 | 1.135 | 0.50 / 1.75 |
| `ShapeMatcher` translation | px | 0.003 | 0.011 | 0.02 / 0.03 |
| `ShapeMatcher` rotation | deg | 0.000 | 0.001 | 0.05 / 0.05 |
| `ShapeMatcher` scale, 0.5–2.0× | fraction | 0.0014 | 0.0003 | 0.003 / 0.001 |
| `ShapeMatcher` position under scale, 0.5–2.0× | px | 0.0218 | 0.0039 | 0.04 / 0.01 |
| estimate-then-verify scale, 0.5–2.0× | fraction | 0.0014 | 0.0003 | 0.003 / 0.001 |
| estimate-then-verify position | px | 0.0218 | 0.0039 | 0.04 / 0.01 |
| rectified crop repeatability | 8-bit grey levels | 0.88 | 1.69 | 1.3 / 2.5 |
| `corr::displacement`, quadratic | px | 0.0239 | 0.0164 | 0.04 / 0.025 |
| `corr::displacement`, Lucas–Kanade | px | 0.0204 | 0.0141 | 0.035 / 0.022 |

How to read the hard rows:
- **`Edge2DDetector`.** The worst cell is blur σ = 3 with 5 LSB noise.
- **`fit_circle`.** The worst cell is a 30° arc (nearly a chord) with 10% gross outliers.
  At arcs of 90° or more it is well inside 0.05 px.
- **Scale rows.** Every scale in the sweep found every rotation.

## Real data: shape matching

`examples/pose_audit` checks every recovered pose against corrmatch's masked ZNCC of the
pose-warped reference against the scene, an independent algorithm over different data.
The data is 1280×1024 beverage can ends: the model is built from the first frame of each
folder, the search is full 360°, and `min_contrast` is tuned per folder (see the
[shape-matching guide](shape-matching.md#getting-a-clean-model)).

| Folder | Found | Median score | Median ZNCC | Min ZNCC | Angle coverage | Median time |
|---|---|---|---|---|---|---|
| dome illumination | 50 / 50 | 0.998 | 0.961 | 0.916 | 321° | 5.6 ms |
| bright field | 50 / 50 | 0.997 | 0.915 | 0.850 | 322° | 6.6 ms |
| dark field | 50 / 50 | 0.962 | 0.953 | 0.847 | 321° | 8.3 ms |
| second product, dome | 48 / 48 | 0.997 | 0.939 | 0.907 | 336° | 26.0 ms |
| third product, bright field | 19 / 19 | 0.998 | 0.929 | 0.821 | 231° | 11.2 ms |
| production line, bright field | 39 / 39 | 0.978 | 0.921 | 0.823 | 46° | 9.5 ms |

Angle coverage is 360° minus the largest gap between recovered angles. A deliberately wrong
angle or a 25 px offset lowers the ZNCC by 0.3 or more, which `tests/corrmatch_bridge.rs`
pins.

`pose_audit xcheck` runs corrmatch's own rotation-enabled search next to `ShapeMatcher` and
compares the two poses. Over 20 frames per folder, the disagreement is |Δpos| p95
0.87–1.31 px and |Δangle| p95 0.35–0.66°. These figures add up both matchers' errors,
including corrmatch's rotation-grid quantization.

## Real data: the can-end chain

`examples/inspect_canend` locates the tab, takes its pose as the fixture, measures the rim
with 96 radial calipers in the tab's frame, and fits a circle with a Tukey (2 px) loss.
Tolerance is 2 px on `max_dev`.

| Folder | Frames measured | Mean radius | σ | Per-frame rms |
|---|---|---|---|---|
| dome illumination | 50 / 50 | 365.237 px | 0.282 px | 0.20–0.54 px |
| dark field | 50 / 50 | 365.696 px | 0.307 px | 0.28–0.60 px |

All 96 calipers survive the robust fit in every frame. σ is an upper bound on
repeatability, because it is measured over different physical cans, so part-to-part
variation of the rim is inside it.

The can-end dataset is not distributed with this repository.
