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

Bead tracking on a 1280×1024 frame: an S-shaped bead 50 px wide and 1080 px long, its
prior 2 px off and bent by a further 1 px, every configured pass run:

| Operation | Time |
|---|---|
| `track`, 100 stations, 1 pass | 0.44 ms |
| `track`, 300 stations, 1 pass | 1.30 ms |
| `track`, 300 stations, 3 passes | 2.54 ms |
| `track`, 300 stations, 3 passes, reach ±30 px | 2.97 ms |
| `track`, 300 stations, 3 passes, 30% of the bead in gaps and a distractor beside it | 2.58 ms |
| `explain_bead`, 300 stations, 3 passes | 3.60 ms |

Where the time goes, in the 3-pass case:
- **Each strip costs about 2.1 µs.** A call measures one strip per station in each pass and
  once more in the final stage: 1200 strips here.
- **The caliper is nearly all of it.** Sampling the strips' profiles takes 75% of the time,
  and smoothing them and locating their edges 21%. The solves, the pairing and the curve
  geometry take about 3%.
- **Most strips refill the caliper's smoothing kernel.** A strip's sample spacing differs
  from the last one's by a few ulps, so 955 of the 1200 strips refill the kernel in place,
  about 37 ns each: 1.4% of the call.
- **A longer reach lengthens every tracking strip, and a gap costs as much as the bead.**
  The strips in a gap are sampled all the same.

## Accuracy envelopes

Each row sweeps a synthetic fixture with known subpixel ground truth. It reports the worst
bias and standard deviation found anywhere in the sweep. The envelope is what
`tests/accuracy.rs` enforces, at about 1.5× the measured value.

The first table's fixtures are anti-aliased edges from an analytic Gaussian-CDF profile,
with seeded uniform noise of up to 5 LSB, quantized to `u8`.

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

### Strips and calipers on CaliperBench's image model

These rows use the image model of CaliperBench's synthetic tiles:

- straight edges blurred by a Gaussian PSF of σ 0, 0.6, 1.2 or 2.5 px and integrated
  exactly over each pixel;
- edges at 0, 10, 30 and 45° to the pixel grid, at subpixel phases 0, ¼, ½ and ¾;
- seeded Gaussian noise of 0, 2 or 5 DN on a 160 DN step, quantized to 8 bits.

Strips are 40 px long with 81 samples and use CaliperBench's textbook settings: σ of one
sample, a radius-3 Gaussian then central differences, `threshold` 0.01 on a `[0, 1]`
image. The rect, arc and radial calipers use the default `MeasureConfig`.

| Operator | Unit | Worst \|bias\| | Worst σ | Envelope (bias / σ) |
|---|---|---:|---:|---|
| strip step, `GradientPeak` (parabola) | px | 0.176 | 0.805 | 0.27 / 1.2 |
| strip step, `GradientPeak` (log-parabola) | px | 0.176 | 0.805 | 0.27 / 1.2 |
| strip step, `MidpointCrossing` | px | 0.032 | 0.175 | 0.05 / 0.27 |
| strip step, `HalfContrast` | px | 0.045 | 0.170 | 0.07 / 0.26 |
| strip bar centre, widths 2–10 px | px | 0.191 | 0.735 | 0.29 / 1.1 |
| strip bar width, 10 px, noise-free | px | 0.071 | 0.120 | 0.11 / 0.18 |
| strip bar width, 2 and 3 px, noise-free | px | 3.379 | 0.224 | 5.1 / 0.34 |
| strip 15° off the edge normal, 3 px wide | px | 0.083 | 0.659 | 0.13 / 1.0 |
| strip 30° off the edge normal, 3 px wide | px | 0.168 | 0.801 | 0.26 / 1.2 |
| `Caliper` (rect), pixel-integrated step | px | 0.067 | 0.304 | 0.10 / 0.46 |
| `Caliper` (radial and arc), radii 20 and 40 px | px | 0.074 | 0.229 | 0.11 / 0.35 |

How to read them:
- **The worst cell is PSF σ 2.5 px with 5 DN noise, in every row.**
  - The textbook smoothing, one 0.5 px sample, leaves the broad gradient peak of such an
    edge to the noise. That is where the gradient rows' σ of 0.8 px comes from.
  - The level methods read the smoothed profile itself and stay under 0.18 px.
  - Without noise, every step row is within 0.06 px of bias and 0.07 px of σ.
- **Narrow bars read wide.**
  - A bar of 2 or 3 px under a 2.5 px PSF reads up to 3.4 px too wide. The two edges'
    responses overlap and push their peaks apart.
  - At 10 px the width is within 0.07 px.
  - The bar's centre is unbiased by symmetry, and its rows include noise.
- **Radial and arc calipers.** `MeasureRadial` reads the disc's radius with at most 0.022
  px of bias at any caliper width, which is the reason it averages along the arc.
  `MeasureArc` reads a spoke's position along the arc to 0.074 px.

### The bead tracker

`BeadTracker` with the default config tracks the bead fixture of `tests/accuracy/bead.rs`:
- a line at 17°, an arc of radius 150 px, or a sine of amplitude 8 px and period 160 px;
- a light bead 8, 30 or 60 px wide, blurred by σ 0.8 or 1.5 px across the centreline and
  integrated over each pixel;
- seeded Gaussian noise of 0, 2 or 5 DN on 140 DN of contrast, quantized to 8 bits;
- a prior 2 px off the centreline and bent by a further 1 px over the bead's length.

The width range is half to one and a half times the bead's width. A cell pools every
station of 5 noisy trials.

| Quantity | Unit | Worst \|bias\| | Worst σ | Envelope (bias / σ) |
|---|---|---:|---:|---|
| final centre, distance from the true centreline | px | 0.0113 | 0.0869 | 0.017 / 0.13 |
| final width, minus the true width | px | 0.0213 | 0.1696 | 0.032 / 0.25 |
| refined centreline, distance from the true one | px | 0.0119 | 0.0487 | 0.018 / 0.073 |

How to read them:
- **The worst cell is σ 1.5 px with 5 DN of noise, in every row.** Without noise, every bias
  is under 0.011 px and every spread under 0.016 px, at every width.
- **The refined centreline is quieter than the final centres.** A final centre is one
  station's pair; the curve is the regularised solve over all of them.
- **On the arc, the curve sits up to 0.01 px towards the centre of curvature.** A tracking
  strip averages lines up to 2 px either side of its station, and on a bend a straight
  strip reads the centre towards the concave side ([limitations](bead.md#limitations)).
  On a 50 px bead along the same arc, a tracking `half_width` of 0 instead of 2 takes the
  curve's mean offset from 0.010 to 0.004 px.

## Bead tracking: convergence

The basins are pinned by `tests/bead.rs`, which fails if one shrinks by more than 10%.

**The fixtures.** The arc (R = 150 px, 210 px long) and a sine (amplitude 12 px, period
160 px, 253 px long), each with a bead 30 px wide, blurred by σ 1.2 px, under 2 DN of
noise. They are tracked with the default config, except that `min_width` is 20 px.

**The perturbations.** The prior is the true centreline from 15 px in from each end,
moved off it by one of the perturbations below.

**Converging** means that after the default 3 passes every refined station is within
0.1 px of the true centreline. A basin is the largest perturbation that converges.

| Perturbation of the prior | Arc | Sine |
|---|---:|---:|
| translation along the normal, either way | 15.0 px | 15.0 px |
| rotation about its midpoint | 21.0° | 15.9° |
| a sine of wavelength `L/2` along the normal (105 and 126 px) | 7.2 px | 12.8 px |
| a sine of wavelength `L/4` (53 and 63 px) | 3.3 px | 4.1 px |
| a Gaussian bump at the middle, σ 5 px of arc | 0.91 px | 0.69 px |
| a Gaussian bump, σ 15 px | 15.5 px | 16.4 px |
| 5 px of translation, plus a rotation | 16.9° | 16.3° |
| 5 px of translation, plus a bump of σ 15 px | 10.3 px | 10.0 px |

How to read it:
- **A translation converges up to the tracking reach, `track.max_offset`.** Beyond it, every
  station is rejected: with `Offset`, then with `NoPair` once a strip holds only one of
  the bead's edges, then with `Caliper(NoEdge)`. At no translation tried, up to 80 px,
  does a station take anything but the bead for a hit.
- **A rotation converges well beyond the reach at the ends.** 21° moves the arc's ends by
  38 px. A rotation is a straight correction, which the bending penalty does not resist,
  so the stations within reach carry the rest.
- **Local error is what the bending length decides.** A bump 5 px wide converges only
  below 1 px.
- **Past a rotation's or a sine's basin, the stations left off the bead are at the curve's
  ends;** past a bump's, they are at the bump.

### The bending length

The local-deformation basins on the arc, and the noise the refined centreline keeps on a
straight bead 30 px wide whose prior is 2 px off and bent by a further 1 px, averaged over
8 seeds:

| `bending_px` | Bump σ 5 px | Bump σ 15 px | Sine at `L/4` | Coarsest polygon | Curve rms / max, 2 DN | Curve rms / max, 5 DN |
|---|---:|---:|---:|---:|---:|---:|
| 8 | 0.22 px | 3.0 px | 0.62 px | a vertex every 14 px | 0.009 / 0.023 px | 0.021 / 0.049 px |
| 4 (default) | 0.91 px | 15.5 px | 3.3 px | every 17 px | 0.011 / 0.027 px | 0.028 / 0.064 px |
| 3 | 2.0 px | 15.5 px | 9.6 px | every 32 px | 0.013 / 0.030 px | 0.030 / 0.071 px |
| 2 | 13.2 px | 15.5 px | 10.5 px | its two ends | 0.014 / 0.033 px | 0.034 / 0.079 px |

- **A shorter bending length widens every local basin and lets more noise into the curve.**
  The noise columns do not change with more passes, because the curve converges in fewer.
- **More passes help little.** With 6 passes instead of 3, no basin widens by more than
  1.7 px, except that `bending_px` 2 then converges from a bump of σ 15 px as tall as
  24 px, beyond the reach.
- **A schedule that shortens the bending length from call to call pays only with more
  passes.** It was tried by feeding each result to the next call as its prior.
  - 8, 4 and 2 px with one pass each lets in the noise of 3 px, but its local basins are
    no wider than those of a fixed 3 px over the same 3 passes, and some narrower.
  - With two passes each, every basin is wider than a fixed 3 px over the same 6 passes,
    at the same noise; it converges even from the polygon of the arc's two ends.

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
