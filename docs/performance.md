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

Continuous profiles (`ProfileSampler`, measured 2026-10-16) on a 1280×1024 `u8` frame,
stations on a circle of radius 300 px with radial directions, offsets 0.5 px apart:

| Operation | σ 1 px | σ 2.5 px |
|---|---:|---:|
| 100 stations × 11 offsets | 128 µs | 572 µs |
| 100 stations × 41 offsets | 495 µs | 2.05 ms |
| 400 stations × 11 offsets | 510 µs | 2.17 ms |
| 400 stations × 41 offsets | 2.00 ms | 6.29 ms |
| one strip caliper, 40 px, 41 samples (`Derivative::Continuous`) | 5.4 µs | 23.5 µs |

A sample costs about 0.12 µs at σ 1 px and 0.4–0.5 µs at σ 2.5 px, growing with the
support's `π (4σ)²` pixels. The same strip with the default derivative takes 0.39 µs.

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
| `corr::TranslationEstimator`, prepared two-mask 320×97, successful refinement and regions | 2.92 ms |
| `corr::TranslationEstimator`, prepared two-mask 1280×1024, successful refinement and regions | 124 ms |
| `corr::SimilarityEstimator`, prepared two-mask f32 320×97, native refinement and regions | 6.53 ms |
| `corr::SimilarityEstimator`, prepared two-mask f32 1280×1024, nearby prediction | 242 ms |
| `metric::rig::RigPoseEstimator::estimate`, 1000 steps × 2 / 3 / 8 views | 1.02 / 1.50 / 2.43 ms |
| `metric::rig::RigPoseEstimator::estimate`, 10 000 steps × 3 views | 14.9 ms |
| `metric::rig::RigPoseEstimator::smooth`, 1000 / 10 000 steps | 93 µs / 0.93 ms |

The masked translation timings were measured 2026-10-12 with 10 Criterion samples,
1 s warmup and 2 s measurement request. Inputs are deterministic u8 analytic texture,
with a central foreground exclusion, displacement `(0.7, −0.3)`, and default settings.
The bench asserts acceptance, so these include correlation, photometry refinement,
covariance and all regional checks. Preparation, I/O and caller filtering are excluded.
The native allocation gate finds zero allocations after preparation even when masks
change. The existing displacement baselines on the same run were 1.58 ms (quadratic)
and 1.72 ms (Lucas–Kanade), consistent with the published numbers above.
There was no prior benchmark for the new operator. Its strip case leaves about 2.08 ms
of a 5 ms frame budget; this is not a claim that a complete application sustains 200 FPS.

Bead tracking on a 1280×1024 frame: an S-shaped bead 50 px wide and 1080 px long, its
prior 2 px off and bent by a further 1 px, every configured pass run:

| Operation | Time |
|---|---|
| `track`, 100 stations, 1 pass | 0.45 ms |
| `track`, 300 stations, 1 pass | 1.35 ms |
| `track`, 300 stations, 3 passes | 2.60 ms |
| `track`, 300 stations, 3 passes, reach ±30 px | 3.03 ms |
| `track`, 300 stations, 3 passes, 30% of the bead in gaps and a distractor beside it | 2.66 ms |
| `explain_bead`, 300 stations, 3 passes | 3.68 ms |

With coupled pairing (ADR-0019), on the same S-bend with a bead 6 px wide under 2 DN of
noise, a width range of 2–12 px, a reach of ±8 px and 300 stations: the prototype's state
space, on the tracking stage's 1 px profile grid and the final stage's 0.5 px one.

| Operation | Time |
|---|---|
| `track`, per-station pairing, 3 passes | 0.95 ms |
| `track`, coupled pairing, 1 pass | 1.51 ms |
| `track`, coupled pairing, 3 passes | 2.46 ms |

- **Within the tracker's budget.** The coupled call costs less than the default 3-pass
  `track` above, 2.60 ms, with one tracking pass or three.
- **The path's planes are pruned.** A one-wall plane has no state where both walls lie on
  the image, and the min-path skips it there, which is exact; that took the 3-pass call
  from 2.76 to 2.46 ms. The passes between the first and the final stage run the forward
  pass alone, since their margins are not reported.

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

`measure`'s crate-private numerical cores (ADR-0020), measured 2026-10-16 on an Apple M4
Pro (`benches/measure_numeric.rs`):

| Operation | Time |
|---|---|
| robust Whittaker smoother, 1 000 samples, 1 + 5 Huber solves | 116 µs |
| robust Whittaker smoother, 10 000 samples | 1.22 ms |
| robust Whittaker smoother, 100 000 samples | 12.1 ms |
| min-path, 300 layers × 65 × 21 cells × 2 planes, bands 8 and 2 | 2.99 ms |
| min-path, 61 rays × 113 radii, band 8, ends pinned | 25.0 µs |

- **The smoother is linear in its length.** It takes about 20 ns per sample and solve at
  every size: 119, 132 and 123 ns per sample over its six solves. The bead tracker's own
  solves are a few hundred samples long.
- **The min-path is linear in its states.** It costs about 3.7 ns per state for the
  forward pass, the backward pass and the backtrack together. Most of that is the band
  passes.
- **The bench's state space is the Python prototype's.** It has a ±8 px reach on a ¼ px
  grid, widths of 2–12 px on a ½ px grid, and present and absent planes. The tracker's
  coupled pairing runs on its stages' profile grids instead (1 px when tracking, 0.5 px
  in the final stage), and prunes the planes a layer lacks; its `track` call is in the
  [speed table](#speed).

## Linear filters

Apple M4 Pro (NEON), 1280×1024, best of 25 after 3 warm-up, single thread; libblur measured
in the same session. Ratios are libblur's time divided by ours, so above 1 means we are faster.

| case | ours | libblur | libblur / ours |
|---|---|---|---|
| box `u8`, r = 1, 2, 4, 8 (exact) | 0.63 to 1.06 ms | 0.83 to 0.91 ms | 1.3x, 0.93x, 0.79x (r = 1, 4, 8) |
| Gaussian `f32`, σ = 1 | 0.85 ms | 0.69 ms | 0.81x |
| Gaussian `f32`, σ = 2 | 1.33 ms | 1.21 ms | 0.92x |
| Gaussian `f32`, σ = 4 | 2.13 ms | 2.43 ms | 1.14x |
| Gaussian `f32`, σ = 6 | 3.52 ms | 3.89 ms | 1.10x |
| binomial `u8`, order 2, 4, 6 | 0.63, 0.87, 1.11 ms | no comparator | n/a |

x86_64 (GitHub `ubuntu-latest`, same run, single thread): the box is 1.8x to 2.3x faster
than libblur on the default target, and 0.91x to 1.13x on `x86-64-v3` (AVX2, FMA). The
Gaussian is behind libblur on x86_64: 0.47x to 0.67x on the default target, and 0.57x to
0.99x with AVX2 and FMA. The cause is not identified. Fusing the tap update gains at most
10%, and only where FMA is hardware: without it `mul_add` is a library call, 10 to 20
times slower. The shared runner moved timings by up to 40% between runs, so only ratios
within one run count. The workflow is `filter-bench.yml`.

The box is exact on every target; libblur's box differs from the rounded integer mean by
up to 1 LSB. The Gaussian is exact below σ = 8 and a three-box cascade above it, about
4.5 to 4.8 ms at every σ from 8 on the M4, so it is flat in σ. The cascade's error budget is
in ADR-0023.

## Accuracy envelopes

Masked translation, measured 2026-10-12: analytic f32 texture at nine subpixel shifts,
independent masks excluding a central rectangle, gain 1.3 and offset 17, and five
deterministic trials per cell. The noisy sweep adds independent uniform noise up to
5 grey levels to both images; values below are the worst per-axis cell bias and spread.

| Operator | Unit | Worst bias | Worst σ | Envelope bias / σ |
|---|---|---|---|---|
| masked translation, clean | px | 0.0011 | 0 (deterministic repeats) | 0.002 / 0.000001 |
| masked translation, noise 0/2/5 GL | px | 0.0871 | 0.0104 | 0.14 / 0.016 |

Additional tests cover independent validity masks, excluded non-finite pixels, static
distractors, odd dimensions, strided views, unequal image sizes, predicted motion,
forward/backward closure, repetitive patterns, search limits, refinement budgets,
noise-dependent covariance and intensity-unit invariance. An observable 0.01 rad
rotation is rejected as a translation-model mismatch. A single textured quadrant
reports that regional consistency is unavailable. Covariance is a local approximation;
the suite does not certify its statistical coverage or include systematic bias in it.

Each row sweeps a synthetic fixture with known subpixel ground truth. It reports the worst
bias and standard deviation found anywhere in the sweep. The envelope is what
`tests/accuracy.rs` enforces, at about 1.5× the measured value.

The first table's fixtures are anti-aliased edges from an analytic Gaussian-CDF profile,
with seeded uniform noise of up to 5 LSB, quantized to `u8`.

| Operator | Unit | Worst \|bias\| | Worst σ | Envelope (bias / σ) |
|---|---|---:|---:|---|
| `Edge1DDetector` position | px | 0.042 | 0.095 | 0.07 / 0.15 |
| `Edge2DDetector` position | px | 0.097 | 1.608 | 0.15 / 2.40 |
| `Caliper` (rect) position | px | 0.026 | 0.062 | 0.05 / 0.12 |
| `fit_circle` radius and centre | px | 0.325 | 1.135 | 0.50 / 1.75 |
| `ShapeMatcher` translation | px | 0.003 | 0.011 | 0.02 / 0.03 |
| `ShapeMatcher` rotation | deg | 0.000 | 0.001 | 0.05 / 0.05 |
| `ShapeMatcher` scale, 0.5–2.0× | fraction | 0.0017 | 0.0012 | 0.003 / 0.0018 |
| `ShapeMatcher` position under scale, 0.5–2.0× | px | 0.0191 | 0.0048 | 0.04 / 0.01 |
| estimate-then-verify scale, 0.5–2.0× | fraction | 0.0013 | 0.0002 | 0.003 / 0.001 |
| estimate-then-verify position | px | 0.0191 | 0.0048 | 0.04 / 0.01 |
| rectified crop repeatability | 8-bit grey levels | 0.88 | 1.69 | 1.3 / 2.5 |
| `corr::displacement`, quadratic | px | 0.0239 | 0.0164 | 0.04 / 0.025 |
| `corr::displacement`, Lucas–Kanade | px | 0.0204 | 0.0141 | 0.035 / 0.022 |

How to read the hard rows:
- **`Edge2DDetector`.** The worst cell is blur σ = 3 with 5 LSB noise at 45°. Its σ is
  noise edgels chained to the line: a quarter of that cell's edgels lie more than 1.5 px
  off it, and the rest have σ 0.69 px and no bias. Every other cell is under 0.35 px, and
  at 30° and 45° up to blur σ = 1 under 0.05 px.
- **`ShapeMatcher` scale.** The worst cell is 1.37×, where the 120° and 240° scenes read
  0.0025 high and the 0° scene reads true. Every other scale's σ is at most 0.0002.
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

### Sub-pixel phase sweep and the locking index

A locator can be unbiased at phases 0, ¼, ½ and ¾ and biased between them. These rows
(`tests/accuracy/phase.rs`, measured 2026-10-16) sweep the phase finely on the same
pixel-integrated step, 40 to 200 DN:

- 20 sub-pixel phases × edges at 0, 15, 30 and 45° × no blur, PSF σ 0.5 and σ 1.0 px;
- noise-free and under 2 DN of seeded Gaussian noise (6 trials);
- worst |bias| and σ are per phase, so a bias that changes sign between phases is not
  averaged away;
- the **locking index** is the max/min bin ratio of a 10-bin histogram of the located
  positions' fractional pixel `x`, over 200 phases × 3 trials under 2 DN. The sweep moves
  every sample's true position by exactly one pixel in equal steps, so an unbiased
  locator gives about 1. An empty bin makes it ∞.

Errors are along the true normal. The strips use CaliperBench's textbook settings (above)
except "default strip", which uses `MeasureConfig::default()` (one sample per pixel,
derivative of Gaussian σ 1 px). The rect caliper is the default `MeasureConfig`, 5 px wide.
`Edge1DDetector` (σ 1 px) runs on each image row; `Edge2DDetector` uses its default
config, and an edgel's `x` is where its tangent crosses its own pixel row.

| Locator | Worst \|bias\| (px) | Worst σ (px) | Locking index, no blur / σ 0.5 / σ 1.0 | Envelope (bias / σ / index) |
|---|---:|---:|---|---|
| strip, `GradientPeak` (parabola) | 0.101 | 0.079 | 4.26 / 1.81 / 1.35 | 0.15 / 0.12 / 5.9 |
| strip, `GradientPeak` (log-parabola) | 0.109 | 0.078 | 3.66 / 1.72 / 1.31 | 0.17 / 0.12 / 5.0 |
| strip, `MidpointCrossing` | 0.054 | 0.063 | 1.83 / 1.20 / 1.12 | 0.08 / 0.094 / 2.25 |
| strip, `HalfContrast` | 0.055 | 0.060 | 1.81 / 1.21 / 1.12 | 0.084 / 0.09 / 2.2 |
| default strip, `GradientPeak` | 0.065 | 0.064 | 1.79 / 1.48 / 1.37 | 0.097 / 0.095 / 2.2 |
| `Caliper` (rect), default | 0.070 | 0.019 | 2.10 / 1.54 / 1.35 | 0.105 / 0.029 / 2.65 |
| `Edge1DDetector`, `Parabolic3` | 0.032 | 0.057 | 1.24 / 1.50 / 1.39 | 0.048 / 0.086 / 1.75 |
| `Edge1DDetector`, `Gaussian3` | 0.043 | 0.052 | 1.69 / 1.10 / 1.07 | 0.065 / 0.078 / 2.05 |
| `Edge1DDetector`, `Centroid { radius: 2 }` | 0.181 | 0.186 | 2.88 / 111 / ∞ | 0.27 / 0.28 / ∞ |
| `Edge2DDetector`, `ParabolicAlongNormal` | 0.031 | 0.029 | 1.49 / 1.47 / 1.40 | 0.046 / 0.043 / 1.74 |

The index envelope is `1 + 1.5 (index − 1)`.

How to read them:
- **Every locator locks above 1.2 on an unblurred step**, the worst blur here. With σ 0.5
  px of optical blur only the level methods and the log-parabola come down to about 1.2.
  The worst bias is also at no blur, about 0.1 px for the textbook gradient strips.
- **At σ 1.0 px the gradient-peak profiles still lock at 1.31–1.39**: the textbook and
  default strips, the rect caliper and the 1-D parabola. The level methods (1.12) and
  the 1-D log-parabola (1.07) show what an index this close to 1 looks like at this noise.
  The continuous profile below is the remedy.
- **`Edgel`s are as close to the edge as a 1-D profile.** `ParabolicAlongNormal` fits
  the pre-suppression gradient magnitude at the edgel's pixel and its two neighbours along
  the image axis closest to the normal, the axis suppression compares along; the 1-D
  parabola's index (1.39 at σ 1.0) is its counterpart. Fitting the suppressed magnitude
  instead leaves two neighbours on a one-pixel ridge at zero, so every edgel stays on its
  pixel centre (bias 0.45 px, index ∞ at every blur); that is why the fit uses the
  pre-suppression magnitude. Suppression along the
  diagonal also kept two pixels per row of a 30° or 45° edge; along the axis it keeps one.
- **`Centroid { radius: 2 }` locks badly** once the response is wider than its window: the
  window's truncation pulls the centroid to the peak sample. The response's σ is about
  `√(σ² + σ_psf² + 1/12)` samples (1.04, 1.16 and 1.47 here). Measured at σ 1.0 px of
  blur, radius 3 locks at 7.1, radius 4 at 1.54 and radius 5 at 1.27; a radius of at
  least 3× the response's σ keeps the index under about 1.5 at every blur here.

#### Continuous profiles

`Derivative::Continuous` samples the tapered 2-D derivative of Gaussian at each profile
point (`ProfileSampler`) and refines each peak on that continuous response. The rows are
the default strip (one sample per pixel, `Strongest` rising edge) on the same sweep,
changing only the derivative and σ:

| Strip | Worst \|bias\| (px) | Worst σ (px) | Locking index, no blur / σ 0.5 / σ 1.0 | Envelope (bias / σ / index) |
|---|---:|---:|---|---|
| default derivative, σ 1 px (the row above) | 0.065 | 0.064 | 1.79 / 1.48 / 1.37 | 0.097 / 0.095 / 2.2 |
| default derivative, σ 2.5 px | 0.039 | 0.050 | 1.45 / 1.13 / 1.22 | 0.059 / 0.076 / 1.7 |
| `Continuous`, σ 1 px | 0.080 | 0.037 | 2.66 / 1.38 / 1.15 | 0.12 / 0.056 / 3.5 |
| `Continuous`, σ 1.5 px | 0.032 | 0.020 | 1.47 / 1.08 / 1.09 | 0.048 / 0.031 / 1.71 |
| `Continuous`, σ 2.5 px | 0.012 | 0.013 | 1.19 / 1.07 / 1.07 | 0.018 / 0.019 / 1.2 |

The worst |bias| per σ and blur, over angles, phases and noise:

| Strip | No blur | PSF σ 0.5 px | PSF σ 1.0 px |
|---|---:|---:|---:|
| default derivative, σ 1 px | 0.065 | 0.064 | 0.046 |
| default derivative, σ 2.5 px | 0.039 | 0.028 | 0.022 |
| `Continuous`, σ 1 px | 0.080 | 0.029 | 0.014 |
| `Continuous`, σ 1.5 px | 0.032 | 0.010 | 0.008 |
| `Continuous`, σ 2.5 px | 0.012 | 0.007 | 0.006 |

How to read them:
- **At σ 2.5 px the continuous profile stays at or under 1.2 over the whole sweep,**
  unblurred steps included, and its worst bias is 0.012 px. Its index row is pinned at
  1.2 rather than 1.5× the measured value. From PSF σ 0.5 px, σ 1.5 px is enough (1.08).
- **A wider σ does not fix the bilinear profile.** The default derivative at σ 2.5 px still
  locks at 1.45 unblurred and 1.22 at PSF σ 1.0 px, with 3× the continuous profile's
  bias: the interpolation's knots, not the smoothing, cause it.
- **The unblurred step is the hard case for any linear locator.** A pixel-integrated step
  with no optical blur aliases into the kernel's passband, which gives a bias odd in the
  phase that falls with σ (0.080, 0.032 and 0.012 px at σ 1, 1.5 and 2.5 px) and with blur.
  At σ 1 px the continuous profile is unbiased to 0.014 px with PSF σ 1.0 px and locks
  less than the default there (1.15 against 1.37), but locks more on the unblurred step.
- **Its spread is lower** at every σ: the 2-D kernel also averages across the scan.

The same check on the whole bead tracker (`tests/accuracy/bead_phase.rs`): a straight light
bar 16 px wide with its normal along +x, the angle at which the strips lock most; the
three blurs; 2 DN of noise; a prior 2 px off; 200 phases. The rows read both final-stage
edges of every station:

| `BeadTracker` | Worst \|bias\| (px) | Locking index, no blur / σ 0.5 / σ 1.0 | Envelope (bias / index) |
|---|---:|---|---|
| default config | 0.045 | 1.76 / 1.16 / 1.13 | 0.067 / 2.15 |
| both stages `Continuous`, σ 2.5 px | 0.015 | 1.17 / 1.07 / 1.06 | 0.022 / 1.2 |

#### The diagnostic on real data

`tools/registration_eval/phase_histogram.py` computes the same histogram over a
`sequence.json` manifest, without ground truth. It takes `detect_edges` edgels whose normal
is within 25° of an axis and whose pixel lies in the eroded mask. A flat histogram is
necessary, not sufficient, for an unbiased locator. `--refine dog` re-locates the same
edgels with a continuous derivative of Gaussian (σ 1 px, tapered to zero at 4σ) as a
reference that does not lock. `--refine sampler` does it with the library's
`ProfileSampler`, which also honours the mask.

On private development sequences A–D, with the visibility mask, eroded by 5 px:

| Sequence | Edgels | Detector index | Detector normalized range | DoG reference index | Reference range |
|---|---:|---:|---|---:|---|
| A | 100,897 | 1.31 | 0.84–1.10 | 1.06 | 0.97–1.02 |
| B | 73,132 | 1.45 | 0.80–1.15 | 1.07 | 0.97–1.03 |
| C | 14,213 | 1.41 | 0.79–1.12 | 1.07 | 0.96–1.03 |
| D | 9,331 | 1.35 | 0.85–1.15 | 1.18 | 0.92–1.09 |
| all | 197,573 | 1.32 | 0.85–1.12 | 1.06 | 0.96–1.03 |

The normalized range is the lowest and highest bin as a multiple of the uniform rate.
With the current fit. The earlier suppressed-magnitude fit, on 214,678 edgels (more, since
suppression kept two pixels per row of an oblique edge), gave an index of 81–316 per sequence
and 140 pooled, with bins at up to 5× the uniform rate: its edgels sat on pixel centres. With
the registration-validity mask instead (76,193 edgels, the tracked object excluded), the
pooled detector index is 1.33, against 69 before the fix, and the reference's 1.08. The
detector is now about as flat on real data as on the synthetic sweep; the derivative of
Gaussian is flatter still.

With `--refine sampler` at σ 1 px, on the same edgels, the indices are A 1.06, B 1.07,
C 1.07, D 1.18 and 1.06 pooled, the reference's to within 0.01: the library's sampler
reproduces it, and no edgel's support reaches the eroded mask's edge. At σ 2.5 px the
tool is not a fair test of the wider kernel. It re-locates each edgel within ±1.1 px of
the detector's fine-scale position, and the wider kernel's peak can lie beyond that. On
the 165,837 edgels whose support stays in the mask, it gives 1.06 pooled.

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

With coupled pairing the same sweep gives the same rows to the third decimal (0.0113 /
0.0868, 0.0213 / 0.1696, 0.0120 / 0.0487): the walls snap to the same caliper edges.
`tests/accuracy/bead_coupled.rs` adds the hard cases on a light bead 8 px wide at 60 DN
contrast under 2 DN, along a line, an arc and a sine, from a prior 2 px off. Each row is
both walls' signed error over every final hit, worst over the curves:

| Case | Worst \|bias\| | Worst σ | Envelope (bias / σ) |
|---|---:|---:|---|
| clean | 0.0025 | 0.0578 | 0.004 / 0.087 |
| clean, both stages on `Derivative::Continuous` (σ 1 px) | 0.0041 | 0.0674 | 0.0062 / 0.10 |
| width from 6 to 12 px along the curve | 0.0055 | 0.0608 | 0.0083 / 0.091 |
| a 3 px stripe 2 px outside the `+n` wall (that wall only) | 0.4561 | 0.0513 | 0.68 / 0.077 |
| no bead over 30 px (no hit inside the gap) | 0.0028 | 0.0579 | 0.0042 / 0.087 |
| a 1.5 px highlight inside the bead, off centre | 0.1722 | 0.2142 | 0.26 / 0.32 |

- **The snap uses the default derivative.** At this blur the continuous profile's walls
  are noisier, so coupled pairing refines on the stage's own caliper edges and leaves
  `Derivative::Continuous` a stage setting.
- **A merge in the derivative stays a bias.** The stripe 2 px outside a wall pulls it
  0.46 px inwards, and the highlight, whose edge lies 2 px inside the `+n` wall, biases
  the walls by −0.17 px on average with a σ of 0.21 px: the path chooses between edges and
  cannot separate them.
- **A parallel bead flags itself.** From a prior midway between two identical beads 14 px
  apart, the first pass's margin is 0.02, against 9.1 for one bead alone
  (`tests/bead.rs`).
- **Occluded profiles.** On the synthetic profiles with an occluder, with the given occlusion and the
  `+n` side off the image from station 45, the path chooses absence where every sample is
  occluded and keeps the `−n` wall alone in the one-sided stretch; the walls are within
  0.003 px of the truth away from the three stations next to each end and break, when a
  clearance of 4 px keeps the internal reversal's edge from standing in for the wall.
  Without it the reversal, as strong as the wall, pairs with the visible wall along the
  whole curve.

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

## Real data: bead tracking on DamSegment cracks

A crack in concrete is a dark line a few pixels wide, with rough walls, in a textured
surface. Cracks test whether the tracker locks onto a real curvilinear structure from a
perturbed prior and follows it. The data is DamSegment, by V. Gharehbaghi, C. R. Bennett,
R. Lequesne, H. Zhao and J. Li: Mendeley Data,
[doi:10.17632/z5z6gtt5t4.1](https://doi.org/10.17632/z5z6gtt5t4.1), licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It holds 1500 photographs of
concrete, 640×640, in Easy, Medium and Hard sets of 500, with hand-drawn crack masks.

**The reference.** Each crack mask is skeletonised and split at its junctions into
non-branching paths of at least 80 px: 8680 paths (995, 2793 and 4892 per set). The
reference width is the mask's, `2·EDT − 1`. The masks are rasterised polygons, about
twice as wide as the dark line, and their centreline lies a pixel or two from it. The
reference is pixel-level, so these numbers measure robustness, not subpixel accuracy:
the synthetic rows above measure that.

**The runs.** Each path is tracked from itself and from 22 perturbations of it, 199,640
calls in all:
- translations of 1 to 12 px;
- rotations that move its ends by 2 to 12 px;
- a sine of wavelength `L/2` with an amplitude of 1 to 6 px;
- a Gaussian bump, σ 10 px, 2 to 12 px high;
- Douglas–Peucker polygons of 10 and 6 vertices;
- a quarter of its length cut off one end.

The image is BT.601 luma. The config is the default, except:
- `polarity` dark;
- a width range of `max(2, w/4)` to `2w` for a mask width `w`;
- a reach (`track.max_offset`) of 8 px and a `spacing` of 2 px;
- a `threshold` of 3;
- no obliquity gate in the final stage.

**What counts.** A station is *on the crack* within half the mask's width of its
centreline, plus 1 px. A call is *locked* when 90% of its stations are on the crack. It is
*converged* when 90% lie within 1 px of the curve tracked from the unperturbed path, a
test that needs no annotation.

From the 18 perturbations within the reach (displacements up to 8 px, the polygons and the
truncation):

| Set | Paths | Centre from the mask's centreline, median / p95 | Support | Width, measured / mask | Locked from the path itself | Locked | Converged | Time per call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Easy | 995 | 1.59 / 5.69 px | 91% | 6.3 / 10.9 px | 91% | 87% | 62% | 0.45 ms |
| Medium | 2793 | 1.74 / 5.02 px | 90% | 4.8 / 9.5 px | 92% | 88% | 74% | 0.34 ms |
| Hard | 4892 | 1.33 / 4.09 px | 86% | 3.9 / 6.6 px | 89% | 83% | 82% | 0.23 ms |
| all | 8680 | 1.51 / 4.78 px | 88% | 4.4 / 8.2 px | 90% | 85% | 77% | 0.27 ms |

The basin, over all three sets. The last column counts the calls that did not lock but
still report a support of at least 0.5:

| Perturbation of the prior | Prior on the crack | Locked | Converged | Not locked, support ≥ 0.5 |
|---|---:|---:|---:|---:|
| translation, 2 px | 99% | 89% | 82% | 10% |
| translation, 4 px | 45% | 87% | 69% | 12% |
| translation, 6 px | 7% | 78% | 54% | 20% |
| translation, 8 px (the reach) | 1% | 51% | 29% | 42% |
| translation, 10 px | 0% | 21% | 10% | 55% |
| translation, 12 px | 0% | 6% | 2% | 51% |
| rotation, ends moved 8 px | 6% | 72% | 48% | 26% |
| rotation, ends moved 12 px | 1% | 28% | 12% | 67% |
| sine, amplitude 6 px | 14% | 83% | 57% | 16% |
| bump, 8 px high | 31% | 85% | 80% | 14% |
| bump, 12 px high | 15% | 62% | 50% | 35% |
| polygon of 6 vertices | 97% | 90% | 88% | 9% |

Against scikit-image's `active_contour`, on 120 of the paths (40 per set) from the same
priors within the reach. The snake is open, with free ends, `w_line = −1` and
scikit-image's other defaults, on the luma smoothed by a Gaussian of σ 3 px:

| Method | Centre, median / p95 | On the crack | Locked | Converged | Time per call |
|---|---:|---:|---:|---:|---:|
| `BeadTracker` | 1.37 / 4.78 px | 96% | 86% | 76% | 0.31 ms |
| `active_contour` | 1.56 / 13.18 px | 82% | 64% | 92% | 523 ms |

How to read them:
- **The reference is good to about 1.5 px.** From the path itself, the tracked curve sits
  a median 1.46 px from the mask's centreline. That includes an offset of 0.6 px in +y and
  none in x, roughly uniform over the image. On a synthetic line the same config reads
  the centre exactly, so the offset is the annotation's registration or the cracks'
  lighting, not the tracker. The 10% of paths that do not lock even from themselves are
  mostly narrow masks: in the ones inspected, the tracker sits on the dark line and the
  mask's centreline runs beside it. The masks are about twice as wide as the dark line
  the tracker measures, so the width column is a sanity check, not an error.
- **The tracker locks to about its reach, but where exactly it ends depends on the prior.**
  - Locked stays at 87% from a 4 px translation and 78% from 6 px, and falls past the
    8 px reach.
  - Converged falls faster. From 4 px off, 69% of calls end within 1 px of the curve
    tracked from the path itself.
  - A strip on rough concrete can hold more than one dark pair, and the score favours the
    pair nearer the station, so the prior decides between them.
  - Three passes almost never bring every station's correction under `tol` on a rough
    crack: 99.9% of calls stop on `PassLimit`. On a subset of 387 paths, 6 or 10 passes
    raise converged by 6 to 12 points and lower locked by 2 to 3.
  - The corrections shrink from pass to pass, then level off. On 300 paths with 10
    passes, the median rms correction falls from 0.25 px at the third pass to 0.09 px at
    the tenth, about a tenth of the stations' scatter about the solved curve, and the
    median largest correction to 0.33 px. Only 13% of those calls end `Converged`.
- **Failure is mostly silent.** 91% of the calls that do not lock still report a support
  of at least 0.5: a median of 0.84, against 0.94 for a call that locked. Neither
  support, `center_rms` nor `longest_gap` tells the two apart well.
  - Nor do the final stage's `confidence` or a `min_margin`. Over 5400 calls on 300 paths,
    the chance that a locked call scores better than one that did not (the AUC) is 0.71
    for support, 0.68 for the longest gap, 0.59 for `center_rms` and 0.48 for the
    median `confidence`. With a `min_margin` of 0.1 to 0.3, it is 0.66 to 0.68 for
    support, and one more call in a hundred locks.
  - A support threshold that keeps 90% of the locked calls still passes 65% of those
    that did not lock.
  - Beyond the reach, the tracker takes the next dark structure for the crack: a pit, a
    shadow, a parallel crack. It measures that structure as it would the bead.
  - The synthetic fixtures have nothing beyond the reach to lock onto. On a textured
    surface, keep the prior within the reach, or check the result some other way.
- **The library's defaults for the final stage suit clean edges, not rough ones.** With
  the default threshold of 5 and the final stage's obliquity gate of 30°, support falls
  from 88% to 55%, while locked and converged do not change: the final stage only
  measures. Most of the lost
  stations are `NoPair`, where one edge of the pair fails the obliquity gate: a rough
  crack wall's gradient direction is not a reliable test. Most of the rest are
  `Caliper(NoEdge)`, below the threshold.
- **`active_contour`** converges more consistently, 92%, and from further: 45% of its calls
  lock from a 10 px translation, against 21%.
  - Its free ends slide along the crack and past it, in some calls off the image, so only
    64% of its calls stay on the crack.
  - It takes 523 ms per call, median: up to 2500 iterations on one core.
- **Speed.** A call takes 0.27 ms, median, and 0.89 ms at p95, through the Python bindings.
  That is 4.2 µs per station, for a median of 64 stations, each measured in 3 tracking
  passes and the final stage.

The dataset is not distributed with this repository.
[`tools/bead_eval/README.md`](../tools/bead_eval/README.md) downloads it and reproduces the
numbers.

## Finding a bead without a prior, and choosing both walls

Two offline evaluations on DamSegment cracks. Methods and full tables are in
[`tools/bead_eval/README.md`](../tools/bead_eval/README.md#results).

- **Acquisition:** a detected path seeds the tracker about as well as the truth does, so a
  ridge detector's sub-pixel output adds nothing. The open problem is choosing the bead
  among other lines.
- **Coupled pairing** (ADR-0019) locks
  more often and accepts fewer wrong locks at equal coverage. Its margin does not separate
  the wrong locks that remain (AUC about 0.5).

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

## Planar rig substrate pose

`metric::rig::RigPoseEstimator`, measured 2026-10-16. The speed rows above use the
`rig` bench: one view 3 px wrong every tenth step, so the biweight runs from both starts
there. A step costs about 0.5 µs per view.

### Synthetic accuracy

The fixture (`tests/rig/fixture.rs`) has three affine views 120° apart around the plane
origin, 30 mm out, at 20 px/mm with their own shear and aspect, and a known per-step
increment: 0.4 mm along x with a slow sine across it and 0.011° of rotation per step on
average. Shifts get 0.05 px of Gaussian noise per axis, and the estimator runs with its
defaults (shift floor 0.05 px). Per-step increment error over 300 steps and eight seeds:

| Component | Bias | σ |
|---|---:|---:|
| tx | −0.000005 mm | 0.0015 mm |
| ty | 0.00002 mm | 0.0015 mm |
| θ | −0.00002° | 0.0028° |

The accuracy suite pins the worst seed: translation |bias| 0.0002 mm and σ 0.0016 mm
(envelope 0.0004 / 0.0025), rotation |bias| 0.0003° and σ 0.0029° (envelope 0.0005 /
0.0045). The cases `tests/rig_pose.rs` checks:

- **Noise-free** shifts give every increment to 2·10⁻⁵ mm and 2·10⁻⁶ rad, and the
  300-step pose to 2·10⁻³ mm.
- **The final pose** after 300 noisy steps lies within 4σ of its reported covariance in
  every seed.
- **A bridged step** carries the previous increment with a larger covariance; after three
  bridged steps in 60 the pose is within 0.05 mm.
- **A gross view** 3 px (~60σ) wrong is rejected by the biweight, leaving the step within
  0.004 mm and 10⁻⁴ rad. Huber alone gives it weight below 0.1 but leaves at least twice
  that error.
- **A single-view step** holds the previous rotation increment (or zero), and its
  translation is off by the rotation error times the lever to the origin.
- **The smoother** brings the increment RMS below 0.6 of the raw one at 0.2 px of noise,
  fills a three-step gap to 0.02 mm and gates a consistent 0.5 mm spike.
- **A 1% plane-height error** leaves the views agreeing to 10⁻³ mm while the path-length
  ratio reads 0.990.

### Synthetic sequence benchmark

`tools/sequence_bench` presets `rig-bridged`, `rig-gross` and `rig-single-view`, seed 86,
measured 2026-10-16 on `main` at 2d3294b with this operator, by
`tools/sequence_bench/rig_eval.py`. The settings were frozen before the run:
`TranslationEstimator` defaults except `max_region_error=None` (the substrate rotates
0.48° per frame on the arc, so one view's regions disagree by design), the shift lifted
at the centroid of the previous frame's registration mask, and `RigPoseEstimator`
defaults anchored at the given frame-0 pose. Errors are per-step increment errors over
the 39 steps against `truth.poses`.

| Preset | tx bias / σ (su) | ty bias / σ (su) | θ bias / σ (°) | Provenance | Path ratio |
|---|---|---|---|---|---:|
| rig-bridged | 0.033 / 0.018 | −0.008 / 0.003 | −0.0007 / 0.0083 | bridged at frames 20, 21; all others measured | 0.987 |
| rig-gross | 0.035 / 0.021 | 0.018 / 0.016 | 0.0016 / 0.0082 | the moving camera rejected on all ten steps it moves (times 14→15 to 23→24), the other two weighted ≥ 0.96 | 0.988 |
| rig-single-view | 0.034 / 0.018 | −0.003 / 0.020 | −0.0007 / 0.0083 | single-view at frames 20, 21, with view 0 | 0.988 |

The translation bias is the registration's, not the lift's: against the exact image motion
of each lift point, the accepted shifts err by 0.002 ± 0.037 px in x and 0.052 ± 0.034 px
in y (111 shifts, rig-bridged), and the 1.2% path shortfall follows from it. Under that
error, the steps' residual scale has median 1.00 at the default 0.05 px shift floor; the
registration's own covariances (σ 0.003–0.005 px) would put every view 10–50σ off.

## Real data: masked translation

Evaluated 2026-10-12 on all adjacent pairs of six private 320×97 camera-strip sequences:
development sequences A–D, which select settings, and held-out sequences E and F, which use
the identical frozen configuration. Each frame belongs to one camera group, and a group
change is a handover, counted as an explicit failure. Caller masks exclude stationary
foreground, unlit regions, the tracked bead's support and registered blank-difference
support; foreground is dilated by 3 px. Caller preprocessing uses masked normalized DoG at
sigma 1/4 px; registration uses the native estimator defaults. Source images and
annotations are not distributed.

| Split / run | Accepted / pairs | Coverage | Closure median / p95 (px) | Python estimate median / p95 (ms) |
|---|---|---|---|---|
| development A | 284 / 484 | 58.7% | 0.0096 / 0.0398 | 4.11 / 6.29 |
| development B | 327 / 546 | 59.9% | 0.0343 / 0.3112 | 3.45 / 6.06 |
| development C | 70 / 139 | 50.4% | 0.0301 / 0.0753 | 3.67 / 4.71 |
| development D | 47 / 259 | 18.1% | 0.0216 / 0.1383 | 2.38 / 3.22 |
| held out E | 104 / 197 | 52.8% | 0.0134 / 0.0331 | 4.52 / 5.91 |
| held out F | 16 / 237 | 6.8% | 0.0915 / 0.2079 | 2.12 / 5.91 |

Coverage counts every pair, including camera handovers. Closure statistics include only
pairs accepted in both directions. Python estimate times include image/mask conversion
but exclude I/O, mask preparation and filtering; those preparation costs are separately
reported by the evaluator. Their p95 can exceed the 5 ms online budget in the roadmap.

The failure counts (model mismatch / ambiguity / refinement budget / camera handover)
are 103/60/37/0, 147/5/66/1, 67/2/0/0, 204/3/5/0, 89/0/2/2 and 126/93/1/1 in table
order. Low coverage is a material limitation: translation alone does not cover these
runs, particularly turns and the unlike held-out part. Even accepted estimates have
nonzero closure, and local covariance does not account for accumulated drift. Similarity
registration and anchored sequence consistency need separate validation.

Median distances to another estimator's saved raw motion estimates are 0.024, 0.079,
0.051, 0.070, 0.030 and 0.113 px. Those saved estimates are generated comparisons,
not independent motion ground truth. Blank differences give foreground support;
precise bead boundaries still come from original bead images.

In the local propagation demonstration, a saved generated ribbon at frame 150 of sequence A
is transported independently in both directions. It produces 67 backward and 55 forward
accepted measurements before explicit motion-model rejection. Transported independent
walls, centerline and sections remain separate from caliper observations and the tracker's
regularized curve. Unsupported sections have typed rejection results; observations
never feed back into the immutable seed or registration. Overlays show transported walls
and local wall measurements. No image or annotation artifacts are committed.

## Local masked similarity and anchored sequence experiments

Known-transform accuracy and the private-sequence results are in
[`tools/registration_eval/README.md`](../tools/registration_eval/README.md#results). The
similarity estimator stays behind `corr-experimental` until its regional check is
recalibrated (roadmap, registration calibration).
