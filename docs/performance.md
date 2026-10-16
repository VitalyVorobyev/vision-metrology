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

`measure`'s crate-private numerical cores (ADR-0020), measured 2026-10-16 on an Apple M4
Pro (`benches/measure_numeric.rs`):

| Operation | Time |
|---|---|
| robust Whittaker smoother, 1 000 samples, 1 + 5 Huber solves | 119 µs |
| robust Whittaker smoother, 10 000 samples | 1.32 ms |
| robust Whittaker smoother, 100 000 samples | 12.3 ms |
| min-path, 300 layers × 65 × 21 cells × 2 planes, bands 8 and 2 | 3.31 ms |
| min-path, 61 rays × 113 radii, band 8, ends pinned | 24.7 µs |

- **The smoother is linear in its length.** It takes about 20 ns per sample and solve at
  every size: 119, 132 and 123 ns per sample over its six solves. The bead tracker's own
  solves are a few hundred samples long.
- **The min-path is linear in its states.** It costs about 4 ns per state for the forward
  pass, the backward pass and the backtrack together. Most of that is the band passes.
- **The coupled state space is the prototype's.** It has a ±8 px reach on a ¼ px grid,
  widths of 2–12 px on a ½ px grid, and present and absent planes. At 300 stations
  it costs more than the tracker's whole 3-pass `track` call. ADR-0019 requires a bench
  that shows the tracker's budget holds, so a smaller reach, a coarser grid or fewer
  stations are its levers.

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
| `Edge2DDetector`, `ParabolicAlongNormal` | 0.450 | 0.500 | ∞ / ∞ / ∞ | 0.68 / 0.75 / ∞ |

The index envelope is `1 + 1.5 (index − 1)`.

How to read them:
- **Every locator locks above 1.2 on an unblurred step**, the worst blur here. With σ 0.5
  px of optical blur only the level methods and the log-parabola come down to about 1.2.
  The worst bias is also at no blur, about 0.1 px for the textbook gradient strips.
- **At σ 1.0 px the gradient-peak profiles still lock at 1.31–1.39**: the textbook and
  default strips, the rect caliper and the 1-D parabola. The level methods (1.12) and
  the 1-D log-parabola (1.07) show what an index this close to 1 looks like at this noise.
  The continuous profile below is the remedy.
- **`Edge2DDetector` does not refine at all.** `ParabolicAlongNormal` fits its parabola to
  the non-maximum-suppressed magnitude. Along the normal of a one-pixel ridge, both
  neighbours are suppressed to zero, so the offset is zero and the edgel stays on its pixel
  centre: bias up to ½ px, the same with refinement off, at every blur. The older
  `edge2d_position` row above missed it because its edge sits on pixel centres at 0°.
- **`Centroid { radius: 2 }` locks badly** once the response is wider than its window: the
  window's truncation pulls the centroid to the peak sample.

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
| A | 108,685 | 316 | 0.015–4.82 | 1.10 | 0.97–1.06 |
| B | 79,125 | 81 | 0.059–4.78 | 1.06 | 0.97–1.02 |
| C | 16,598 | 122 | 0.034–4.12 | 1.18 | 0.92–1.08 |
| D | 10,270 | 254 | 0.020–5.19 | 1.20 | 0.91–1.10 |
| all | 214,678 | 140 | 0.034–4.77 | 1.08 | 0.97–1.05 |

The normalized range is the lowest and highest bin as a multiple of the uniform rate.
With the registration-validity mask instead (84,369 edgels, the tracked object excluded),
the pooled detector index is 69 and the reference's 1.06. The real data confirm the
synthetic finding: the 2-D detector's edgels sit on pixel centres, at up to 5× the uniform
rate, while a continuous derivative of Gaussian on the same edges is flat to within 10%.

With `--refine sampler` at σ 1 px, on the same edgels, the indices are A 1.10, B 1.06,
C 1.18, D 1.21 and 1.08 pooled, the reference's to within 0.01: the library's sampler
reproduces it, and no edgel's support reaches the eroded mask's edge. At σ 2.5 px the tool
is not a fair test. It re-locates each edgel within ±1.1 px of the detector's fine-scale
position, and for about a quarter of them the wider kernel's peak lies beyond that, so
they keep the detector's locking.

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

## Finding a bead without a prior

The tracker needs a prior within its reach. For a first frame with no reference part, CAD
or robot path or taught frame, something else has to find the bead. These runs measure
how well a ridge detector does that, and whether what it finds is a good prior. They are
an offline evaluation with Python packages, not a library feature.

**The detectors,** each given the bead's expected width:
- scikit-image's `sato` and `meijering` ridge filters, at scales of `w/(2√3)` and `w/2`.
  The response is thresholded with hysteresis, relative to its median and spread, then
  skeletonised and split at junctions into non-branching paths. Their width is the
  thresholded mask's;
- [`ridge-detector`](https://pypi.org/project/ridge-detector/), a Python implementation
  after C. Steger's detector ("An Unbiased Detector of Curvilinear Structures", *IEEE
  PAMI* 20(2), 1998), with sub-pixel points and widths. On a bead wider than 8 px it runs
  on a mean-pooled image, because its contrast thresholds round to 0 at large scales.

**Synthetic beads,** 1280×1024 with exact truth: a light ribbon of 60 DN contrast, blurred
by σ 1 px, along a line, an arc (R 300 px) or a sine; 3, 8 or 30 px wide. Each is
rendered clean (2 DN of noise), noisy (8 DN), beside a 40 DN step edge 10 px beyond the
bead, or broken by a 30 px gap, with two noise seeds: 72 images. *Found* means at least
80% of the bead within `max(2 px, w/2)` of one path; *precision* is the share of found
length that lies on the bead.

| Detector | Found, without a gap | Precision: clean / noisy / beside the step | Centre, median / p95 | Width read at 3 / 8 / 30 px | Time per image at 3 / 8 / 30 px |
|---|---:|---:|---:|---:|---:|
| `sato` | 100% | 100% / 100% / 62% | 0.12 / 1.99 px | 2.6 / 7.1 / 29.6 px | 0.39 / 0.21 / 0.57 s |
| `meijering` | 100% | 100% / 100% / 39% | 0.12 / 1.40 px | 2.6 / 7.1 / 29.3 px | 0.41 / 0.23 / 0.59 s |
| `ridge-detector` | 100% | 100% / 7% / 100% | 0.03 / 0.54 px | 4.1 / 8.4 / 31.4 px | 1.3 / 1.3 / 0.09 s |

Each found path, cut to where it lies on the bead, is then the prior of a `track` call
with the default config and a width range of `0.5w` to `1.5w`. *Locked* means 90% of the
stations within `w/2 + 1 px` of the true centreline, and *converged* 90% within 1 px of the
curve tracked from the true centreline:

| Prior | Calls | Refined centreline, median / p95 from the truth | Locked | Converged |
|---|---:|---:|---:|---:|
| found by `sato` | 60 | 0.033 / 0.147 px | 100% | 100% |
| found by `meijering` | 54 | 0.037 / 0.139 px | 100% | 100% |
| found by `ridge-detector` | 54 | 0.022 / 0.148 px | 100% | 100% |
| the true centreline | 60 | 0.018 / 0.111 px | 100% | – |
| the true centreline, moved 4 px | 60 | 0.018 / 0.108 px | 100% | 100% |

**Real cracks.** The same detectors on the DamSegment images and reference paths of the
section above (Gharehbaghi et al., [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)),
expecting dark lines 3, 5 and 8 px wide. *Precision* is the share of found length within
1 px of the crack mask, so an unannotated dark line, such as a formwork seam, counts
against it. Each found piece is tracked with the config of the section above:

| Detector, setting | Cracks touched | Found in one piece | Precision | Paths per image | Time per image | Tracked from a found piece: locked / converged |
|---|---:|---:|---:|---:|---:|---:|
| `sato`, threshold 3 | 38% | 14% | 14% | 113 | 0.29 s | 99% / 91% |
| `sato`, threshold 8 | 20% | 8% | 35% | 26 | 0.22 s | – |
| `meijering`, threshold 3 | 42% | 15% | 20% | 85 | 0.28 s | 100% / 89% |
| `ridge-detector`, contrast 20/40 | 81% | 24% | 5% | 508 | 1.8 s | 99% / 83% |

*Touched* asks for 80% of a crack's reference path within `max(2 px, w/2)` of any found
path, *found in one piece* of a single path. The threshold is in robust deviations of the
filter's response above its median. From the reference path itself, the same cracks lock
in 96–97% of the calls, and from a 4 px translation of it, 95% lock and 82–88% converge.

How to read them:
- **On a clean part every detector finds the bead, and what it finds is as good a prior as
  the truth.** The tracker ends on the curve it reaches from the true centreline.
  - A skeleton path, a tenth of a pixel off the centre at the median and up to 2 px at
    the 95th percentile, is enough: the tracker measures the centre and the width
    itself. Steger's sub-pixel centre adds nothing the tracker needs.
  - From a skeleton path, the refined curve keeps a little of its pixel-scale wiggle
    after three passes: 0.033 px against 0.018 px from the truth
    ([limitations](bead.md#limitations)).
- **Each detector also reports what is not the bead.**
  - `meijering` at every width, and `sato` mostly at 30 px, follow the step edge.
  - `ridge-detector` passes noise ridges at its default contrast on the noisy 3 px bead,
    about 1100 paths per image; its strictest setting removes them.
  - A gap breaks the bead into two paths, except under `sato` at 30 px, whose 15 px scale
    bridges it.
- **On cracks, the choice is the problem.** `ridge-detector` touches 81% of the cracks,
  but among about 500 paths per image, and only a quarter of the cracks come out in one
  piece. The filters find fewer cracks and fewer other lines. No setting gives both.
- **A found piece seeds the tracker as well as the annotation does.** From a found piece,
  99–100% of the calls lock, and 83–91% end on the curve tracked from the reference path,
  against 82–88% from a 4 px translation of it. The piece is cut to the crack with the
  reference's help, so these rows measure how good a found path is as a prior, not how
  to choose it.
- **The detectors filter the whole image at the bead's scale.** In scikit-image, the
  filter alone takes 0.15–0.53 s per 1280×1024 image and 0.18–0.20 s per 640×640 crack
  image, on one core, against 0.27 ms for a `track` call on a crack. Times are medians,
  measured one image at a time.

The dataset is not distributed with this repository.
[`tools/bead_eval/README.md`](../tools/bead_eval/README.md#acquisition-without-a-prior)
reproduces these numbers.

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
reported by the evaluator. Their p95 can exceed the eventual 5 ms online budget.

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

Measured 2026-10-12 on Apple M4 Pro. The native successful-path Criterion benchmarks
use deterministic f32 texture, an excluded central rectangle, 0.01 rad rotation and
1.005 scale, with 0.7/-0.3 px centre motion. The 320×97 case starts at identity; the
1280×1024 case needs an explicit nearby prediction (0.008 rad / 1.004 scale) to stay
inside the translation initializer's capture range. An unpredicted larger case correctly
failed ambiguity, so its timing is not presented as successful registration.
There are ten samples, one second of warmup and two seconds requested measurement.
Prepared native times exclude I/O/preprocessing/Python conversion; allocation tests
include an empty first mask followed by all-valid evidence and count zero allocations.

6.53 ms exceeds the eventual 5 ms online budget by 1.53 ms before acquisition or tracking.
The existing translation benchmark is 2.92 ms. Similarity's prediction warp, six-parameter
solve, rank checks and independently refined regional maps add work; the two fixtures
also differ in pixel type and motion, so this is not a controlled speedup comparison.
The larger 242 ms result is an offline-window scale, not an online budget claim.

### Known-transform accuracy

161×97 analytic f32 texture, an excluded central rectangle, three rotations
(-0.02/0.01/0.03 rad), two scales (0.985/1.02), gain 1.3, offset 17, five seeded
trials and independent uniform 0/2/5 GL noise in both images. Corner rows report the
worst x/y component bias/spread over all four corners. Other rows use radians/log-scale.

| Quantity | Worst bias | Worst σ | Pinned bias / σ envelope |
|---|---|---|---|
| Clean corner coordinate, px | 0.00329 | 0 (deterministic repeats) | 0.005 / 0.000001 |
| Noisy corner coordinate, px | 0.02164 | 0.02195 | 0.035 / 0.035 |
| Angle, rad | 0.0001113 | 0.0001408 | 0.00018 / 0.00022 |
| Log-scale | 0.0002021 | 0.0001640 | 0.00032 / 0.00026 |

Unit fixtures cover reciprocal composition, photometry, masks/invalid pixels,
strided and odd views, predicted large motion, angle search limits, native motion rank
loss, a single supported quadrant with unavailable consistency, and incompatible regional
motion rejected as model mismatch. A known 64-frame noisy sequence composes native estimates
and constrains against an independently observed first keyframe every eight steps. Anchors
reduce mean error against truth and keep corner error below 0.1 px; unsupported constraints
leave the pose unchanged. Covariance has no certified statistical coverage.

### Private sequence results

The following compares the two operators using identical prepared inputs. Sequences A–F
keep their development/held-out split, camera groups and masks. Similarity defaults are
identical in all runs. The raw multi-strip sequence G has 141 frames, with each of its
three strips evaluated independently using the explicit caller-selected valid texture ROI
x/y/w/h = 5/5/100/85. It has no camera groups, foreground mask, blank pair or reference
curve; there is no invented bead seed.

Native acceptance is distinct from caller-owned sequence support. Sequence steps require
accepted reverse registration and at most 0.25 px maximum corner closure; the gate was
added after observing development-run closure failures. Anchors obey the same reciprocal
policy and are attempted every eight supported steps within a segment. Camera handover,
registration or reciprocity loss breaks composition. Failed constraints stay unavailable.
The closure column reports all native estimates accepted in both directions, including
those withheld by the sequence gate. Small closure sample counts are explicit.

| Run | Translation accepted | Similarity accepted / pairs | Reciprocal sequence steps | Closure samples; median / p95 px | Python estimate median / p95 ms |
|---|---|---|---|---|---|
| G | 289 | 147 / 420 (35.0%) | 118 | 124; 0.069 / 0.237 | 2.47 / 4.01 |
| A | 284 | 53 / 484 (11.0%) | 20 | 22; 0.099 / 0.268 | 8.31 / 13.86 |
| B | 327 | 68 / 546 (12.5%) | 31 | 32; 0.077 / 0.162 | 7.43 / 10.69 |
| C | 70 | 0 / 139 (0.0%) | 0 | 0; unavailable / unavailable | 7.74 / 10.67 |
| D | 47 | 10 / 259 (3.9%) | 2 | 3; 0.201 / 1.401 | 4.86 / 7.63 |
| E | 104 | 9 / 197 (4.6%) | 2 | 2; 0.120 / 0.183 | 10.25 / 13.31 |
| F | 16 | 6 / 237 (2.5%) | 1 | 1; 0.157 / 0.157 | 3.57 / 14.28 |

Native similarity failure counts below retain every reason. Most failures are regional
map disagreement; extra parameters and full-corner extrapolation are substantially less
stable on these strips than a translation-only shift. G's small texture ROI also
loses rank in two regions. The accepted sample in D still has a 1.401 px p95 reciprocal
closure, so native acceptance alone is not sufficient for trustworthy sequence transport.

| Run | Model mismatch | Ambiguity | Low texture | Refinement | Search bound | Handover |
|---|---|---|---|---|---|---|
| G | 101 | 76 | 89 | 5 | 2 | 0 |
| A | 352 | 60 | 17 | 2 | 0 | 0 |
| B | 411 | 11 | 0 | 55 | 0 | 1 |
| C | 137 | 2 | 0 | 0 | 0 | 0 |
| D | 236 | 2 | 0 | 11 | 0 | 0 |
| E | 186 | 0 | 0 | 0 | 0 | 2 |
| F | 123 | 93 | 0 | 14 | 0 | 1 |

G provides three reciprocally supported independent anchor corrections; A–F currently
provide none. Closures and generated cached estimates are not
independent motion truth. The six-run comparison is exploratory validation of this new
stage; it does not establish accuracy of private pixels or a production reference-learning
pipeline. Regional conditioning, noise and actual model/lighting departures need controlled
investigation before broader curve extension or fusion. Translation remains available
with its existing contract rather than being replaced by the lower-coverage model.

Development outputs include local per-pair PNG blends and standalone HTML viewers with
camera/model controls, frame playback, masks, mapped grids, failure timelines and chained
versus anchored trajectories. Existing translation seed-segment outputs can be overlaid
separately, retaining amber predictions and green caliper evidence plus geometry provenance.
Detailed reports/pixels are private artifacts outside the repository; no source images,
annotations or per-frame geometry are published.
