# ADR-0021: Continuous profiles: a tapered derivative of Gaussian sampled at the point

- Status: Accepted
- Date: 2026-10-16

## Context

A caliper builds its profile by bilinear interpolation along the scan and then
differentiates it in 1-D. Between pixel centres a bilinear profile is linear, so its
derivative is piecewise constant in the profile's position, and a peak refined on it
clusters at particular sub-pixel phases. The phase sweep of the accuracy suite measures
that clustering as a locking index, the max/min ratio of a histogram of the located
positions' fractional parts. Every profile locator locks there, the default strip
included. A wider σ does not remove it, because the interpolation, not the smoothing, puts
the knots in.

The remedy is to evaluate the derivative of a Gaussian at the continuous point, straight
from the pixels: `R(p) = Σ_q I(q) ∇G_σ(p − q)·n`. A hard cut-off of its support makes `R`
jump whenever a pixel enters or leaves it, which moves a peak refined next to it. Bead
tracking, the strips of `measure` and later evidence fusion all need such samples along
arbitrary normals, with a validity mask.

## Decision

- **`vm_primitives::edge::ProfileSampler`** is the operator. It sits with the other edge
  primitives because it depends on nothing but an image view, and `measure` is one of
  several consumers.
  - `sample(img, mask, stations, offsets, cfg, out)` fills a caller-owned
    `[Option<ProfileSample>]`, station-major. A `ProfileStation` is a point and a direction,
    normalised on use. A `ProfileSample` is the smoothed value and the derivative along the
    direction. `ProfileSamplerConfig` holds σ alone.
  - It owns the kernel constants and the 1-D scratch, so a reused sampler allocates
    nothing (invariant 6).
- **The kernel.** It is `w(d) = exp(−|d|²/2σ²)·(1 − |d|²/r²)²` with `r = 4σ`, over the
  pixels strictly inside the disc of radius `r`.
  - The taper takes the weight and its gradient to zero at `r`, so value and derivative
    are C¹ in `p`.
  - The value is `Σ I w / Σ w`, exact on a constant.
  - The derivative is `Σ (I − value)·((q − p)·n/σ²)·w / C`. `C` is the kernel's continuous
    second moment along `n`, a closed form, so a ramp reads its slope. Subtracting the
    value makes a constant read exactly zero.
  - The radius is not configurable: 4σ is where the Gaussian factor is below 4·10⁻⁴.
- **The Gaussian is separable; the taper is not.** Each sample evaluates the 1-D Gaussian
  at its support's continuous column and row offsets, two exponentials per axis through a
  recurrence. The taper is a polynomial in `|d|²` per pixel. This is the part of the
  planned `filter` module this needs. Image-wide separable filtering stays in `filter`.
- **Validity is strict.** A sample is `None` when any pixel of its support lies outside the
  image, is excluded by the mask (zero), or is not finite. There is no valid-weight
  fraction: a partial support would bias the derivative towards the side that remains,
  and nothing would report how much. Border fill is never evidence.
- **Storage is `f32`, accumulation `f64`** (invariant 20). Points and offsets are `f32`;
  each sample's position and sums are `f64`.
- **`measure::Derivative::Continuous`** uses the sampler in every caliper placement.
  - Each profile sample is the mean, over the lines across, of the sampler's value and
    derivative along the scan direction. The response is scaled to input units per
    profile sample, like the other derivatives.
  - Peaks found on the samples are refined on the continuous response: a parabola
    through three evaluations `0.1σ` apart along the scan, iterated as Newton steps.
    The refinement has no grid of its own to lock to.
  - The level methods read the smoothed value as the smoothed profile.
  - An invalid sample reads 0 and carries no response. With `OffImage::Fill` a gradient
    peak is still found on the valid part; a peak next to an invalid sample, or whose
    refinement reaches one, is dropped. A level method rejects such a profile as
    `OffImage`.
- **It is an option, not the default.** The default derivative's results are what the
  accuracy envelopes, the can-end baseline and every existing recipe are pinned to, and
  the continuous profile reads every pixel of a disc per sample where the bilinear one
  reads four. The phase sweep's numbers and the costs, per σ, are in
  `docs/performance.md`. They show that the continuous profile stops locking at a σ of a
  few pixels, where a bilinear profile still locks.

## Alternatives

- **A configurable valid-weight fraction.** It would let a sample next to a mask or border
  count, with a bias that depends on the excluded side's content. Strict validity, with
  the support the caller chose through σ, is the definition that cannot mislead.
- **A hard cut-off at 3σ or 4σ.** It is cheaper by the taper's polynomial, and its jumps
  move a parabola-refined peak by several hundredths of a pixel.
- **Filtering the whole image with separable derivative kernels, then sampling the
  filtered images bilinearly.** It is cheaper when a profile is dense over a large area.
  But it reintroduces interpolation knots in the derivative, the cause of the locking,
  and it spends work on pixels no profile reads.
- **Refining with an analytic second derivative.** It needs a third kernel per sample.
  The finite-difference parabola converges in two or three steps, and its spacing is far
  below the kernel's width.
- **Making `Continuous` the default.** It would change results everywhere and cost time
  on every strip. Whether it should is a release decision, recorded in the backlog.

## Consequences

- Profiles along any normal, with a mask, are public, allocation-free API with Python
  parity (`ProfileSampler`).
- Any caliper, and through `BeadCaliper::profile` the bead tracker, can locate edges
  without locking to the grid. A caller chooses σ by the sweep's numbers.
- Even this operator has a residual sampling bias on an ideal, pixel-integrated step with
  no optical blur. It is odd in the phase and falls with σ and with blur; the sweep
  reports it.
- The phase sweep pins both the default and the continuous rows, so a change to either
  shows as a moved envelope.
