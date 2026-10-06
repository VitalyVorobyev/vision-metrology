//! Strip and caliper rows on CaliperBench's image model.
//!
//! The fixtures are CaliperBench's synthetic tiles (its `docs/synthetic.md`, "Image
//! model"): straight edges blurred by an isotropic Gaussian PSF of σ ∈ {0, 0.6, 1.2,
//! 2.5} px and integrated over each unit pixel. The pixel mean has a closed form, an
//! exact second finite difference of the second antiderivative of Φ (a first difference
//! for an axis-aligned edge), evaluated here in `f64`. Edges run at 0, 10, 30 and 45° to
//! the pixel grid and sit at subpixel phases 0, .25, .5 and .75; Gaussian noise of 0, 2
//! or 5 DN (seeded, 10 trials when noisy) is added before 8-bit quantisation. Levels are
//! 40 and 200 DN.
//!
//! Strips are 40 px long with 81 samples (0.5 px apart) and use the textbook settings
//! CaliperBench's baselines use: σ of one sample, a radius-3 Gaussian then central
//! differences, a response floor of 0.01 on a `[0, 1]` image, strict bounds.
//!
//! Each row reports, like the rest of the suite, the worst cell of its grid: a cell is
//! one blur, angle and noise level (and width, or caliper width), its errors pooled over
//! phases and trials; `bias` is the largest |mean error| and `sigma` the largest spread.
//!
//! - **Steps** (`strip_step_*`): one edge, scanned along its normal by each `Locate`
//!   method; `strip_oblique_*` crosses it at 15° and 30° with a strip 3 px wide.
//! - **Bars** (`strip_bar_*`): a bright bar, its two edges read as rising then falling.
//!   The centre row sweeps widths 2–10 px with noise; the width rows are noise-free and
//!   pin the systematic width bias, which for widths near the PSF is outward.
//! - **Calipers** with the default config: `MeasureRect` across a step for half-widths
//!   1–6 px, and `MeasureRadial` on a disc and `MeasureArc` across a spoke, for radii 20
//!   and 40 px.

use std::f64::consts::{FRAC_1_SQRT_2, PI};
use std::num::NonZeroUsize;

use vision_metrology::measure::{
    Caliper, Derivative, EdgeSelect, EdgeSequence, Locate, MeasureArc, MeasureConfig,
    MeasureRadial, MeasureRect, MeasureStrip, OffImage, PolaritySelect, ProfileConfig,
};
use vision_metrology::{Image, Point2f, SubpixRefine};

use super::{Measured, mean_std};

pub(super) const LO: f64 = 40.0;
pub(super) const HI: f64 = 200.0;
const PSF_SIGMAS: [f64; 4] = [0.0, 0.6, 1.2, 2.5];
const ANGLES_DEG: [f64; 4] = [0.0, 10.0, 30.0, 45.0];
const PHASES: [f64; 4] = [0.0, 0.25, 0.5, 0.75];
const NOISES_DN: [f64; 3] = [0.0, 2.0, 5.0];
const NOISY_TRIALS: u64 = 10;
const SIZE: usize = 64;
const STRIP_LEN: f64 = 40.0;
const STRIP_SAMPLES: usize = 81;

// ── CaliperBench's closed form ───────────────────────────────────────────

/// The standard normal CDF.
fn ndtr(z: f64) -> f64 {
    0.5 * libm::erfc(-z * FRAC_1_SQRT_2)
}

fn npdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * PI).sqrt()
}

/// First antiderivative of the blurred unit step `Φ(x/σ)`.
fn ramp1(x: f64, sigma: f64) -> f64 {
    if sigma == 0.0 {
        return x.max(0.0);
    }
    let z = x / sigma;
    sigma * (z * ndtr(z) + npdf(z))
}

/// Second antiderivative of the blurred unit step `Φ(x/σ)`.
fn ramp2(x: f64, sigma: f64) -> f64 {
    if sigma == 0.0 {
        return 0.5 * x.max(0.0).powi(2);
    }
    let z = x / sigma;
    0.5 * sigma * sigma * ((z * z + 1.0) * ndtr(z) + z * npdf(z))
}

/// Mean of `Φ(s/σ)` over a unit pixel whose centre has signed distance `offset` from the
/// edge, `(a, b)` the edge normal's components.
fn step_coverage(offset: f64, a: f64, b: f64, sigma: f64) -> f64 {
    let (a, b) = if a.abs() >= b.abs() {
        (a.abs(), b.abs())
    } else {
        (b.abs(), a.abs())
    };
    if offset.abs() >= 0.5 * (a + b) + 12.0 * sigma {
        return if offset > 0.0 { 1.0 } else { 0.0 };
    }
    if b < 1e-12 {
        return (ramp1(offset + a / 2.0, sigma) - ramp1(offset - a / 2.0, sigma)) / a;
    }
    (ramp2(offset + (a + b) / 2.0, sigma)
        - ramp2(offset + (a - b) / 2.0, sigma)
        - ramp2(offset - (a - b) / 2.0, sigma)
        + ramp2(offset - (a + b) / 2.0, sigma))
        / (a * b)
}

/// The unit normal at `deg` from +x, with CaliperBench's exact zeros on the axes.
fn normal(deg: f64) -> (f64, f64) {
    let (s, c) = deg.to_radians().sin_cos();
    let snap = |v: f64| if v.abs() < 1e-12 { 0.0 } else { v };
    (snap(c), snap(s))
}

/// A noise-free `size × size` image, in DN, of parallel straight edges along `n`:
/// `LO + (HI − LO)·Σ sign·coverage(n·p − distance)` over `(distance, sign)` steps.
pub(super) fn render_steps(
    size: usize,
    n: (f64, f64),
    steps: &[(f64, f64)],
    sigma: f64,
) -> Vec<f64> {
    let mut out = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let proj = n.0 * x as f64 + n.1 * y as f64;
            let cover: f64 = steps
                .iter()
                .map(|&(d, sign)| sign * step_coverage(proj - d, n.0, n.1, sigma))
                .sum();
            out.push(LO + (HI - LO) * cover);
        }
    }
    out
}

/// Seeded Gaussian noise: splitmix64 and Box–Muller.
struct Gauss(u64);

impl Gauss {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        ((z >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.unit(), self.unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * PI * v).cos()
    }
}

/// Add `noise_dn` of Gaussian noise (seeded by `seed`), round and clip to 8 bits.
fn quantize(clean: &[f64], noise_dn: f64, seed: u64) -> Vec<u8> {
    let mut rng = Gauss(seed);
    clean
        .iter()
        .map(|&v| {
            let noisy = if noise_dn > 0.0 {
                v + noise_dn * rng.normal()
            } else {
                v
            };
            noisy.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

/// 8-bit pixels as `f32` in `[0, 1]`, the scale CaliperBench measures on.
fn unit_image(size: usize, px: &[u8]) -> Image<f32> {
    let data = px.iter().map(|&v| f32::from(v) / 255.0).collect();
    Image::from_vec(size, size, data).expect("valid image")
}

fn u8_image(size: usize, px: Vec<u8>) -> Image<u8> {
    Image::from_vec(size, size, px).expect("valid image")
}

/// The seeds of one cell: one noise-free pass, or `NOISY_TRIALS` noisy ones.
fn trials(noise_dn: f64) -> u64 {
    if noise_dn > 0.0 { NOISY_TRIALS } else { 1 }
}

fn seed(cell: u64, trial: u64) -> u64 {
    0xA076_1D64_78BD_642F ^ cell.wrapping_mul(0xE703_7ED1_A0B4_28DB) ^ trial
}

// ── strips ───────────────────────────────────────────────────────────────

/// The textbook settings for a strip sampled every `spacing` px.
fn textbook(locate: Locate, select: EdgeSelect, spacing: f32) -> MeasureConfig {
    MeasureConfig {
        threshold: 0.01,
        polarity: PolaritySelect::Any,
        select,
        locate,
        max_obliquity_deg: 180.0,
        profile: ProfileConfig {
            sigma: spacing,
            derivative: Derivative::SmoothThenCentral {
                radius_px: 3.0 * spacing,
            },
            off_image: OffImage::Reject,
            ..ProfileConfig::default()
        },
    }
}

fn in_order(first: PolaritySelect, second: Option<PolaritySelect>) -> EdgeSelect {
    EdgeSelect::StrongestInOrder(EdgeSequence { first, second })
}

const PARABOLIC: Locate = Locate::GradientPeak {
    refine: SubpixRefine::Parabolic3,
};

/// A strip `STRIP_LEN` long through `(32, 32)` shifted 0.3 px along `u`, `half_width`
/// wide with `across` lines; and the scan distance from its start at which its centre
/// line meets the line `n·p = d`.
fn strip_through(
    u: (f64, f64),
    half_width: f32,
    across: usize,
    n: (f64, f64),
    d: f64,
) -> (MeasureStrip, f64) {
    let c = 32.0;
    let mid = (c + 0.3 * u.0, c + 0.3 * u.1);
    let start = (mid.0 - 0.5 * STRIP_LEN * u.0, mid.1 - 0.5 * STRIP_LEN * u.1);
    let end = (mid.0 + 0.5 * STRIP_LEN * u.0, mid.1 + 0.5 * STRIP_LEN * u.1);
    let strip = MeasureStrip {
        start: Point2f::new(start.0 as f32, start.1 as f32),
        end: Point2f::new(end.0 as f32, end.1 as f32),
        half_width,
        samples: NonZeroUsize::new(STRIP_SAMPLES),
        across: NonZeroUsize::new(across),
    };
    // From the start the strip actually got, after rounding to `f32`.
    let (sx, sy) = (f64::from(strip.start.x), f64::from(strip.start.y));
    let truth = (d - n.0 * sx - n.1 * sy) / (n.0 * u.0 + n.1 * u.1);
    (strip, truth)
}

/// Where the edge sits in a cell: `n·p = n·(32, 32) + phase`.
fn edge_at(n: (f64, f64), phase: f64) -> f64 {
    32.0 * (n.0 + n.1) + phase
}

/// Worst cell of a single-step sweep: one edge (rising along the scan), crossed by a
/// strip at `obliquity_deg` from its normal, `half_width`/`across` wide.
fn step_sweep(locate: Locate, obliquity_deg: f64, half_width: f32, across: usize) -> Measured {
    let select = in_order(PolaritySelect::Rising, None);
    let spacing = (STRIP_LEN / (STRIP_SAMPLES - 1) as f64) as f32;
    let mut cells = Vec::new();
    for (si, &psf) in PSF_SIGMAS.iter().enumerate() {
        for (ai, &angle) in ANGLES_DEG.iter().enumerate() {
            let n = normal(angle);
            let u = normal(angle + obliquity_deg);
            for (ni, &noise) in NOISES_DN.iter().enumerate() {
                let mut errs = Vec::new();
                for (pi, &phase) in PHASES.iter().enumerate() {
                    let d = edge_at(n, phase);
                    let clean = render_steps(SIZE, n, &[(d, 1.0)], psf);
                    let (strip, truth) = strip_through(u, half_width, across, n, d);
                    for trial in 0..trials(noise) {
                        let cell = ((si * 4 + ai) * 3 + ni) as u64 * 4 + pi as u64;
                        let img = unit_image(SIZE, &quantize(&clean, noise, seed(cell, trial)));
                        let mut cal = Caliper::strip(strip, textbook(locate, select, spacing));
                        let edges = cal.measure(&img.as_view()).unwrap_or_else(|r| {
                            panic!(
                                "step rejected: {locate:?} psf={psf} angle={angle} \
                                 obliquity={obliquity_deg} noise={noise} phase={phase}: {r:?}"
                            )
                        });
                        errs.push((f64::from(edges[0].t) - truth) as f32);
                    }
                }
                cells.push(mean_std(&errs));
            }
        }
    }
    Measured::worst_of(cells.into_iter())
}

pub(super) fn step_parabolic_sweep() -> Measured {
    step_sweep(PARABOLIC, 0.0, 0.0, 1)
}

pub(super) fn step_gaussian_sweep() -> Measured {
    let locate = Locate::GradientPeak {
        refine: SubpixRefine::Gaussian3,
    };
    step_sweep(locate, 0.0, 0.0, 1)
}

pub(super) fn step_midpoint_sweep() -> Measured {
    let locate = Locate::MidpointCrossing {
        endpoint_samples: NonZeroUsize::new(3).expect("nonzero"),
        min_contrast: 0.05,
    };
    step_sweep(locate, 0.0, 0.0, 1)
}

pub(super) fn step_half_contrast_sweep() -> Measured {
    let locate = Locate::HalfContrast {
        flank_near_px: 3.0,
        flank_far_px: 8.0,
        tol_px: 0.01,
        max_iter: NonZeroUsize::new(5).expect("nonzero"),
        min_contrast: 0.05,
    };
    step_sweep(locate, 0.0, 0.0, 1)
}

/// A strip 3 px wide (3 lines) crossing the edge at 15° from its normal.
pub(super) fn oblique_15_sweep() -> Measured {
    step_sweep(PARABOLIC, 15.0, 1.0, 3)
}

/// A strip 3 px wide (3 lines) crossing the edge at 30° from its normal.
pub(super) fn oblique_30_sweep() -> Measured {
    step_sweep(PARABOLIC, 30.0, 1.0, 3)
}

/// What a bar row reports.
#[derive(Clone, Copy)]
enum BarQuantity {
    Center,
    Width,
}

/// Worst cell of a bright-bar sweep over `widths` and `noises`, scanned along its normal
/// with the parabolic method: the edges are the strongest rising, then the strongest
/// falling.
fn bar_sweep(widths: &[f64], noises: &[f64], quantity: BarQuantity) -> Measured {
    let select = in_order(PolaritySelect::Rising, Some(PolaritySelect::Falling));
    let spacing = (STRIP_LEN / (STRIP_SAMPLES - 1) as f64) as f32;
    let mut cells = Vec::new();
    for (wi, &w) in widths.iter().enumerate() {
        for (si, &psf) in PSF_SIGMAS.iter().enumerate() {
            for (ai, &angle) in ANGLES_DEG.iter().enumerate() {
                let n = normal(angle);
                for (ni, &noise) in noises.iter().enumerate() {
                    let mut errs = Vec::new();
                    for (pi, &phase) in PHASES.iter().enumerate() {
                        let d = edge_at(n, phase);
                        let steps = [(d - 0.5 * w, 1.0), (d + 0.5 * w, -1.0)];
                        let clean = render_steps(SIZE, n, &steps, psf);
                        let (strip, center) = strip_through(n, 0.0, 1, n, d);
                        for trial in 0..trials(noise) {
                            let cell = (((wi * 4 + si) * 4 + ai) * 3 + ni) as u64 * 4 + pi as u64;
                            let px = quantize(&clean, noise, seed(cell | (1 << 40), trial));
                            let img = unit_image(SIZE, &px);
                            let mut cal =
                                Caliper::strip(strip, textbook(PARABOLIC, select, spacing));
                            let e = cal.measure(&img.as_view()).unwrap_or_else(|r| {
                                panic!(
                                    "bar rejected: w={w} psf={psf} angle={angle} \
                                     noise={noise} phase={phase}: {r:?}"
                                )
                            });
                            let (t1, t2) = (f64::from(e[0].t), f64::from(e[1].t));
                            errs.push(match quantity {
                                BarQuantity::Center => (0.5 * (t1 + t2) - center) as f32,
                                BarQuantity::Width => ((t2 - t1) - w) as f32,
                            });
                        }
                    }
                    cells.push(mean_std(&errs));
                }
            }
        }
    }
    Measured::worst_of(cells.into_iter())
}

pub(super) fn bar_center_sweep() -> Measured {
    bar_sweep(&[2.0, 3.0, 5.0, 10.0], &NOISES_DN, BarQuantity::Center)
}

// The width rows are noise-free: they pin the width's systematic bias. Under noise each
// edge jitters as a step does, and the width by √2 of that.

pub(super) fn bar_width_w10_sweep() -> Measured {
    bar_sweep(&[10.0], &[0.0], BarQuantity::Width)
}

pub(super) fn bar_width_w2_w3_sweep() -> Measured {
    bar_sweep(&[2.0, 3.0], &[0.0], BarQuantity::Width)
}

// ── rect, arc and radial calipers ────────────────────────────────────────

/// `MeasureRect` with the default config (derivative of Gaussian, σ 1 px, parabola,
/// threshold 5 DN) across a pixel-integrated step, for caliper half-widths 1, 3 and 6.
pub(super) fn rect_pixel_integrated_sweep() -> Measured {
    let mut cells = Vec::new();
    for (hi, &half_width) in [1.0f32, 3.0, 6.0].iter().enumerate() {
        for (si, &psf) in PSF_SIGMAS.iter().enumerate() {
            for (ai, &angle) in ANGLES_DEG.iter().enumerate() {
                let n = normal(angle);
                for (ni, &noise) in NOISES_DN.iter().enumerate() {
                    let mut errs = Vec::new();
                    for (pi, &phase) in PHASES.iter().enumerate() {
                        let d = edge_at(n, phase);
                        let clean = render_steps(SIZE, n, &[(d, 1.0)], psf);
                        let center = Point2f::new(32.3, 31.8);
                        let truth = d - n.0 * f64::from(center.x) - n.1 * f64::from(center.y);
                        let rect = MeasureRect {
                            center,
                            angle: angle.to_radians() as f32,
                            half_len: 15.0,
                            half_width,
                        };
                        for trial in 0..trials(noise) {
                            let cell = (((hi * 4 + si) * 4 + ai) * 3 + ni) as u64 * 4 + pi as u64;
                            let img = u8_image(
                                SIZE,
                                quantize(&clean, noise, seed(cell | (2 << 40), trial)),
                            );
                            let cfg = MeasureConfig {
                                polarity: PolaritySelect::Rising,
                                select: EdgeSelect::Strongest,
                                ..MeasureConfig::default()
                            };
                            let mut cal = Caliper::rect(rect, cfg);
                            let e = cal.measure(&img.as_view()).unwrap_or_else(|r| {
                                panic!(
                                    "rect rejected: half_width={half_width} psf={psf} \
                                     angle={angle} noise={noise} phase={phase}: {r:?}"
                                )
                            });
                            errs.push((f64::from(e[0].t) - truth) as f32);
                        }
                    }
                    cells.push(mean_std(&errs));
                }
            }
        }
    }
    Measured::worst_of(cells.into_iter())
}

/// A bright disc of radius `r` centred at `c`, each pixel integrated with the
/// straight-edge closed form along its own radial direction: the iso-levels are circles
/// and the 50 % level sits on `r` exactly. (The σ²κ/2 shift a real blurred circle's
/// crossing has is not modelled; the row measures the caliper, not the optics.)
fn render_disc(size: usize, c: (f64, f64), r: f64, sigma: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(size * size);
    for y in 0..size {
        for x in 0..size {
            let (dx, dy) = (x as f64 - c.0, y as f64 - c.1);
            let rho = dx.hypot(dy).max(1e-9);
            let cover = step_coverage(r - rho, dx / rho, dy / rho, sigma);
            out.push(LO + (HI - LO) * cover);
        }
    }
    out
}

/// Radial position of a disc's edge through `MeasureRadial` (default config, 8 caliper
/// angles), and arc position of a radial edge (a spoke) through `MeasureArc`, both in px.
pub(super) fn arc_radial_sweep() -> Measured {
    let sigmas = [0.6, 1.2, 2.5];
    let radii = [20.0, 40.0];
    let mut cells = Vec::new();

    // Radial: the disc's edge, read outward (bright to dark).
    let size = 112;
    let c = (56.3, 55.8);
    let cfg = MeasureConfig {
        polarity: PolaritySelect::Falling,
        select: EdgeSelect::Strongest,
        ..MeasureConfig::default()
    };
    for (ri, &radius) in radii.iter().enumerate() {
        for (si, &psf) in sigmas.iter().enumerate() {
            for (hi, &half_width) in [2.0f32, 5.0, 10.0].iter().enumerate() {
                for (ni, &noise) in NOISES_DN.iter().enumerate() {
                    let mut errs = Vec::new();
                    for (pi, &phase) in PHASES.iter().enumerate() {
                        let truth = radius + phase;
                        let clean = render_disc(size, c, truth, psf);
                        for trial in 0..trials(noise) {
                            let cell = (((ri * 3 + si) * 3 + hi) * 3 + ni) as u64 * 4 + pi as u64;
                            let px = quantize(&clean, noise, seed(cell | (3 << 40), trial));
                            let img = u8_image(size, px);
                            for k in 0..8 {
                                let angle = (10.0 + 45.0 * f64::from(k)).to_radians() as f32;
                                let radial = MeasureRadial {
                                    center: Point2f::new(c.0 as f32, c.1 as f32),
                                    radius: radius as f32,
                                    angle,
                                    half_len: 8.0,
                                    half_width,
                                };
                                let mut cal = Caliper::radial(radial, cfg);
                                let e = cal.measure(&img.as_view()).unwrap_or_else(|r| {
                                    panic!(
                                        "radial rejected: r={radius} psf={psf} \
                                         half_width={half_width} noise={noise}: {r:?}"
                                    )
                                });
                                errs.push((radius + f64::from(e[0].t) - truth) as f32);
                            }
                        }
                    }
                    cells.push(mean_std(&errs));
                }
            }
        }
    }

    // Arc: a spoke through the centre, crossed along a circle (dark to bright).
    let size = 96;
    let c = (48.2, 47.7);
    let cfg = MeasureConfig {
        polarity: PolaritySelect::Rising,
        select: EdgeSelect::Strongest,
        ..MeasureConfig::default()
    };
    for (ri, &radius) in radii.iter().enumerate() {
        for (si, &psf) in sigmas.iter().enumerate() {
            for (ai, &spoke_deg) in [10.0f64, 30.0, 45.0, 80.0].iter().enumerate() {
                for (ni, &noise) in NOISES_DN.iter().enumerate() {
                    let mut errs = Vec::new();
                    for (pi, &phase) in PHASES.iter().enumerate() {
                        // The spoke at angle ψ: bright where the angle exceeds ψ.
                        let psi = spoke_deg.to_radians() + phase / radius;
                        let n = (-psi.sin(), psi.cos());
                        let d = n.0 * c.0 + n.1 * c.1;
                        // The line through the centre; the arc meets only its spoke at ψ.
                        let clean = render_steps(size, n, &[(d, 1.0)], psf);
                        let start = spoke_deg.to_radians() - 12.0 / radius;
                        let truth = (psi - start) * radius;
                        let arc = MeasureArc {
                            center: Point2f::new(c.0 as f32, c.1 as f32),
                            radius: radius as f32,
                            angle_start: start as f32,
                            angle_extent: (24.0 / radius) as f32,
                            half_width: 3.0,
                        };
                        for trial in 0..trials(noise) {
                            let cell = (((ri * 3 + si) * 4 + ai) * 3 + ni) as u64 * 4 + pi as u64;
                            let px = quantize(&clean, noise, seed(cell | (4 << 40), trial));
                            let img = u8_image(size, px);
                            let mut cal = Caliper::arc(arc, cfg);
                            let e = cal.measure(&img.as_view()).unwrap_or_else(|r| {
                                panic!(
                                    "arc rejected: r={radius} psf={psf} spoke={spoke_deg} \
                                     noise={noise}: {r:?}"
                                )
                            });
                            errs.push((f64::from(e[0].t) - truth) as f32);
                        }
                    }
                    cells.push(mean_std(&errs));
                }
            }
        }
    }
    Measured::worst_of(cells.into_iter())
}

#[test]
fn the_closed_form_matches_quadrature() {
    // Without blur the pixel mean is the area on the bright side.
    assert!((step_coverage(0.2, 1.0, 0.0, 0.0) - 0.7).abs() < 1e-12);
    let (a, b) = normal(45.0);
    assert!((step_coverage(0.0, a, b, 0.0) - 0.5).abs() < 1e-12);
    // 32 × 32 midpoint quadrature of Φ over the pixel, against the closed form.
    for &(a_deg, sigma) in &[(0.0, 0.6), (10.0, 0.6), (30.0, 1.2), (45.0, 2.5)] {
        let (a, b) = normal(a_deg);
        for &offset in &[-1.3, -0.4, 0.0, 0.2, 0.9] {
            let k = 32;
            let mut sum = 0.0;
            for i in 0..k {
                for j in 0..k {
                    let u = (i as f64 + 0.5) / k as f64 - 0.5;
                    let v = (j as f64 + 0.5) / k as f64 - 0.5;
                    let s = offset + a * u + b * v;
                    sum += ndtr(s / sigma);
                }
            }
            let quad = sum / (k * k) as f64;
            let closed = step_coverage(offset, a, b, sigma);
            assert!(
                (quad - closed).abs() < 1e-4,
                "angle {a_deg} σ {sigma} offset {offset}: quadrature {quad}, closed form {closed}"
            );
        }
    }
}
