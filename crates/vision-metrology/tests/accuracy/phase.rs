//! Phase-swept edge rows: the worst bias over a fine sub-pixel phase sweep, and the
//! locking index.
//!
//! A locator can be unbiased at phases 0, ¼, ½ and ¾ and biased between them; on real
//! data that shows as located positions clustering at particular sub-pixel phases. These
//! rows sweep the phase finely on `strip`'s fixture: a straight step box-sampled over
//! each unit pixel (CaliperBench's closed form), optionally blurred by a Gaussian PSF,
//! levels 40 and 200 DN, quantised to 8 bits.
//!
//! - **Grid:** 20 sub-pixel phases × 4 edge angles (0, 15, 30, 45° from +x) × 3 blurs
//!   (none, σ 0.5 and σ 1.0 px), each noise-free and under 2 DN of seeded Gaussian noise
//!   (6 trials). A cell is one blur, angle, noise level and phase. `bias` is the worst
//!   |mean error| over the cells and `sigma` the worst spread: the per-phase bias, not one
//!   pooled over phases, so a bias that changes sign between phases is not averaged away.
//! - **Locking index:** for each blur and angle, 200 phases × 3 trials under 2 DN of
//!   noise; the located positions' pixel `x` coordinates go into a 10-bin histogram of
//!   their fractional parts, and the index is its max/min bin ratio. The sweep moves the
//!   true position of every sample by exactly one pixel in `x`, in equal steps, so an
//!   unbiased locator's histogram is flat up to noise (index ≈ 1) and a phase-dependent
//!   bias `b(φ)` shows as a density `1 / (1 + b′(φ))`. The row reports the worst blur and
//!   angle.
//! - **Continuous profiles:** the default strip with `Derivative::Continuous` at σ 1, 1.5
//!   and 2.5 px, and with the default derivative at σ 2.5 px for comparison.
//!
//! The error is the signed distance along the true normal, in px. Each image carries
//! several independent samples: 7 strips (or 5 rect calipers) spaced along the edge, one
//! 1-D profile per image row, and every interior edgel of the 2-D detector.
//!
//! The `x` coordinate a sample contributes to the histogram is the one the real-data
//! diagnostic (`tools/registration_eval/phase_histogram.py`) reads: the located point's
//! `x` for a profile locator scanning along the normal, the row-crossing `x` for a row
//! profile, and, for a 2-D edgel, the `x` where its tangent crosses the edgel's own pixel
//! row. (An edgel moves from its pixel centre along the normal; at an oblique normal its
//! raw `x` therefore covers only part of each pixel even when it is unbiased. The row
//! crossing removes that geometric gap.)

use std::num::NonZeroUsize;

use vision_metrology::measure::{
    Caliper, Derivative, EdgeSelect, Locate, MeasureConfig, MeasureRect, MeasureStrip,
    PolaritySelect, ProfileConfig,
};
use vision_metrology::{
    Edge1DConfig, Edge1DDetector, Edge2DConfig, Edge2DDetector, EdgePolarity, Image, Point2f,
    SubpixRefine,
};

use super::strip::{
    PARABOLIC, in_order, normal, quantize, render_steps, textbook, u8_image, unit_image,
};

const SIZE: usize = 64;
const BLURS: [f64; 3] = [0.0, 0.5, 1.0];
const ANGLES_DEG: [f64; 4] = [0.0, 15.0, 30.0, 45.0];
const GRID_PHASES: usize = 20;
const GRID_NOISES_DN: [f64; 2] = [0.0, 2.0];
const GRID_TRIALS: u64 = 6;
const DENSE_PHASES: usize = 200;
const DENSE_TRIALS: u64 = 3;
const DENSE_NOISE_DN: f64 = 2.0;
pub(super) const BINS: usize = 10;
/// Tangential offsets of the strips and of the rect calipers along the edge, in px.
const STRIP_OFFSETS: [f64; 7] = [-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0];
const RECT_OFFSETS: [f64; 5] = [-12.0, -6.0, 0.0, 6.0, 12.0];
/// The interior a row profile's truth or a 2-D edgel must lie in.
const INTERIOR: (f64, f64) = (12.0, 52.0);

/// One located sample: its error along the true normal and the `x` the histogram reads.
#[derive(Clone, Copy)]
struct Hit {
    err: f64,
    x: f64,
}

/// How a family of locators moves its samples' truth by one pixel in `x`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// Samples on lines along the normal (strips, rect calipers): the edge moves `1/nx`.
    Scan,
    /// Samples on pixel rows (row profiles, 2-D edgels): the edge moves `nx`.
    Axis,
}

/// One noisy (or clean) image of the edge `n·p = d`.
struct Scene {
    n: (f64, f64),
    d: f64,
    px: Vec<u8>,
    img: Image<u8>,
    unit: Image<f32>,
}

struct Tools {
    edge1d: Edge1DDetector,
    edge2d: Edge2DDetector,
}

/// A locator, its family, and its envelopes: |bias| and sigma in px, the locking index.
struct Locator {
    name: &'static str,
    family: Family,
    locate: fn(&Scene, &mut Tools, &mut Vec<Hit>),
    bias_envelope: f64,
    sigma_envelope: f64,
    locking_envelope: f64,
}

// ── strips and rect calipers ─────────────────────────────────────────────

/// A strip 40 px long along `n` through `(32, 32) + 0.3 n + s t`, `t` the edge tangent.
fn strip_at(n: (f64, f64), s: f64, samples: Option<usize>) -> MeasureStrip {
    let t = (-n.1, n.0);
    let mid = (32.0 + 0.3 * n.0 + s * t.0, 32.0 + 0.3 * n.1 + s * t.1);
    let p = |k: f64| Point2f::new((mid.0 + k * n.0) as f32, (mid.1 + k * n.1) as f32);
    MeasureStrip {
        start: p(-20.0),
        end: p(20.0),
        half_width: 0.0,
        samples: samples.and_then(NonZeroUsize::new),
        across: NonZeroUsize::new(1),
    }
}

/// Measure every strip on the `[0, 1]` image (`unit`) or the 8-bit one.
fn strips(
    scene: &Scene,
    cfg: MeasureConfig,
    samples: Option<usize>,
    unit: bool,
    out: &mut Vec<Hit>,
) {
    let n = scene.n;
    for &s in &STRIP_OFFSETS {
        let strip = strip_at(n, s, samples);
        let (sx, sy) = (f64::from(strip.start.x), f64::from(strip.start.y));
        let truth = scene.d - n.0 * sx - n.1 * sy;
        let mut cal = Caliper::strip(strip, cfg);
        let found = if unit {
            cal.measure(&scene.unit.as_view())
        } else {
            cal.measure(&scene.img.as_view())
        };
        let e = found.unwrap_or_else(|r| panic!("strip rejected: n={n:?} d={}: {r:?}", scene.d));
        let t = f64::from(e[0].t);
        out.push(Hit {
            err: t - truth,
            x: sx + t * n.0,
        });
    }
}

/// CaliperBench's textbook strip: 81 samples 0.5 px apart, σ 0.5 px, central differences.
fn textbook_strip(scene: &Scene, locate: Locate, out: &mut Vec<Hit>) {
    let cfg = textbook(locate, in_order(PolaritySelect::Rising, None), 0.5);
    strips(scene, cfg, Some(81), true, out);
}

fn strip_parabolic(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    textbook_strip(scene, PARABOLIC, out);
}

fn strip_gaussian(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    let locate = Locate::GradientPeak {
        refine: SubpixRefine::Gaussian3,
    };
    textbook_strip(scene, locate, out);
}

fn strip_midpoint(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    let locate = Locate::MidpointCrossing {
        endpoint_samples: NonZeroUsize::new(3).expect("nonzero"),
        min_contrast: 0.05,
    };
    textbook_strip(scene, locate, out);
}

fn strip_half_contrast(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    let locate = Locate::HalfContrast {
        flank_near_px: 3.0,
        flank_far_px: 8.0,
        tol_px: 0.01,
        max_iter: NonZeroUsize::new(5).expect("nonzero"),
        min_contrast: 0.05,
    };
    textbook_strip(scene, locate, out);
}

/// The library's default strip: one sample per pixel, derivative of Gaussian σ 1 px.
fn strip_default(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    let cfg = MeasureConfig {
        polarity: PolaritySelect::Rising,
        select: EdgeSelect::Strongest,
        ..MeasureConfig::default()
    };
    strips(scene, cfg, None, false, out);
}

/// The default strip at σ `sigma` px, with `derivative`: `Derivative::Continuous` rows at
/// σ 1, 1.5 and 2.5 px, and the default derivative at σ 2.5 px, which shows that a wider
/// kernel alone does not remove the bilinear profile's locking.
fn strip_sigma(scene: &Scene, sigma: f32, derivative: Derivative, out: &mut Vec<Hit>) {
    let cfg = MeasureConfig {
        polarity: PolaritySelect::Rising,
        select: EdgeSelect::Strongest,
        profile: ProfileConfig {
            sigma,
            derivative,
            ..ProfileConfig::default()
        },
        ..MeasureConfig::default()
    };
    strips(scene, cfg, None, false, out);
}

fn strip_default_sigma25(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    strip_sigma(scene, 2.5, Derivative::DerivativeOfGaussian, out);
}

fn strip_continuous_sigma10(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    strip_sigma(scene, 1.0, Derivative::Continuous, out);
}

fn strip_continuous_sigma15(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    strip_sigma(scene, 1.5, Derivative::Continuous, out);
}

fn strip_continuous_sigma25(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    strip_sigma(scene, 2.5, Derivative::Continuous, out);
}

/// `MeasureRect` with the default config, 30 px long and 5 px wide, along the normal.
fn rect_default(scene: &Scene, _: &mut Tools, out: &mut Vec<Hit>) {
    let n = scene.n;
    let cfg = MeasureConfig {
        polarity: PolaritySelect::Rising,
        select: EdgeSelect::Strongest,
        ..MeasureConfig::default()
    };
    for &s in &RECT_OFFSETS {
        let center = Point2f::new((32.3 - s * n.1) as f32, (31.8 + s * n.0) as f32);
        let (cx, cy) = (f64::from(center.x), f64::from(center.y));
        let rect = MeasureRect {
            center,
            angle: n.1.atan2(n.0) as f32,
            half_len: 15.0,
            half_width: 2.5,
        };
        let mut cal = Caliper::rect(rect, cfg);
        let e = cal
            .measure(&scene.img.as_view())
            .unwrap_or_else(|r| panic!("rect rejected: n={n:?} d={}: {r:?}", scene.d));
        let t = f64::from(e[0].t);
        out.push(Hit {
            err: t - (scene.d - n.0 * cx - n.1 * cy),
            x: cx + t * n.0,
        });
    }
}

// ── row profiles and the 2-D detector ────────────────────────────────────

/// `Edge1DDetector` (σ 1 px) on every image row whose crossing lies in the interior; the
/// strongest rising peak of each row.
fn rows(scene: &Scene, tools: &mut Tools, refine: SubpixRefine, out: &mut Vec<Hit>) {
    let n = scene.n;
    let cfg = Edge1DConfig {
        sigma: 1.0,
        refine,
        ..Edge1DConfig::default()
    };
    for y in 0..SIZE {
        let truth = (scene.d - n.1 * y as f64) / n.0;
        if !(INTERIOR.0..=INTERIOR.1).contains(&truth) {
            continue;
        }
        let row = &scene.px[y * SIZE..(y + 1) * SIZE];
        let peaks = tools.edge1d.detect_in(row, &cfg);
        let peak = peaks
            .iter()
            .filter(|p| p.polarity == EdgePolarity::Rising)
            .max_by(|a, b| a.strength.total_cmp(&b.strength))
            .unwrap_or_else(|| panic!("no rising edge on row {y}: n={n:?} d={}", scene.d));
        let x = f64::from(peak.x);
        out.push(Hit {
            err: (x - truth) * n.0,
            x,
        });
    }
}

fn edge1d_parabolic(scene: &Scene, tools: &mut Tools, out: &mut Vec<Hit>) {
    rows(scene, tools, SubpixRefine::Parabolic3, out);
}

fn edge1d_gaussian(scene: &Scene, tools: &mut Tools, out: &mut Vec<Hit>) {
    rows(scene, tools, SubpixRefine::Gaussian3, out);
}

fn edge1d_centroid(scene: &Scene, tools: &mut Tools, out: &mut Vec<Hit>) {
    rows(scene, tools, SubpixRefine::Centroid { radius: 2 }, out);
}

/// `Edge2DDetector` with its default config (binomial pre-smoothing, automatic
/// hysteresis, `Subpix2D::ParabolicAlongNormal`): every interior edgel within 1.5 px of
/// the true line.
fn edge2d_parabolic(scene: &Scene, tools: &mut Tools, out: &mut Vec<Hit>) {
    let n = scene.n;
    let edgels = tools
        .edge2d
        .detect(&scene.img.as_view(), &Edge2DConfig::default());
    let inside = |v: f32| (INTERIOR.0..=INTERIOR.1).contains(&f64::from(v));
    let before = out.len();
    for e in edgels.iter().filter(|e| inside(e.p.x) && inside(e.p.y)) {
        let (px, py) = (f64::from(e.p.x), f64::from(e.p.y));
        let err = n.0 * px + n.1 * py - scene.d;
        if err.abs() > 1.5 || e.n.x.abs() < 0.5 {
            continue;
        }
        let x = px + (py - e.idx.1 as f64) * f64::from(e.n.y) / f64::from(e.n.x);
        out.push(Hit { err, x });
    }
    assert!(
        out.len() > before,
        "no interior edgels: n={n:?} d={}",
        scene.d
    );
}

// ── the sweep ────────────────────────────────────────────────────────────

/// What a row reports, per blur (the order of [`BLURS`]): the worst |bias| and sigma in
/// px over angles, phases and noise levels, and the worst locking index over angles.
#[derive(Debug, Clone, Copy, Default)]
struct Phased {
    bias: [f64; BLURS.len()],
    sigma: [f64; BLURS.len()],
    locking: [f64; BLURS.len()],
}

fn worst(v: &[f64]) -> f64 {
    v.iter().copied().fold(0.0, f64::max)
}

fn mean_std(v: &[Hit]) -> (f64, f64) {
    let n = v.len() as f64;
    let mean = v.iter().map(|h| h.err).sum::<f64>() / n;
    let var = v.iter().map(|h| (h.err - mean).powi(2)).sum::<f64>() / n;
    (mean, var.sqrt())
}

/// The max/min ratio of a histogram's bins (infinite when a bin is empty).
pub(super) fn locking_index(hist: &[u64; BINS]) -> f64 {
    let max = hist.iter().copied().max().unwrap_or(0) as f64;
    let min = hist.iter().copied().min().unwrap_or(0) as f64;
    if min == 0.0 { f64::INFINITY } else { max / min }
}

pub(super) fn bin(x: f64) -> usize {
    ((x - x.floor()) * BINS as f64).min(BINS as f64 - 1.0) as usize
}

pub(super) fn seed(parts: [u64; 5]) -> u64 {
    parts.iter().fold(0x5851_F42D_4C95_7F2D, |h, &p| {
        (h ^ p).wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(29)
    })
}

fn scene(clean: &[f64], noise: f64, seed: u64, n: (f64, f64), d: f64) -> Scene {
    let px = quantize(clean, noise, seed);
    Scene {
        n,
        d,
        unit: unit_image(SIZE, &px),
        img: u8_image(SIZE, px.clone()),
        px,
    }
}

/// Run every locator of `family` over the grid and the dense sweep, on shared images.
fn sweep(family: Family, locators: &[&Locator]) -> Vec<Phased> {
    let mut tools = Tools {
        edge1d: Edge1DDetector::new(1.0),
        edge2d: Edge2DDetector::new(),
    };
    let mut out = vec![Phased::default(); locators.len()];
    let mut hits: Vec<Vec<Hit>> = vec![Vec::new(); locators.len()];
    let fam = family as u64;
    for (bi, &blur) in BLURS.iter().enumerate() {
        for (ai, &angle) in ANGLES_DEG.iter().enumerate() {
            let n = normal(angle);
            // The shift of `d` that moves every sample's truth one pixel in `x`.
            let rate = match family {
                Family::Scan => 1.0 / n.0,
                Family::Axis => n.0,
            };
            let d0 = 32.0 * (n.0 + n.1);
            let cell = (bi * ANGLES_DEG.len() + ai) as u64;

            for k in 0..GRID_PHASES {
                let d = d0 + rate * k as f64 / GRID_PHASES as f64;
                let clean = render_steps(SIZE, n, &[(d, 1.0)], blur);
                for (ni, &noise) in GRID_NOISES_DN.iter().enumerate() {
                    hits.iter_mut().for_each(Vec::clear);
                    let trials = if noise > 0.0 { GRID_TRIALS } else { 1 };
                    for trial in 0..trials {
                        let s = scene(
                            &clean,
                            noise,
                            seed([fam, cell, k as u64, ni as u64, trial]),
                            n,
                            d,
                        );
                        for (loc, h) in locators.iter().zip(hits.iter_mut()) {
                            (loc.locate)(&s, &mut tools, h);
                        }
                    }
                    for (o, h) in out.iter_mut().zip(&hits) {
                        let (mean, std) = mean_std(h);
                        o.bias[bi] = o.bias[bi].max(mean.abs());
                        o.sigma[bi] = o.sigma[bi].max(std);
                    }
                }
            }

            let mut hists = vec![[0u64; BINS]; locators.len()];
            for k in 0..DENSE_PHASES {
                let d = d0 + rate * k as f64 / DENSE_PHASES as f64;
                let clean = render_steps(SIZE, n, &[(d, 1.0)], blur);
                for trial in 0..DENSE_TRIALS {
                    let s = scene(
                        &clean,
                        DENSE_NOISE_DN,
                        seed([fam, cell, k as u64, 9, trial]),
                        n,
                        d,
                    );
                    for ((loc, h), hist) in
                        locators.iter().zip(hits.iter_mut()).zip(hists.iter_mut())
                    {
                        h.clear();
                        (loc.locate)(&s, &mut tools, h);
                        h.iter().for_each(|hit| hist[bin(hit.x)] += 1);
                    }
                }
            }
            for (o, hist) in out.iter_mut().zip(&hists) {
                o.locking[bi] = o.locking[bi].max(locking_index(hist));
            }
        }
    }
    out
}

/// Envelopes measured 2026-10-16 (`cargo test -p vision-metrology --test accuracy
/// --all-features phase -- --nocapture`), |bias| and sigma pinned at ~1.5× the measured
/// worst cell and the locking index `L` at `1 + 1.5 (L − 1)`; the measured numbers are in
/// `docs/performance.md`. `edge1d_centroid` leaves a histogram bin empty (index ∞), so
/// only its bias and sigma are guarded; it is recorded in `docs/dev/backlog.md`
/// ("Edges"). `edge2d_parabolic_along_normal` re-measured 2026-10-16 after issue #99.
/// `strip_continuous_sigma2.5` pins its index at 1.2, the acceptance bar for a locator
/// that does not lock, rather than 1.5× its measured 1.19.
const LOCATORS: &[Locator] = &[
    Locator {
        name: "strip_textbook_parabolic",
        family: Family::Scan,
        locate: strip_parabolic,
        bias_envelope: 0.15,
        sigma_envelope: 0.12,
        locking_envelope: 5.9,
    },
    Locator {
        name: "strip_textbook_gaussian",
        family: Family::Scan,
        locate: strip_gaussian,
        bias_envelope: 0.17,
        sigma_envelope: 0.12,
        locking_envelope: 5.0,
    },
    Locator {
        name: "strip_textbook_midpoint",
        family: Family::Scan,
        locate: strip_midpoint,
        bias_envelope: 0.08,
        sigma_envelope: 0.094,
        locking_envelope: 2.25,
    },
    Locator {
        name: "strip_textbook_half_contrast",
        family: Family::Scan,
        locate: strip_half_contrast,
        bias_envelope: 0.084,
        sigma_envelope: 0.09,
        locking_envelope: 2.2,
    },
    Locator {
        name: "strip_default",
        family: Family::Scan,
        locate: strip_default,
        bias_envelope: 0.097,
        sigma_envelope: 0.095,
        locking_envelope: 2.2,
    },
    Locator {
        name: "strip_default_sigma2.5",
        family: Family::Scan,
        locate: strip_default_sigma25,
        bias_envelope: 0.059,
        sigma_envelope: 0.076,
        locking_envelope: 1.7,
    },
    Locator {
        name: "strip_continuous_sigma1",
        family: Family::Scan,
        locate: strip_continuous_sigma10,
        bias_envelope: 0.12,
        sigma_envelope: 0.056,
        locking_envelope: 3.5,
    },
    Locator {
        name: "strip_continuous_sigma1.5",
        family: Family::Scan,
        locate: strip_continuous_sigma15,
        bias_envelope: 0.048,
        sigma_envelope: 0.031,
        locking_envelope: 1.71,
    },
    Locator {
        name: "strip_continuous_sigma2.5",
        family: Family::Scan,
        locate: strip_continuous_sigma25,
        bias_envelope: 0.018,
        sigma_envelope: 0.019,
        locking_envelope: 1.2,
    },
    Locator {
        name: "caliper_rect_default",
        family: Family::Scan,
        locate: rect_default,
        bias_envelope: 0.105,
        sigma_envelope: 0.029,
        locking_envelope: 2.65,
    },
    Locator {
        name: "edge1d_parabolic",
        family: Family::Axis,
        locate: edge1d_parabolic,
        bias_envelope: 0.048,
        sigma_envelope: 0.086,
        locking_envelope: 1.75,
    },
    Locator {
        name: "edge1d_gaussian",
        family: Family::Axis,
        locate: edge1d_gaussian,
        bias_envelope: 0.065,
        sigma_envelope: 0.078,
        locking_envelope: 2.05,
    },
    Locator {
        name: "edge1d_centroid",
        family: Family::Axis,
        locate: edge1d_centroid,
        bias_envelope: 0.27,
        sigma_envelope: 0.28,
        locking_envelope: f64::INFINITY,
    },
    Locator {
        name: "edge2d_parabolic_along_normal",
        family: Family::Axis,
        locate: edge2d_parabolic,
        bias_envelope: 0.046,
        sigma_envelope: 0.043,
        locking_envelope: 1.74,
    },
];

#[test]
fn phase_sweep_envelopes_hold() {
    let mut failures = Vec::new();
    for family in [Family::Scan, Family::Axis] {
        let locators: Vec<&Locator> = LOCATORS.iter().filter(|l| l.family == family).collect();
        for (loc, m) in locators.iter().zip(sweep(family, &locators)) {
            let (bias, sigma, locking) = (worst(&m.bias), worst(&m.sigma), worst(&m.locking));
            eprintln!(
                "{}: bias={bias:.4} sigma={sigma:.4} locking={locking:.3} \
                 (envelope {:.4}/{:.4}/{:.3}); per blur {BLURS:?}: bias {:.4?} sigma {:.4?} \
                 locking {:.3?}",
                loc.name,
                loc.bias_envelope,
                loc.sigma_envelope,
                loc.locking_envelope,
                m.bias,
                m.sigma,
                m.locking
            );
            if bias > loc.bias_envelope
                || sigma > loc.sigma_envelope
                || locking > loc.locking_envelope
            {
                failures.push(loc.name);
            }
        }
    }
    assert!(failures.is_empty(), "outside their envelopes: {failures:?}");
}

#[test]
fn the_locking_index_separates_locked_from_uniform() {
    // Uniform positions: every bin equal.
    let mut uniform = [0u64; BINS];
    (0..1000).for_each(|k| uniform[bin((f64::from(k) + 0.5) / 1000.0 + 7.0)] += 1);
    assert_eq!(locking_index(&uniform), 1.0);
    // A locator that halves every distance to the nearest pixel centre.
    let mut locked = [0u64; BINS];
    for k in 0..1000 {
        let truth = (f64::from(k) + 0.5) / 1000.0 + 7.0;
        locked[bin(truth.round() + 0.5 * (truth - truth.round()))] += 1;
    }
    assert!(locking_index(&locked) == f64::INFINITY);
    // Negative coordinates take their fractional part toward −∞.
    assert_eq!(bin(-0.25), 7);
}
