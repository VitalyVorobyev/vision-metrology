//! C1 — accuracy regression suite (`docs/roadmap.md`).
//!
//! Performance has benches; accuracy did not have a comparable harness before
//! this file. Each row below sweeps a synthetic fixture with a known
//! subpixel/geometric ground truth — antialiased via an analytic Gaussian-CDF
//! edge profile, per the `tests-synthetic-fixtures` skill — across the
//! sweep in `docs/roadmap.md`'s C1 table, and reports the *worst* bias and
//! standard deviation found anywhere in the grid. Envelopes were measured
//! once (2026-08-20, with `-- --nocapture` added to the usual `cargo test`
//! invocation to see each row's numbers) and pinned at roughly 1.5x that
//! measurement — the exact numbers are in each row's comment below, so a
//! regression shows up as a failing assertion instead of a silent drift.
//!
//! Adding an operator is one row in [`ROWS`]: write a `fn() -> Measured` that
//! runs its own sweep and returns the worst-case bias/sigma, add a row with
//! an envelope, done — `accuracy_envelopes_hold` drives every row the same
//! way.
//!
//! Noise is expressed in "LSB" (u8 grayscale steps): a seeded xorshift64*
//! (the workspace's no-external-`rand` convention, e.g.
//! `examples/gen_illustrations.rs`) adds uniform noise in `[-lsb, lsb]` to
//! each pixel before quantizing to `u8`.

use vision_metrology::fit::{FitConfig, RansacConfig, RobustLoss, fit_circle};
use vision_metrology::matching::{
    CropSpec, Refinement, ShapeMatcher, ShapeModel, ShapeModelBuilder, ShapeModelConfig,
    ShapeSearchConfig,
};
use vision_metrology::measure::{Caliper, MeasureConfig, MeasureRect, PolaritySelect};
use vision_metrology::warp::{Interp, Map};
use vision_metrology::{
    BorderMode, Edge1DConfig, Edge1DDetector, Edge2DConfig, Edge2DDetector, EdgePolarity, Image,
    Point2f, Rect2f, SmoothKind, SubpixRefine, Vec2f, wrap_angle,
};
use vm_primitives::{Hysteresis, Subpix2D};

// ── shared fixtures ──────────────────────────────────────────────────────

/// A tiny seeded xorshift64* — deterministic noise without an external RNG
/// dependency (invariant 12). Same construction as
/// `examples/gen_illustrations.rs`'s `Lcg`.
struct Lcg(u64);

impl Lcg {
    fn next_unit(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }

    /// Uniform noise in `[-amp, amp]`.
    fn signed(&mut self, amp: f32) -> f32 {
        (self.next_unit() * 2.0 - 1.0) * amp
    }
}

/// Abramowitz & Stegun 7.1.26 (max error 1.5e-7) — plenty for an 8-bit fixture.
fn erf(x: f32) -> f32 {
    let sign = x.signum();
    let x = x.abs();
    let (a1, a2, a3, a4, a5, p): (f32, f32, f32, f32, f32, f32) = (
        0.254_829_6,
        -0.284_496_74,
        1.421_413_7,
        -1.453_152,
        1.061_405_4,
        0.327_591_1,
    );
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Standard normal CDF, used to rasterise a step edge blurred by Gaussian
/// `sigma`: `I(d) = dark + (bright - dark) * gaussian_cdf(d / sigma)`, `d`
/// the signed distance from the true edge. This is the continuous limit of
/// the pillbox/optics blur a real edge shows, so the ground-truth subpixel
/// position is exact regardless of `sigma`.
fn gaussian_cdf(x: f32) -> f32 {
    0.5 * (1.0 + erf(x * std::f32::consts::FRAC_1_SQRT_2))
}

const DARK: f32 = 20.0;
const BRIGHT: f32 = 200.0;

fn mean_std(v: &[f32]) -> (f32, f32) {
    let n = v.len() as f32;
    let mean = v.iter().sum::<f32>() / n;
    let var = v.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// Bias and spread of one operator's worst sweep cell, in that operator's own
/// units (pixels for every row here, degrees for the rotation row).
#[derive(Debug, Clone, Copy)]
struct Measured {
    bias: f32,
    sigma: f32,
}

impl Measured {
    fn worst_of(cells: impl Iterator<Item = (f32, f32)>) -> Self {
        let mut bias = 0.0f32;
        let mut sigma = 0.0f32;
        for (b, s) in cells {
            bias = bias.max(b.abs());
            sigma = sigma.max(s);
        }
        Self { bias, sigma }
    }
}

// ── Edge1D ───────────────────────────────────────────────────────────────

/// 64-sample antialiased step at `edge_x`, blurred by `sigma`, `lsb` uniform
/// noise added per sample (seeded by `trial`), quantised to `u8`.
fn edge1d_signal(edge_x: f32, sigma: f32, lsb: f32, trial: u64) -> Vec<u8> {
    let mut rng = Lcg(0x9E37_79B9_7F4A_7C15 ^ trial.wrapping_mul(0x2545_F491));
    (0..64)
        .map(|i| {
            let d = i as f32 - edge_x;
            let v = DARK + (BRIGHT - DARK) * gaussian_cdf(d / sigma) + rng.signed(lsb);
            v.round().clamp(0.0, 255.0) as u8
        })
        .collect()
}

fn edge1d_sweep() -> Measured {
    let sigmas = [0.5f32, 1.0, 2.0, 3.0];
    let noises = [0.0f32, 2.0, 5.0];
    let offsets = [0.0f32, 0.25, 0.5, 0.75];
    const TRIALS_NOISY: u64 = 25;

    let mut det = Edge1DDetector::new(1.0);
    let mut cells = Vec::new();

    for &sigma in &sigmas {
        for &lsb in &noises {
            for &offset in &offsets {
                let edge_x = 32.0 + offset;
                let trials = if lsb > 0.0 { TRIALS_NOISY } else { 1 };
                let mut errs = Vec::with_capacity(trials as usize);
                for trial in 0..trials {
                    let signal = edge1d_signal(edge_x, sigma, lsb, trial);
                    let cfg = Edge1DConfig {
                        sigma,
                        refine: SubpixRefine::Parabolic3,
                        ..Edge1DConfig::default()
                    };
                    let peaks = det.detect_in(&signal, &cfg);
                    let peak = peaks
                        .iter()
                        .filter(|p| p.polarity == EdgePolarity::Rising)
                        .max_by(|a, b| a.strength.total_cmp(&b.strength))
                        .unwrap_or_else(|| {
                            panic!("no rising edge at sigma={sigma} lsb={lsb} offset={offset}")
                        });
                    errs.push(peak.x - edge_x);
                }
                cells.push(mean_std(&errs));
            }
        }
    }
    Measured::worst_of(cells.into_iter())
}

// ── Edge2D and Caliper share an antialiased 2-D straight-edge fixture ────

/// `size x size` antialiased straight edge: the true line is
/// `dot(n, p - center) = offset`, `n = (cos(angle), sin(angle))` pointing
/// dark-to-bright, blurred by `sigma`, `lsb` uniform noise per pixel.
fn straight_edge_image(
    size: usize,
    angle: f32,
    offset: f32,
    sigma: f32,
    lsb: f32,
    trial: u64,
) -> (Image<u8>, Vec2f, Point2f) {
    let (cs, sn) = (angle.cos(), angle.sin());
    let n = Vec2f::new(cs, sn);
    let center = Point2f::new(size as f32 / 2.0, size as f32 / 2.0);
    let mut rng = Lcg(0xC2B2_AE3D_27D4_EB4F ^ trial.wrapping_mul(0x9E37_79B9));

    let mut data = vec![0u8; size * size];
    for y in 0..size {
        for x in 0..size {
            let d = n.x * (x as f32 - center.x) + n.y * (y as f32 - center.y) - offset;
            let v = DARK + (BRIGHT - DARK) * gaussian_cdf(d / sigma) + rng.signed(lsb);
            data[y * size + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    (
        Image::from_vec(size, size, data).expect("valid image"),
        n,
        center,
    )
}

const EDGE2D_SIZE: usize = 96;
/// Edgels within this margin of the border are dropped — `BorderMode::Clamp`
/// and NMS both distort the last couple of rows/columns.
const EDGE2D_MARGIN: f32 = 16.0;

fn edge2d_sweep() -> Measured {
    let angles_deg = [0.0f32, 30.0, 45.0, 60.0, 90.0];
    let sigmas = [0.5f32, 1.0, 2.0, 3.0];
    let noises = [0.0f32, 2.0, 5.0];
    const TRIALS_NOISY: u64 = 5;

    let mut det = Edge2DDetector::new();
    let mut cells = Vec::new();

    for &angle_deg in &angles_deg {
        let angle = angle_deg.to_radians();
        for &sigma in &sigmas {
            // `Hysteresis::Auto` derives thresholds from *this frame's* peak
            // response — at a heavy blur + high noise combination the true
            // edge's own peak is weak enough that noise-driven local maxima
            // elsewhere in the (otherwise flat) image cross the low
            // threshold and hysteresis-chain across the whole frame. A real
            // system characterises its scene contrast once and holds the
            // threshold fixed rather than re-deriving it from a noisy frame,
            // so measure the clean peak at this sigma once and reuse it —
            // representative *and* avoids that failure mode.
            let (clean, _, _) = straight_edge_image(EDGE2D_SIZE, angle, 0.0, sigma, 0.0, 0);
            let (_, clean_grads) = det.detect_with_gradients(
                &clean.as_view(),
                &Edge2DConfig {
                    smooth_kind: SmoothKind::None,
                    ..Edge2DConfig::default()
                },
            );
            let clean_peak = clean_grads.mag.iter().copied().fold(0.0f32, f32::max);
            let hysteresis = Hysteresis::Manual {
                low: 0.2 * clean_peak,
                high: 0.4 * clean_peak,
            };

            for &lsb in &noises {
                let trials = if lsb > 0.0 { TRIALS_NOISY } else { 1 };
                let mut errs = Vec::new();
                for trial in 0..trials {
                    let (img, n, center) =
                        straight_edge_image(EDGE2D_SIZE, angle, 0.0, sigma, lsb, trial);
                    let cfg = Edge2DConfig {
                        smooth_kind: SmoothKind::None,
                        hysteresis,
                        subpix: Subpix2D::ParabolicAlongNormal,
                        ..Edge2DConfig::default()
                    };
                    let edgels = det.detect(&img.as_view(), &cfg);
                    let lo = EDGE2D_MARGIN;
                    let hi = EDGE2D_SIZE as f32 - EDGE2D_MARGIN;
                    for e in &edgels {
                        if e.p.x < lo || e.p.x > hi || e.p.y < lo || e.p.y > hi {
                            continue;
                        }
                        let err = n.x * (e.p.x - center.x) + n.y * (e.p.y - center.y);
                        errs.push(err);
                    }
                }
                assert!(
                    !errs.is_empty(),
                    "no interior edgels at angle={angle_deg} sigma={sigma} lsb={lsb}"
                );
                cells.push(mean_std(&errs));
            }
        }
    }
    Measured::worst_of(cells.into_iter())
}

// ── Caliper (measure) ────────────────────────────────────────────────────

fn caliper_sweep() -> Measured {
    let angles_deg = [0.0f32, 30.0, 45.0, 60.0, 90.0];
    let sigmas = [0.5f32, 1.0, 2.0, 3.0];
    let noises = [0.0f32, 2.0, 5.0];
    let half_widths = [1.0f32, 3.0, 6.0, 10.0];
    let offsets = [0.0f32, 0.25, 0.5, 0.75];
    const TRIALS_NOISY: u64 = 15;

    let mut cells = Vec::new();

    for &angle_deg in &angles_deg {
        let angle = angle_deg.to_radians();
        for &sigma in &sigmas {
            for &half_width in &half_widths {
                for &lsb in &noises {
                    let trials = if lsb > 0.0 { TRIALS_NOISY } else { 1 };
                    let mut errs = Vec::with_capacity(trials as usize * offsets.len());
                    for &offset in &offsets {
                        for trial in 0..trials {
                            let (img, _n, center) =
                                straight_edge_image(EDGE2D_SIZE, angle, offset, sigma, lsb, trial);
                            let rect = MeasureRect {
                                center,
                                angle,
                                half_len: 15.0,
                                half_width,
                            };
                            let cfg = MeasureConfig {
                                sigma,
                                polarity: PolaritySelect::Rising,
                                ..MeasureConfig::default()
                            };
                            let mut cal = Caliper::rect(rect, cfg);
                            let edges = cal.measure(&img.as_view()).unwrap_or_else(|r| {
                                panic!(
                                    "caliper rejected at angle={angle_deg} sigma={sigma} \
                                     half_width={half_width} lsb={lsb} offset={offset}: {r:?}"
                                )
                            });
                            errs.push(edges[0].t - offset);
                        }
                    }
                    cells.push(mean_std(&errs));
                }
            }
        }
    }
    Measured::worst_of(cells.into_iter())
}

// ── fit_circle ───────────────────────────────────────────────────────────

/// `n` points around a circle of `radius`, covering `extent_deg` degrees
/// starting at 0, jittered radially by `noise_px` and with `outlier_frac`
/// of the points replaced by points well off the circle.
fn circle_points(
    center: Point2f,
    radius: f32,
    extent_deg: f32,
    noise_px: f32,
    outlier_frac: f32,
    trial: u64,
) -> Vec<Point2f> {
    const N: usize = 40;
    let mut rng = Lcg(0x1234_5678_9ABC_DEF0 ^ trial.wrapping_mul(0xA24B_AED4));
    let extent = extent_deg.to_radians();
    let n_outliers = ((N as f32) * outlier_frac).round() as usize;

    (0..N)
        .map(|i| {
            let phi = if N > 1 {
                i as f32 / (N - 1) as f32 * extent
            } else {
                0.0
            };
            if i < n_outliers {
                // Gross outlier: 8-15 px off the circle, not a near-miss.
                let off = 8.0 + rng.next_unit() * 7.0;
                let dir = rng.next_unit() * std::f32::consts::TAU;
                center
                    + Vec2f::new(phi.cos(), phi.sin()) * radius
                    + Vec2f::new(dir.cos(), dir.sin()) * off
            } else {
                let r = radius + rng.signed(noise_px);
                center + Vec2f::new(phi.cos(), phi.sin()) * r
            }
        })
        .collect()
}

fn fit_circle_sweep() -> Measured {
    let extents_deg = [30.0f32, 90.0, 180.0, 360.0];
    let noises_px = [0.0f32, 0.05, 0.15];
    let outlier_fracs = [0.0f32, 0.10];
    const TRIALS: u64 = 20;

    let truth = Point2f::new(150.0, 120.0);
    let radius = 60.0f32;
    // RANSAC finds an outlier-free consensus set before the Tukey-weighted
    // geometric refinement polishes it. Without RANSAC, a 30 deg arc with
    // 10% gross outliers among only 40 points corrupts the algebraic (Taubin)
    // initial fit badly enough that Tukey's own annealing (which starts from
    // *that* fit's residuals) down-weights every point — a real degenerate
    // case, not a test artifact.
    //
    // A 30 deg arc is nearly a straight chord, and near-collinear points are
    // famously ill-posed for circle fitting: circles of very different
    // radius/centre can all pass within a small tolerance of most of the
    // same near-straight point cloud, so RANSAC's own consensus metric
    // becomes ambiguous — a tighter `inlier_tol` (0.3 px, tuned to this
    // sweep's noise scale rather than the 1.0 px generic default) and a
    // higher `min_inliers` bar cut most of that down, but not all of it: a
    // couple of trials at extent=30 deg + 10% outliers still land a
    // catastrophically wrong consensus (multi-pixel radius error). That is
    // this operator's genuine worst case, not a bug, and it is exactly why
    // `min_inliers` exists — see the envelope note on the row below.
    let cfg = FitConfig {
        loss: RobustLoss::Tukey { c: 3.0 },
        ransac: Some(RansacConfig {
            iters: 1000,
            inlier_tol: 0.3,
            min_inliers: 20,
            ..RansacConfig::default()
        }),
        ..FitConfig::default()
    };

    let mut cells = Vec::new();
    for &extent in &extents_deg {
        for &noise in &noises_px {
            for &outlier_frac in &outlier_fracs {
                let trials = if noise > 0.0 || outlier_frac > 0.0 {
                    TRIALS
                } else {
                    1
                };
                let mut radius_errs = Vec::with_capacity(trials as usize);
                let mut center_errs = Vec::with_capacity(trials as usize);
                for trial in 0..trials {
                    let pts = circle_points(truth, radius, extent, noise, outlier_frac, trial);
                    let fit = fit_circle(&pts, &cfg).unwrap_or_else(|e| {
                        panic!("fit_circle failed at extent={extent} noise={noise} outlier_frac={outlier_frac}: {e}")
                    });
                    radius_errs.push(fit.model.radius - radius);
                    let dc = fit.model.center - truth;
                    center_errs.push((dc.x * dc.x + dc.y * dc.y).sqrt());
                }
                let (radius_bias, _) = mean_std(&radius_errs);
                let (_, center_sigma) = mean_std(&center_errs);
                cells.push((radius_bias, center_sigma));
            }
        }
    }
    Measured::worst_of(cells.into_iter())
}

// ── ShapeMatcher: a synthetic L-shape (mirrors tests/shape_matching.rs's ──
// SDF + smoothstep renderer, kept local rather than shared across the two
// separate test binaries — see that file for the fuller shape zoo). ───────

const SM_SIZE: usize = 320;

fn sdf_box(p: (f32, f32), half: (f32, f32)) -> f32 {
    let dx = p.0.abs() - half.0;
    let dy = p.1.abs() - half.1;
    let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt();
    outside + dx.max(dy).min(0.0)
}

/// An L-shape: asymmetric, with straight edges and corners, so both
/// translation and rotation are well constrained.
fn sdf_l_shape(p: (f32, f32)) -> f32 {
    let a = sdf_box((p.0, p.1 + 25.0), (35.0, 10.0));
    let b = sdf_box((p.0 + 25.0, p.1), (10.0, 35.0));
    a.min(b)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `l_shape_at` at an explicit true `scale`: the L-shape's SDF is evaluated
/// in *model* units (`(mx, my) / scale`), then the returned distance is
/// scaled back up by `scale` before the antialiasing smoothstep. That keeps
/// the rendered edge's blur width at a fixed ~1 image pixel regardless of
/// `scale` — matching a camera's own blur, which does not shrink just
/// because the part in view got smaller — rather than the edge sharpening
/// or softening as an artifact of the fixture.
fn l_shape_at_scaled(cx: f32, cy: f32, angle: f32, scale: f32) -> Image<u8> {
    let (sn, cs) = angle.sin_cos();
    let mut data = vec![0u8; SM_SIZE * SM_SIZE];
    for y in 0..SM_SIZE {
        for x in 0..SM_SIZE {
            let dx = x as f32 - cx;
            let dy = y as f32 - cy;
            let mx = (cs * dx + sn * dy) / scale;
            let my = (-sn * dx + cs * dy) / scale;
            let d_image = sdf_l_shape((mx, my)) * scale;
            let t = smoothstep(-1.0, 1.0, -d_image);
            data[y * SM_SIZE + x] = (DARK + (BRIGHT - DARK) * t).round().clamp(0.0, 255.0) as u8;
        }
    }
    Image::from_vec(SM_SIZE, SM_SIZE, data).expect("valid image")
}

fn l_shape_at(cx: f32, cy: f32, angle: f32) -> Image<u8> {
    l_shape_at_scaled(cx, cy, angle, 1.0)
}

fn l_shape_roi() -> Rect2f {
    Rect2f {
        x: 90.0,
        y: 90.0,
        width: 140.0,
        height: 140.0,
    }
}

fn build_l_shape_model() -> ShapeModel {
    let reference = l_shape_at(160.0, 160.0, 0.0);
    ShapeModelBuilder::new()
        .build(
            &reference.as_view(),
            l_shape_roi(),
            &ShapeModelConfig::default(),
        )
        .expect("L-shape model builds")
}

/// Where the model's reference point (the point-cloud centroid, offset `o`
/// from the shape's own origin at `(160, 160)`) lands under pose `(cx, cy,
/// angle)` — mirrors `tests/shape_matching.rs`'s `expected_position`.
fn expected_position(model: &ShapeModel, cx: f32, cy: f32, angle: f32) -> Point2f {
    let o = Vec2f::new(model.origin().x - 160.0, model.origin().y - 160.0);
    let (sn, cs) = angle.sin_cos();
    Point2f::new(cx + (cs * o.x - sn * o.y), cy + (sn * o.x + cs * o.y))
}

fn find_one(scene: &Image<u8>, model: &ShapeModel, cfg: &ShapeSearchConfig) -> Point2f {
    ShapeMatcher::new()
        .find(&scene.as_view(), model, cfg)
        .into_iter()
        .next()
        .expect("model must be found in its own synthetic scene")
        .position
}

fn shape_matcher_translation_sweep() -> Measured {
    let model = build_l_shape_model();
    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };
    let base = (160.0f32, 160.0f32);
    let steps = [0.0f32, 0.25, 0.5, 0.75, 1.0];

    let mut errs = Vec::with_capacity(steps.len());
    for &dx in &steps {
        let (cx, cy) = (base.0 + dx, base.1);
        let scene = l_shape_at(cx, cy, 0.0);
        let position = find_one(&scene, &model, &cfg);
        let want = expected_position(&model, cx, cy, 0.0);
        errs.push(position.x - want.x);
    }
    let (bias, sigma) = mean_std(&errs);
    Measured { bias, sigma }
}

fn shape_matcher_rotation_sweep() -> Measured {
    let model = build_l_shape_model();
    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };

    let mut errs_deg = Vec::new();
    for deg in (0..360).step_by(30) {
        let angle = (deg as f32).to_radians();
        let scene = l_shape_at(160.0, 160.0, angle);
        let m = ShapeMatcher::new()
            .find(&scene.as_view(), &model, &cfg)
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("not found at {deg} deg"));
        let err = wrap_angle(m.angle() - wrap_angle(angle)).to_degrees();
        errs_deg.push(err);
    }
    let (bias, sigma) = mean_std(&errs_deg);
    Measured { bias, sigma }
}

// ── ShapeMatcher: scale sweep (roadmap C1 / warp-wave decision 9) ─────────
//
// The matcher's scale search is a discrete scan (roadmap plan, decision 9):
// the plan's working hypothesis was that it degrades well before the edges
// of a wide `scale_range`, and this sweep exists to measure that honestly
// rather than assume it. It deliberately does not touch any matching code —
// only measures it.
//
// The model is taught at scale 1.0 with `scale_range = (0.45, 2.1)`; scenes
// are rendered at 12 true scales geometrically spaced over [0.5, 2.0], each
// at 3 rotations. Per scale: found-rate (across the 3 rotations), and — only
// over the sub-range where the *pinned baseline* found every trial — scale
// bias/sigma and position bias/sigma.
//
// **Measured result (2026-08-20)**: on this clean, noise-free, clutter-free
// synthetic fixture, found-rate is 100% at *every* one of the 12 scales —
// the hypothesis that this operator degrades badly away from 1.0 does not
// hold here, given a model whose own `scale_range` was taught wide enough to
// match what is searched. Scale bias stays under 0.15% and position bias
// under 0.03 px across the whole 0.5-2.0x range; see the full per-scale
// table in `BASELINE_FOUND_RATE`'s and the two envelope rows' comments. This
// does not contradict the plan's field data (canend, which is noisy and
// cluttered) or its degradation hypothesis for a model taught at the
// *default* `scale_range = (1.0, 1.0)` — it isolates one variable: given a
// correctly-configured model, the discrete scan itself is not the bottleneck
// on a clean scene.

/// 12 scales geometrically spaced over `[lo, hi]` inclusive.
fn geometric_scales(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let t = i as f32 / (n - 1) as f32;
            lo * (hi / lo).powf(t)
        })
        .collect()
}

/// [`expected_position`], generalised to a nonzero true scale.
fn expected_position_scaled(
    model: &ShapeModel,
    cx: f32,
    cy: f32,
    angle: f32,
    scale: f32,
) -> Point2f {
    let o = Vec2f::new(model.origin().x - 160.0, model.origin().y - 160.0);
    let (sn, cs) = angle.sin_cos();
    Point2f::new(
        cx + scale * (cs * o.x - sn * o.y),
        cy + scale * (sn * o.x + cs * o.y),
    )
}

fn build_l_shape_model_scale_range() -> ShapeModel {
    let reference = l_shape_at_scaled(160.0, 160.0, 0.0, 1.0);
    ShapeModelBuilder::new()
        .build(
            &reference.as_view(),
            l_shape_roi(),
            &ShapeModelConfig {
                scale_range: (0.45, 2.1),
                ..ShapeModelConfig::default()
            },
        )
        .expect("scale-range L-shape model builds")
}

const N_SCALES: usize = 12;
const SCALE_ROTATIONS_DEG: [f32; 3] = [0.0, 120.0, 240.0];

/// Found-rate baseline, pinned from the 2026-08-20 measurement below. A run
/// that finds the model in *fewer* trials than this at any scale is a
/// regression — the found set must not shrink. Measured full table (12
/// scales, geometrically spaced 0.5..2.0, 3 rotations {0, 120, 240} deg
/// each; `cargo test -p vision-metrology --all-features --test accuracy --
/// --nocapture`, ~123s):
///
/// | scale  | found | scale bias | scale sigma | pos bias (px) | pos sigma (px) |
/// |-------:|:-----:|-----------:|-------------:|---------------:|-----------------:|
/// | 0.5000 | 3/3   | -0.0014    | 0.0000       | 0.0218          | 0.0012            |
/// | 0.5672 | 3/3   | -0.0014    | 0.0003       | 0.0151          | 0.0023            |
/// | 0.6433 | 3/3   | -0.0007    | 0.0001       | 0.0070          | 0.0039            |
/// | 0.7297 | 3/3   | -0.0006    | 0.0001       | 0.0083          | 0.0010            |
/// | 0.8278 | 3/3   | -0.0002    | 0.0002       | 0.0063          | 0.0028            |
/// | 0.9389 | 3/3   |  0.0000    | 0.0002       | 0.0020          | 0.0006            |
/// | 1.0650 | 3/3   |  0.0005    | 0.0002       | 0.0050          | 0.0014            |
/// | 1.2081 | 3/3   |  0.0006    | 0.0001       | 0.0053          | 0.0036            |
/// | 1.3704 | 3/3   |  0.0005    | 0.0001       | 0.0097          | 0.0001            |
/// | 1.5544 | 3/3   |  0.0007    | 0.0001       | 0.0058          | 0.0039            |
/// | 1.7632 | 3/3   |  0.0006    | 0.0002       | 0.0105          | 0.0002            |
/// | 2.0000 | 3/3   |  0.0006    | 0.0003       | 0.0131          | 0.0003            |
///
/// Found every trial at every scale — see the module-level comment above
/// for why that does not contradict the plan's field-data hypothesis.
const BASELINE_FOUND_RATE: [f32; N_SCALES] = [1.0; N_SCALES];

struct ScaleCell {
    scale_stat: (f32, f32),
    pos_stat: (f32, f32),
}

/// Runs the full scale x rotation grid once, asserting the found-rate
/// regression guard as a side effect, and returns per-scale bias/sigma for
/// both the recovered scale and the recovered position.
fn scale_sweep_cells() -> Vec<ScaleCell> {
    let model = build_l_shape_model_scale_range();
    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };
    let scales = geometric_scales(0.5, 2.0, N_SCALES);

    scales
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let mut scale_errs = Vec::with_capacity(SCALE_ROTATIONS_DEG.len());
            let mut pos_errs = Vec::with_capacity(SCALE_ROTATIONS_DEG.len());
            let mut found = 0usize;
            for &rot_deg in &SCALE_ROTATIONS_DEG {
                let angle = rot_deg.to_radians();
                let scene = l_shape_at_scaled(160.0, 160.0, angle, s);
                if let Some(m) = ShapeMatcher::new()
                    .find(&scene.as_view(), &model, &cfg)
                    .into_iter()
                    .next()
                {
                    found += 1;
                    scale_errs.push(m.scale() - s);
                    let want = expected_position_scaled(&model, 160.0, 160.0, angle, s);
                    let d = m.position - want;
                    pos_errs.push((d.x * d.x + d.y * d.y).sqrt());
                }
            }
            let found_rate = found as f32 / SCALE_ROTATIONS_DEG.len() as f32;
            let scale_stat = if scale_errs.is_empty() {
                (0.0, 0.0)
            } else {
                mean_std(&scale_errs)
            };
            let pos_stat = if pos_errs.is_empty() {
                (0.0, 0.0)
            } else {
                mean_std(&pos_errs)
            };
            eprintln!(
                "scale_sweep: scale={s:.4} found={found}/{} found_rate={found_rate:.2} \
                 scale_bias={:.4} scale_sigma={:.4} pos_bias={:.4} pos_sigma={:.4}",
                SCALE_ROTATIONS_DEG.len(),
                scale_stat.0,
                scale_stat.1,
                pos_stat.0,
                pos_stat.1
            );
            assert!(
                found_rate + 1e-6 >= BASELINE_FOUND_RATE[i],
                "scale sweep regression at scale idx {i} (s={s:.4}): found_rate {found_rate:.2} \
                 dropped below the pinned baseline {:.2} — the found set must not shrink",
                BASELINE_FOUND_RATE[i]
            );
            ScaleCell {
                scale_stat,
                pos_stat,
            }
        })
        .collect()
}

/// Scale bias/sigma, gated only over the scales the pinned baseline found
/// at every rotation — see the module-level comment on why the rest of the
/// grid is measured and recorded, not gated.
fn shape_matcher_scale_bias_sweep() -> Measured {
    let cells = scale_sweep_cells();
    Measured::worst_of(
        cells
            .iter()
            .zip(BASELINE_FOUND_RATE)
            .filter(|(_, baseline)| *baseline >= 1.0)
            .map(|(c, _)| c.scale_stat),
    )
}

/// Position bias/sigma (pixels), same gating as [`shape_matcher_scale_bias_sweep`].
fn shape_matcher_scale_position_sweep() -> Measured {
    let cells = scale_sweep_cells();
    Measured::worst_of(
        cells
            .iter()
            .zip(BASELINE_FOUND_RATE)
            .filter(|(_, baseline)| *baseline >= 1.0)
            .map(|(c, _)| c.pos_stat),
    )
}

// ── ShapeMatch::model_frame_map repeatability (roadmap C1 / rectify wave) ──
//
// The number that decides anomaly-pipeline viability: rectify the same
// object, found independently at K slightly different subpixel poses, and
// measure how consistent the resulting canonical crop is. If a taught model
// and identical scene, re-photographed with only sub-pixel jitter in
// position/rotation/scale, cannot rectify to (near-)identical crops, no
// anomaly model trained on those crops can tell object variation from
// rectification noise.
//
// `bias`/`sigma` here are **intensity units** (8-bit grayscale), not pixels:
// per destination pixel, across the K crops, `sigma` is the standard
// deviation and `bias` (against a direct model-frame render of the taught
// reference image, sampled with an identity pose — no search involved) is
// the mean absolute difference. Both are averaged only over pixels every one
// of the K crops *and* the reference agree are valid (`apply_with_mask`'s
// mask, intersected) — the whole point of the mask is to keep border-fill
// pixels out of exactly this kind of measurement.

const RECTIFY_K: usize = 12;

/// A `1x`-resolution crop of the L-shape's own ROI — model-frame coordinates
/// already line up with `l_shape_roi()` since the reference image *is* the
/// model frame.
fn rectify_crop_spec() -> CropSpec {
    CropSpec::new(l_shape_roi(), 1.0)
}

/// Ground truth: the model-frame rect sampled directly out of the taught
/// reference image via an identity `dst -> src` map — no matcher, no search,
/// just the pixels the model was built from.
fn rectify_reference_crop(spec: &CropSpec) -> (Vec<f32>, Vec<u8>) {
    let reference = l_shape_at(160.0, 160.0, 0.0);
    let (w, h) = spec.output_size();
    let (rx, ry) = (spec.rect.x, spec.rect.y);
    let map = Map::from_fn(w, h, move |x, y| (rx + x, ry + y));
    let mut dst = vec![0u8; w * h];
    let mut mask = vec![0u8; w * h];
    map.apply_with_mask(
        &reference.as_view(),
        &mut dst,
        &mut mask,
        Interp::Bilinear,
        BorderMode::Constant(0u8),
    )
    .expect("reference crop resamples");
    (dst.iter().map(|&v| v as f32).collect(), mask)
}

fn rectify_repeatability_sweep() -> Measured {
    let model = build_l_shape_model();
    let spec = rectify_crop_spec();
    let (w, h) = spec.output_size();
    let n = w * h;

    let (reference, ref_mask) = rectify_reference_crop(&spec);

    let cfg = ShapeSearchConfig {
        refinement: Refinement::LeastSquares,
        ..Default::default()
    };

    // Deterministic seed distinct from every other sweep's.
    let mut rng = Lcg(0x51F0_C1A5_9B2E_77AA);
    let mut crops: Vec<Vec<f32>> = Vec::with_capacity(RECTIFY_K);
    let mut valid_all = ref_mask;

    for _ in 0..RECTIFY_K {
        // Subpixel jitter: +-0.5 px translation, +-1 deg rotation, +-1% scale.
        let cx = 160.0 + rng.signed(0.5);
        let cy = 160.0 + rng.signed(0.5);
        let angle = rng.signed(1.0f32.to_radians());
        let scale = 1.0 + rng.signed(0.01);
        let scene = l_shape_at_scaled(cx, cy, angle, scale);

        let m = ShapeMatcher::new()
            .find(&scene.as_view(), &model, &cfg)
            .into_iter()
            .next()
            .expect("model must be found under subpixel jitter");

        let map = m.model_frame_map(&spec);
        let mut dst = vec![0u8; n];
        let mut mask = vec![0u8; n];
        map.apply_with_mask(
            &scene.as_view(),
            &mut dst,
            &mut mask,
            Interp::Bilinear,
            BorderMode::Constant(0u8),
        )
        .expect("crop resamples");

        for i in 0..n {
            if mask[i] == 0 {
                valid_all[i] = 0;
            }
        }
        crops.push(dst.iter().map(|&v| v as f32).collect());
    }

    let mut sigma_sum = 0.0f32;
    let mut diff_sum = 0.0f32;
    let mut count = 0usize;
    for i in 0..n {
        if valid_all[i] == 0 {
            continue;
        }
        let vals: [f32; RECTIFY_K] = std::array::from_fn(|k| crops[k][i]);
        let mean = vals.iter().sum::<f32>() / RECTIFY_K as f32;
        let var = vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / RECTIFY_K as f32;
        sigma_sum += var.sqrt();
        diff_sum += (mean - reference[i]).abs();
        count += 1;
    }
    assert!(
        count > 0,
        "no pixel is valid across every rectified crop and the reference"
    );
    Measured {
        bias: diff_sum / count as f32,
        sigma: sigma_sum / count as f32,
    }
}

// ── the table ────────────────────────────────────────────────────────────

struct Row {
    name: &'static str,
    measure: fn() -> Measured,
    /// Bound on `|bias|`, same units as `measure`'s output.
    bias_envelope: f32,
    /// Bound on `sigma`, same units as `measure`'s output.
    sigma_envelope: f32,
}

/// Envelopes measured 2026-08-20 (`cargo test -p vision-metrology --test
/// accuracy --all-features -- --nocapture`), pinned at ~1.5x the worst cell
/// observed across each sweep — except `fit_circle`, whose worst cell is a
/// known-hard near-degenerate case (see `fit_circle_sweep`'s comment) and
/// whose envelope is intentionally looser than 1.5x:
///
/// | Row                                 | measured \|bias\| | measured sigma | envelope bias | envelope sigma |
/// |--------------------------------------|-------------------:|----------------:|---------------:|-----------------:|
/// | edge1d_position (px)                 | 0.042               | 0.095            | 0.07            | 0.15              |
/// | edge2d_position (px)                 | 0.085               | 1.195            | 0.13            | 1.80              |
/// | caliper_rect_position (px)           | 0.026               | 0.062            | 0.05            | 0.12              |
/// | fit_circle_radius_and_centre (px)    | 0.325               | 1.135            | 0.50            | 1.75              |
/// | shape_matcher_translation (px)       | 0.003               | 0.011            | 0.02            | 0.03              |
/// | shape_matcher_rotation (deg)         | 0.000               | 0.001            | 0.05            | 0.05              |
/// | shape_matcher_scale_bias (frac.)     | 0.0014              | 0.0003           | 0.003           | 0.001             |
/// | shape_matcher_scale_position (px)    | 0.0218              | 0.0039           | 0.04            | 0.01              |
/// | rectify_repeatability (8-bit units)  | 0.8775              | 1.6883           | 1.30            | 2.50              |
///
/// Rows 5-6 are the C1 scale sweep (roadmap Track C1 / warp-wave decision
/// 9): 12 true scales geometrically spaced 0.5..2.0x, 3 rotations each,
/// model taught at scale 1.0 with `scale_range = (0.45, 2.1)`. Every scale
/// found every rotation (100% found-rate — see `BASELINE_FOUND_RATE`'s full
/// per-scale table, which is also the found-rate regression guard), so
/// unlike every other row here the envelope covers the *entire* swept range,
/// not just a well-behaved sub-range.
///
/// The last row is the rectify-wave C1 addition: `ShapeMatch::model_frame_map`
/// repeatability, in 8-bit grayscale intensity units (not pixels — see the
/// module comment above `RECTIFY_K`). This is the number that decides
/// anomaly-pipeline viability — under 1% of full 8-bit range (bias 0.88/255,
/// sigma 1.69/255) for sub-pixel pose jitter, which is small enough that a
/// downstream anomaly model's own noise floor should dominate over
/// rectification noise.
///
/// `edge2d_position` and `fit_circle_radius_and_centre` are the two rows
/// whose worst cell is a genuinely hard corner of the sweep, not a loose
/// envelope for its own sake: `edge2d`'s worst cell is blur sigma=3 combined
/// with 5 LSB noise, where the true edge's own gradient peak is weak enough
/// that per-pixel noise competes with it (see `edge2d_sweep`'s comment on why
/// `Hysteresis::Auto` cannot be used here at all — it would be *worse*, not
/// better); `fit_circle`'s worst cell is a 30 deg arc (nearly a straight
/// chord) with 10% gross outliers, where RANSAC's own consensus metric
/// becomes ambiguous among near-collinear points.
const ROWS: &[Row] = &[
    Row {
        name: "edge1d_position",
        measure: edge1d_sweep,
        bias_envelope: 0.07,
        sigma_envelope: 0.15,
    },
    Row {
        name: "edge2d_position",
        measure: edge2d_sweep,
        bias_envelope: 0.13,
        sigma_envelope: 1.80,
    },
    Row {
        name: "caliper_rect_position",
        measure: caliper_sweep,
        bias_envelope: 0.05,
        sigma_envelope: 0.12,
    },
    Row {
        name: "fit_circle_radius_and_centre",
        measure: fit_circle_sweep,
        // Loose relative to the other rows: the worst cell in this grid is
        // extent=30 deg combined with 10% outliers, a near-degenerate arc
        // where RANSAC's own consensus metric is ambiguous (see the comment
        // on `fit_circle_sweep`). At extent >= 90 deg this operator is well
        // inside 0.05 px per B1's noise-free finding.
        bias_envelope: 0.50,
        sigma_envelope: 1.75,
    },
    Row {
        name: "shape_matcher_translation",
        measure: shape_matcher_translation_sweep,
        bias_envelope: 0.02,
        sigma_envelope: 0.03,
    },
    Row {
        name: "shape_matcher_rotation",
        measure: shape_matcher_rotation_sweep,
        bias_envelope: 0.05,
        sigma_envelope: 0.05,
    },
    Row {
        // Worst measured |bias|/sigma over the full 0.5-2.0x sweep (all 12
        // scales found every trial — see `BASELINE_FOUND_RATE`'s table):
        // |bias| 0.0014, sigma 0.0003 (dimensionless, a fraction of true
        // scale). ~1.5x that measurement, rounded.
        name: "shape_matcher_scale_bias",
        measure: shape_matcher_scale_bias_sweep,
        bias_envelope: 0.003,
        sigma_envelope: 0.001,
    },
    Row {
        // Same sweep, position (px): worst measured |bias| 0.0218, sigma
        // 0.0039. ~1.5-2x that measurement, rounded.
        name: "shape_matcher_scale_position",
        measure: shape_matcher_scale_position_sweep,
        bias_envelope: 0.04,
        sigma_envelope: 0.01,
    },
    Row {
        // Rectify wave, roadmap C1: intensity units (8-bit grayscale), see
        // the module comment above `RECTIFY_K` for what bias/sigma mean here
        // and why. Measured 2026-08-20: bias 0.8775, sigma 1.6883. ~1.5x,
        // rounded.
        name: "rectify_repeatability",
        measure: rectify_repeatability_sweep,
        bias_envelope: 1.3,
        sigma_envelope: 2.5,
    },
];

#[test]
fn accuracy_envelopes_hold() {
    for row in ROWS {
        let m = (row.measure)();
        eprintln!(
            "{}: bias={:.4} sigma={:.4} (envelope {:.4}/{:.4})",
            row.name, m.bias, m.sigma, row.bias_envelope, row.sigma_envelope
        );
        assert!(
            m.bias.abs() <= row.bias_envelope,
            "{}: |bias| {:.4} exceeds envelope {:.4}",
            row.name,
            m.bias.abs(),
            row.bias_envelope
        );
        assert!(
            m.sigma <= row.sigma_envelope,
            "{}: sigma {:.4} exceeds envelope {:.4}",
            row.name,
            m.sigma,
            row.sigma_envelope
        );
    }
}
