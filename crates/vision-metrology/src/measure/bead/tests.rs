//! Unit tests for the bead tracker's parts: stations, the fold guards, the banded solve
//! and its regulariser, IRLS, the pair gates, and config validation. The end-to-end
//! behaviour on rendered beads is in `tests/bead.rs`.

use std::f64::consts::TAU;

use nalgebra::{DMatrix, DVector};
use vm_primitives::{EdgePolarity, Error, Point2f};

use super::config::validate;
use super::curve::{
    V2, arc_lengths, chord_tangents, curvature, offset_window, perp, resample, step_scale,
};
use super::pairing::{Gates, choose, sort_edges};
use super::solve::{Band, Penalty, SolveScratch, solve_offsets};
use super::{BeadCaliper, BeadConfig, BeadPolarity, BeadReject, BeadTuning};
use crate::fit::RobustLoss;
use crate::measure::{Locate, MeasureEdge, RejectReason};

/// A seeded LCG in `[0, 1)`.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn resampled(src: &[V2], n: usize) -> Vec<V2> {
    let mut cum = Vec::new();
    arc_lengths(src, &mut cum);
    let mut out = Vec::new();
    resample(src, &cum, n, &mut out);
    out
}

fn max_dist(a: &[V2], b: &[V2]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(p, q)| (p[0] - q[0]).hypot(p[1] - q[1]))
        .fold(0.0, f64::max)
}

/// `n` points on a circle of radius `r` about the origin, `step` radians apart from
/// angle 0: a polygon with equal chords, turning clockwise on screen for `step > 0`.
fn polygon(r: f64, step: f64, n: usize) -> Vec<V2> {
    (0..n)
        .map(|i| {
            let a = i as f64 * step;
            [r * a.cos(), r * a.sin()]
        })
        .collect()
}

// ── stations ─────────────────────────────────────────────────────────────

#[test]
fn resampling_a_line_is_exact_and_skips_duplicates() {
    let want: Vec<V2> = (0..26).map(|i| [4.0 * i as f64, 0.0]).collect();
    let line = [[0.0, 0.0], [100.0, 0.0]];
    assert!(max_dist(&resampled(&line, 26), &want) < 1e-12);
    // Uneven, collinear and repeated vertices along the same line change nothing.
    let messy = [
        [0.0, 0.0],
        [0.0, 0.0],
        [13.7, 0.0],
        [30.0, 0.0],
        [30.0, 0.0],
        [30.0, 0.0],
        [99.1, 0.0],
        [100.0, 0.0],
        [100.0, 0.0],
    ];
    let got = resampled(&messy, 26);
    assert!(max_dist(&got, &want) < 1e-12, "{got:?}");
    // The ends are the input's ends exactly.
    assert_eq!(got[0], [0.0, 0.0]);
    assert_eq!(got[25], [100.0, 0.0]);
}

#[test]
fn resampling_uniform_stations_is_idempotent() {
    // Equal chords: already uniform, so resampling to the same count returns them.
    let stations = polygon(80.0, 0.05, 41);
    assert!(max_dist(&resampled(&stations, 41), &stations) < 1e-9);
    let line = resampled(&[[3.0, 4.0], [50.0, -7.0]], 33);
    assert!(max_dist(&resampled(&line, 33), &line) < 1e-9);
}

#[test]
fn chord_tangents_are_exact_on_a_circle() {
    let (r, step) = (100.0, 0.04);
    let pts = polygon(r, step, 61);
    let h = 2.0 * r * (0.5 * step).sin();
    let (window, mut tan) = (10.0, Vec::new());
    chord_tangents(&pts, h, window, &mut tan);
    let reach = (window / h).ceil() as usize;
    for (i, t) in tan.iter().enumerate().take(61 - reach).skip(reach) {
        let a = i as f64 * step;
        let truth = [-a.sin(), a.cos()];
        let err = (t[0] - truth[0]).hypot(t[1] - truth[1]);
        assert!(err < 1e-12, "station {i}: tangent off by {err:e}");
    }
    // Clockwise on screen turns towards +n, the centre: κ = +1/R.
    let mut kappa = Vec::new();
    curvature(&tan, h, &mut kappa);
    // κ reads the neighbours' tangents, so one station further in.
    for (i, k) in kappa.iter().enumerate().take(60 - reach).skip(reach + 1) {
        let rel = (k * r - 1.0).abs();
        assert!(rel < 1e-3, "station {i}: κ·R = {}", k * r);
    }
    let n = perp(tan[30]);
    let to_centre = [-pts[30][0] / r, -pts[30][1] / r];
    assert!((n[0] - to_centre[0]).abs() + (n[1] - to_centre[1]).abs() < 1e-12);
}

#[test]
fn a_degenerate_chord_takes_its_neighbours_tangent() {
    // Three stations at one point, then a line: the leading run borrows the first chord.
    let pts = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 2.0]];
    let mut tan = Vec::new();
    chord_tangents(&pts, 1.0, 0.25, &mut tan);
    for t in &tan {
        assert_eq!(*t, [0.0, 1.0]);
    }
}

#[test]
fn the_offset_window_stops_short_of_the_centre_of_curvature() {
    // Concave towards +n: the far side is clipped to 0.9·R, the near side keeps the reach.
    assert_eq!(offset_window(0.1, 15.0), (-15.0, 9.0));
    assert_eq!(offset_window(-0.1, 15.0), (-9.0, 15.0));
    assert_eq!(offset_window(0.01, 15.0), (-15.0, 15.0));
    assert_eq!(offset_window(0.0, 15.0), (-15.0, 15.0));
}

#[test]
fn the_step_scale_keeps_the_curve_from_folding() {
    // Stations on a circle of radius 20, normals towards the centre. Moving all of them
    // 30 px inwards would turn the curve inside out; the step stops at 0.9·R.
    let (r, step) = (20.0, 0.1);
    let pts = polygon(r, step, 21);
    let normals: Vec<V2> = pts.iter().map(|p| [-p[0] / r, -p[1] / r]).collect();
    let alpha = step_scale(&pts, &normals, &[30.0; 21]);
    assert!((alpha * 30.0 - 0.9 * r).abs() < 1e-9, "α = {alpha}");
    // Outwards, or by less than the margin, the full step is taken.
    assert_eq!(step_scale(&pts, &normals, &[-30.0; 21]), 1.0);
    assert_eq!(step_scale(&pts, &normals, &[1.0; 21]), 1.0);
}

// ── the banded solve ────────────────────────────────────────────────────

fn dense(b: &Band) -> DMatrix<f64> {
    let n = b.diag.len();
    DMatrix::from_fn(n, n, |r, c| match r.abs_diff(c) {
        0 => b.diag[r],
        1 => b.off1[r.min(c)],
        2 => b.off2[r.min(c)],
        _ => 0.0,
    })
}

/// `x` from the band solver and from nalgebra's dense Cholesky, `‖Δ‖ / ‖x‖`.
fn band_vs_dense(band: &mut Band, rhs: &[f64]) -> f64 {
    let want = dense(band)
        .cholesky()
        .expect("SPD")
        .solve(&DVector::from_column_slice(rhs));
    assert!(band.factor());
    let mut got = rhs.to_vec();
    band.solve_in_place(&mut got);
    let diff: f64 = got
        .iter()
        .zip(want.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();
    diff.sqrt() / want.norm()
}

#[test]
fn the_band_solver_matches_a_dense_cholesky() {
    let mut rng = Lcg(7);
    let mut band = Band::default();
    let mut worst = 0.0f64;
    for &n in &[1usize, 2, 3, 4, 5, 17, 64, 300] {
        for _ in 0..5 {
            // Diagonally dominant: symmetric positive definite.
            band.zero(n);
            for v in band.off1.iter_mut().chain(band.off2.iter_mut()) {
                *v = 2.0 * rng.next() - 1.0;
            }
            for d in &mut band.diag {
                *d = 4.5 + 4.0 * rng.next();
            }
            let rhs: Vec<f64> = (0..n).map(|_| 20.0 * rng.next() - 10.0).collect();
            worst = worst.max(band_vs_dense(&mut band, &rhs));

            // The tracker's own systems: data weights, damping, tension and bending.
            let w: Vec<f64> = (0..n).map(|_| 0.5 + 0.5 * rng.next()).collect();
            let p = Penalty::new(0.1 * rng.next(), 2.0, 8.0, 2.0 + 4.0 * rng.next());
            band.assemble(&w, p);
            worst = worst.max(band_vs_dense(&mut band, &rhs));
        }
    }
    assert!(worst < 1e-12, "band vs dense: relative error {worst:e}");
}

#[test]
fn a_non_positive_pivot_is_reported() {
    let mut band = Band::default();
    band.zero(3);
    band.diag.copy_from_slice(&[1.0, 1.0, 1.0]);
    band.off1.copy_from_slice(&[2.0, 0.0]);
    assert!(!band.factor());
}

/// The solved amplitude of `sin(ωs)` observed at every station of a long curve at
/// spacing `h`, read off the middle half by least squares.
fn passed_amplitude(h: f64, omega: f64, tuning: &BeadTuning) -> f64 {
    let n = (2000.0 / h) as usize + 1;
    let obs: Vec<f64> = (0..n).map(|i| (omega * i as f64 * h).sin()).collect();
    let valid = vec![true; n];
    let p = Penalty::new(
        f64::from(tuning.damping),
        f64::from(tuning.tension_px),
        f64::from(tuning.bending_px),
        h,
    );
    let mut s = SolveScratch::default();
    solve_offsets(&obs, &valid, p, RobustLoss::None, 1, &mut s).expect("solved");
    let (mut ss, mut sd) = (0.0, 0.0);
    for i in n / 4..3 * n / 4 {
        let basis = (omega * i as f64 * h).sin();
        ss += basis * basis;
        sd += basis * s.d[i];
    }
    sd / ss
}

#[test]
fn the_regulariser_passes_a_frequency_whatever_the_spacing() {
    let tuning = BeadTuning::default();
    let (l1, l2) = (f64::from(tuning.tension_px), f64::from(tuning.bending_px));
    for wavelength in [25.0, 50.0, 100.0, 300.0] {
        let omega = TAU / wavelength;
        let want = 1.0 / (1.0 + (l1 * omega).powi(2) + (l2 * omega).powi(4));
        // At fewer than about 10 stations per wavelength the differences stop
        // approximating the derivatives.
        for h in [1.0, 2.0, 4.0]
            .into_iter()
            .filter(|&h| wavelength >= 10.0 * h)
        {
            let got = passed_amplitude(h, omega, &tuning);
            let rel = (got - want).abs() / want;
            assert!(
                rel < 0.05,
                "wavelength {wavelength}, h {h}: passed {got:.4}, continuous {want:.4}"
            );
        }
    }
}

/// The correction at a station observed `outlier` px off a smooth bead, everything else
/// observing 0.5 px.
fn pulled_by_outlier(loss: RobustLoss) -> f64 {
    let n = 101;
    let mut obs = vec![0.5; n];
    obs[50] = 10.5;
    let mut s = SolveScratch::default();
    let p = Penalty::new(0.0, 2.0, 8.0, 4.0);
    solve_offsets(&obs, &vec![true; n], p, loss, 5, &mut s).expect("solved");
    (s.d[50] - 0.5).abs()
}

#[test]
fn irls_rejects_one_gross_outlier() {
    let plain = pulled_by_outlier(RobustLoss::None);
    let huber = pulled_by_outlier(RobustLoss::Huber { k: 1.0 });
    let tukey = pulled_by_outlier(RobustLoss::Tukey { c: 2.0 });
    eprintln!("pulled by a 10 px outlier: none {plain:.3}, huber {huber:.3}, tukey {tukey:.4}");
    assert!(plain > 0.5, "least squares follows the outlier: {plain}");
    assert!(huber < plain / 5.0, "Huber bounds it: {huber} vs {plain}");
    assert!(tukey < 0.01, "Tukey removes it: {tukey}");
}

// ── pairing ──────────────────────────────────────────────────────────────

const R: EdgePolarity = EdgePolarity::Rising;
const F: EdgePolarity = EdgePolarity::Falling;

fn edge(t: f32, amplitude: f32, polarity: EdgePolarity) -> MeasureEdge {
    MeasureEdge {
        p: Point2f::new(t, 0.0),
        t,
        amplitude,
        polarity,
    }
}

fn gates(polarity: BeadPolarity) -> Gates {
    Gates {
        polarity,
        min_width: 30.0,
        max_width: 80.0,
        max_offset: 15.0,
        clearance: None,
        min_margin: None,
    }
}

/// The chosen pair's `t`s, or the reason; the station at `t = 60`.
fn pick(edges: &[MeasureEdge], g: &Gates) -> Result<(f32, f32), BeadReject> {
    let mut e = edges.to_vec();
    sort_edges(&mut e);
    choose(&e, 60.0, (-15.0, 15.0), g).map(|h| (h.pair.first.t, h.pair.second.t))
}

#[test]
fn each_gate_names_itself() {
    let g = gates(BeadPolarity::Light);
    assert_eq!(
        pick(&[edge(70.0, 9.0, F), edge(90.0, 9.0, R)], &g),
        Err(BeadReject::NoPair)
    );
    assert_eq!(
        pick(&[edge(50.0, 9.0, R), edge(70.0, 9.0, F)], &g),
        Err(BeadReject::Width)
    );
    assert_eq!(
        pick(&[edge(70.0, 9.0, R), edge(110.0, 9.0, F)], &g),
        Err(BeadReject::Offset)
    );
    let crowded = [edge(37.0, 9.0, F), edge(40.0, 9.0, R), edge(80.0, 9.0, F)];
    assert_eq!(pick(&crowded, &g), Ok((40.0, 80.0)));
    let fenced = Gates {
        clearance: Some(5.0),
        ..g
    };
    assert_eq!(pick(&crowded, &fenced), Err(BeadReject::Clearance));
    // An edge further out than the clearance, or between the pair, is no objection.
    let clear = [
        edge(30.0, 9.0, F),
        edge(40.0, 9.0, R),
        edge(55.0, 3.0, R),
        edge(80.0, 9.0, F),
    ];
    assert_eq!(pick(&clear, &fenced), Ok((40.0, 80.0)));
}

#[test]
fn the_reason_is_the_gate_that_removed_the_last_pair() {
    let g = gates(BeadPolarity::Light);
    // One pair fails on width, the other gets as far as the offset window.
    let e = [edge(10.0, 9.0, R), edge(20.0, 9.0, F), edge(60.0, 9.0, F)];
    assert_eq!(pick(&e, &g), Err(BeadReject::Offset));
    // A pair that fails the window cannot be rescued by one that fails on width first.
    let e = [
        edge(100.0, 9.0, R),
        edge(110.0, 9.0, F),
        edge(140.0, 9.0, F),
    ];
    assert_eq!(pick(&e, &g), Err(BeadReject::Offset));
}

#[test]
fn the_best_pair_wins_and_ties_break_deterministically() {
    let g = gates(BeadPolarity::Light);
    // A highlight inside the bead: the outer pair is the only one of a valid width.
    let e = [
        edge(40.0, 9.0, R),
        edge(55.0, 4.0, R),
        edge(65.0, 4.0, F),
        edge(80.0, 9.0, F),
    ];
    assert_eq!(pick(&e, &g), Ok((40.0, 80.0)));
    // A stronger pair beats a centred one when the offset discount is small...
    let e = [edge(40.0, 5.0, R), edge(44.0, 9.0, R), edge(80.0, 9.0, F)];
    assert_eq!(pick(&e, &g), Ok((44.0, 80.0)));
    // ...and at equal amplitudes the discount picks the centred pair.
    let e = [edge(38.0, 9.0, R), edge(42.0, 9.0, R), edge(82.0, 9.0, F)];
    assert_eq!(pick(&e, &g), Ok((38.0, 82.0)));
    let e = [
        edge(40.0, 9.0, R),
        edge(50.0, 9.0, R),
        edge(70.0, 9.0, F),
        edge(80.0, 9.0, F),
    ];
    // (40, 70) and (50, 80) are both 30 wide, offsets −5 and +5: the earlier one wins.
    let narrow = Gates {
        max_width: 32.0,
        ..g
    };
    assert_eq!(pick(&e, &narrow), Ok((40.0, 70.0)));
    // Shuffled input gives the same answer.
    let shuffled = [e[3], e[1], e[0], e[2]];
    assert_eq!(pick(&shuffled, &narrow), Ok((40.0, 70.0)));
}

#[test]
fn a_close_runner_up_is_ambiguous_when_asked() {
    let g = Gates {
        max_width: 32.0,
        min_margin: Some(0.2),
        ..gates(BeadPolarity::Light)
    };
    let e = [
        edge(40.0, 9.0, R),
        edge(50.0, 9.0, R),
        edge(70.0, 9.0, F),
        edge(80.0, 9.0, F),
    ];
    assert_eq!(pick(&e, &g), Err(BeadReject::Ambiguous));
    // A clear winner keeps its margin, and its confidence says so.
    let e = [
        edge(40.0, 9.0, R),
        edge(50.0, 2.0, R),
        edge(70.0, 9.0, F),
        edge(80.0, 9.0, F),
    ];
    assert_eq!(pick(&e, &g), Ok((40.0, 70.0)));
    let mut sorted = e.to_vec();
    sort_edges(&mut sorted);
    let hit = choose(&sorted, 60.0, (-15.0, 15.0), &g).expect("a pair");
    assert!(
        (hit.confidence - (1.0 - 2.0 / 9.0)).abs() < 1e-6,
        "{}",
        hit.confidence
    );
}

#[test]
fn light_and_dark_are_mirror_images() {
    let flip = |e: &MeasureEdge| MeasureEdge {
        polarity: if e.polarity == R { F } else { R },
        ..*e
    };
    let cases: [&[MeasureEdge]; 3] = [
        &[
            edge(40.0, 9.0, R),
            edge(55.0, 4.0, R),
            edge(65.0, 4.0, F),
            edge(80.0, 9.0, F),
        ],
        &[edge(70.0, 9.0, F), edge(90.0, 9.0, R)],
        &[edge(37.0, 9.0, F), edge(40.0, 9.0, R), edge(80.0, 9.0, F)],
    ];
    for e in cases {
        let flipped: Vec<MeasureEdge> = e.iter().map(flip).collect();
        let light = pick(e, &gates(BeadPolarity::Light));
        assert_eq!(light, pick(&flipped, &gates(BeadPolarity::Dark)));
        assert_ne!(light, pick(e, &gates(BeadPolarity::Dark)));
    }
}

// ── config and reasons ──────────────────────────────────────────────────

#[test]
fn invalid_configs_are_rejected() {
    assert!(validate(&BeadConfig::default()).is_ok());
    let d = BeadConfig::default();
    let bad = [
        BeadConfig {
            min_width: 90.0,
            ..d
        },
        BeadConfig { spacing: 0.0, ..d },
        BeadConfig {
            spacing: f32::NAN,
            ..d
        },
        BeadConfig {
            min_margin: Some(1.0),
            ..d
        },
        BeadConfig {
            clearance: Some(-1.0),
            ..d
        },
        BeadConfig {
            track: BeadCaliper {
                locate: Locate::MidpointCrossing {
                    endpoint_samples: std::num::NonZeroUsize::MIN,
                    min_contrast: 1.0,
                },
                ..d.track
            },
            ..d
        },
        BeadConfig {
            measure: BeadCaliper {
                locate: Locate::HalfContrast {
                    flank_near_px: 3.0,
                    flank_far_px: 30.0,
                    tol_px: 0.01,
                    max_iter: std::num::NonZeroUsize::MIN,
                    min_contrast: 0.0,
                },
                ..d.measure
            },
            ..d
        },
        BeadConfig {
            track: BeadCaliper {
                max_offset: 0.0,
                ..d.track
            },
            ..d
        },
        BeadConfig {
            measure: BeadCaliper {
                profile: crate::measure::ProfileConfig {
                    sigma: 0.0,
                    ..d.measure.profile
                },
                ..d.measure
            },
            ..d
        },
        BeadConfig {
            tuning: BeadTuning {
                min_support: 1.5,
                ..d.tuning
            },
            ..d
        },
        BeadConfig {
            tuning: BeadTuning {
                loss: RobustLoss::Tukey { c: 0.0 },
                ..d.tuning
            },
            ..d
        },
    ];
    for (k, cfg) in bad.iter().enumerate() {
        assert!(
            matches!(validate(cfg), Err(Error::InvalidConfig(_))),
            "case {k} should be invalid"
        );
    }
}

#[test]
fn every_reason_has_its_own_slot_and_name() {
    let mut names = std::collections::HashSet::new();
    for (k, r) in BeadReject::ALL.iter().enumerate() {
        assert_eq!(r.index(), k);
        assert!(names.insert(r.as_str()), "{} twice", r.as_str());
    }
    assert_eq!(
        BeadReject::Caliper(RejectReason::OffImage).as_str(),
        "off_image"
    );
}
