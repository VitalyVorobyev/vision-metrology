//! `measure::BeadTracker` end to end, on rendered beads with continuous ground truth.
//!
//! Every scene comes from the ribbon fixture (`examples/common/ribbon.rs`): a light bead
//! 50 px wide, blurred by σ = 1.2 px, at 180 DN on a 40 DN background, along a line at 17°,
//! a clockwise arc of radius 150 px, or a sine of amplitude 12 px and period 160 px. The
//! truth is the fixture's: `nearest(p)` gives a point's arc length and signed distance from
//! the true centreline, and `width(s)` the true width.
//!
//! - **Convergence.** A prior translated by 5 px and bent by a further ±3 px along the
//!   normal converges onto the truth: the refined centreline and the final widths are
//!   checked station by station.
//! - **Determinism.** Repeated calls, a reused tracker and a fresh one agree to the bit;
//!   `u8`, `u16` and `f32` renders of one scene agree to quantisation.
//! - **Conventions.** A reversed prior gives the same curve reversed, with its normals,
//!   offsets and edge order flipped.
//! - **Absence and trouble.** A flat image or the wrong polarity is an `Ok` result with
//!   every station rejected; a bead leaving the image is rejected there with typed
//!   reasons; a gap is bridged and reported; a distractor step beside an edge is rejected
//!   by the clearance gate, and without it pulls the edge only when within a few pixels;
//!   a highlight inside the bead leaves the outer edges chosen; a background gradient
//!   moves nothing; a width that varies along the bead is measured where it is.
//! - **Spacing.** The refined curve does not depend on the station spacing.
//! - **Short-scale error in the prior.** A local bump and a coarse polygon decay over
//!   passes rather than in one: the defaults leave pinned residuals that the final stage
//!   reports, and a shorter bending length converges. A kinked prior is stepped by less
//!   than the full correction, without folding, and its error falls pass by pass.
//! - **Explaining.** `diagnostics::explain_bead` returns `track`'s result to the bit on a
//!   scene with hits, rejections and ambiguous stations, with every station of every
//!   pass and of the final stage traced, and leaves the tracker as it was.
//! - **Convergence basins** (`bead/basins.rs`). How far a prior may be translated,
//!   rotated, bent or bumped and still converge, pinned; no false lock beyond the reach.
//!
//! Stations within `END_MARGIN` px of a ribbon end, or of a gap's end, carry no truth
//! (the fixture cuts the ribbon square there) and are left out of accuracy checks.

#[path = "../examples/common/ribbon.rs"]
mod ribbon;

// How far a prior may be off and still converge.
#[path = "bead/basins.rs"]
mod basins;

use std::f64::consts::TAU;
use std::num::NonZeroUsize;

use ribbon::{Curve, P2, Raster, Ribbon, Scene, Step, Stripe, Width};
use vision_metrology::measure::diagnostics::{BeadStationTrace, explain, explain_bead};
use vision_metrology::measure::{
    BeadCaliper, BeadConfig, BeadPolarity, BeadReject, BeadStop, BeadTracker, BeadTuning, Caliper,
    RejectReason, TrackedBead,
};
use vision_metrology::{Error, Image, Point2f, Vec2f};

const SIZE: (usize, usize) = (340, 280);
const BG: f64 = 40.0;
const FG: f64 = 180.0;
const SIGMA: f64 = 1.2;
const WIDTH: f64 = 50.0;
/// Arc length, px, from a ribbon end or a gap end within which a station carries no truth.
const END_MARGIN: f64 = 10.0;

/// The refined centreline's worst distance from the truth, px, on a noise-free `u8` render
/// with the default config. Measured: 0.019 (line), 0.025 (arc), 0.029 (sine).
const CENTER_TOL: f64 = 0.06;
/// The final stage's worst width error, px, on the same renders. Measured: 0.017 (line),
/// 0.019 (arc), 0.024 (sine).
const WIDTH_TOL: f64 = 0.04;

fn line() -> Curve {
    let a = P2::new(60.0, 70.0);
    Curve::line(a, a + P2::polar(17f64.to_radians()) * 220.0)
}

fn arc() -> Curve {
    Curve::arc(P2::new(170.0, 260.0), 150.0, -2.4, 1.4)
}

fn sine() -> Curve {
    Curve::sine(P2::new(60.0, 150.0), P2::polar(0.1), 240.0, 12.0, 160.0)
}

fn bead(curve: Curve) -> Ribbon {
    Ribbon::new(curve, Width::Const(WIDTH))
}

fn render(ribbon: &Ribbon) -> Raster {
    Scene::new(SIZE.0, SIZE.1, ribbon.clone(), BG, FG, SIGMA).render()
}

/// A prior along `ribbon` from arc length `s0` to `s1`, a vertex every 4 px, moved by
/// `shift` and by `bend(s)` px along the true normal.
fn prior(ribbon: &Ribbon, s0: f64, s1: f64, shift: P2, bend: impl Fn(f64) -> f64) -> Vec<Point2f> {
    let n = ((s1 - s0) / 4.0).ceil() as usize;
    (0..=n)
        .map(|k| {
            let s = s0 + (s1 - s0) * k as f64 / n as f64;
            (ribbon.curve.center(s) + ribbon.curve.normal(s) * bend(s) + shift).point()
        })
        .collect()
}

/// The standard perturbed prior: 15 px in from each end, shifted by (3, −4) px and bent
/// by ±3 px over the ribbon's length.
fn perturbed(ribbon: &Ribbon) -> Vec<Point2f> {
    let len = ribbon.length();
    prior(ribbon, 15.0, len - 15.0, P2::new(3.0, -4.0), |s| {
        3.0 * (TAU * s / len).sin()
    })
}

/// Whether arc length `s` carries truth: away from the ends and from every gap's ends.
fn has_truth(ribbon: &Ribbon, s: f64) -> bool {
    let away = |a: f64| (s - a).abs() > END_MARGIN;
    away(0.0) && away(ribbon.length()) && ribbon.gaps.iter().all(|&(a, b)| away(a) && away(b))
}

/// The worst centreline distance and the worst width error over the stations with truth,
/// and how many stations were checked.
fn errors(bead: &TrackedBead, ribbon: &Ribbon) -> (f64, f64, usize) {
    let (mut center, mut width, mut n) = (0.0f64, 0.0f64, 0);
    for smp in &bead.samples {
        let (s, d) = ribbon.nearest(smp.point);
        if !has_truth(ribbon, s) {
            continue;
        }
        n += 1;
        center = center.max(d.abs());
        if let Ok(hit) = smp.hit {
            width = width.max((f64::from(hit.pair.width) - ribbon.width(s)).abs());
        }
    }
    (center, width, n)
}

fn track(cfg: BeadConfig, img: &Image<u8>, prior: &[Point2f]) -> TrackedBead {
    let mut tracker = BeadTracker::new(cfg).expect("a valid config");
    tracker
        .track(&img.as_view(), prior)
        .expect("a trackable prior")
}

#[test]
fn a_perturbed_prior_converges_onto_the_bead() {
    for (name, curve) in [("line", line()), ("arc", arc()), ("sine", sine())] {
        let ribbon = bead(curve);
        let img = render(&ribbon).to_u8();
        let prior = perturbed(&ribbon);
        let got = track(BeadConfig::default(), &img, &prior);
        let (center, width, n) = errors(&got, &ribbon);
        let stats = got.summary.stats.expect("a bead");
        eprintln!(
            "{name}: {} stations, stop {:?} after {} passes; centre max {center:.4} px, width \
             max err {width:.4} px; center_rms {:.4}, width {:.3} ± {:.4}",
            got.samples.len(),
            got.track.stop,
            got.track.passes.len(),
            stats.center_rms,
            stats.width_mean,
            stats.width_std
        );
        assert!(n > 40, "{name}: only {n} stations checked");
        assert_eq!(got.summary.support, 1.0, "{name}: every station hits");
        assert!(
            center < CENTER_TOL,
            "{name}: refined centreline {center:.4} px from the truth"
        );
        assert!(width < WIDTH_TOL, "{name}: width off by {width:.4} px");
        // The first pass carries the prior's error; the loop then converges.
        let first = &got.track.passes[0];
        assert!(
            first.solve.expect("solved").correction_max > 5.0,
            "{name}: {first:?}"
        );
        assert_eq!(got.track.stop, BeadStop::Converged, "{name}");
        assert!(
            stats.center_rms < 0.05 && stats.center_max_dev < 0.1,
            "{name}: {stats:?}"
        );
    }
}

#[test]
fn noise_moves_the_curve_by_little() {
    // 3 DN of seeded Gaussian noise on 140 DN of contrast.
    for (name, curve) in [("line", line()), ("sine", sine())] {
        let ribbon = bead(curve);
        let img = render(&ribbon).noisy(3.0, 11).to_u8();
        let got = track(BeadConfig::default(), &img, &perturbed(&ribbon));
        let (center, width, _) = errors(&got, &ribbon);
        eprintln!("{name} at 3 DN: centre max {center:.4} px, width max err {width:.4} px");
        // Measured: centre 0.041 (line) and 0.065 (sine) px, width 0.21 and 0.14 px.
        assert!(center < 0.08, "{name}: centre {center:.4}");
        assert!(width < 0.3, "{name}: width {width:.4}");
    }
}

#[test]
fn tracking_is_deterministic() {
    let ribbon = bead(arc());
    let img = render(&ribbon).to_u8();
    let prior = perturbed(&ribbon);
    let mut reused = BeadTracker::new(BeadConfig::default()).expect("valid");
    let first = reused.track(&img.as_view(), &prior).expect("tracked");
    assert_eq!(
        reused.track(&img.as_view(), &prior).expect("tracked"),
        first
    );
    // A tracker that has worked on another scene and size answers like a fresh one.
    let other = bead(sine());
    let small = Scene::new(320, 240, other.clone(), BG, FG, SIGMA)
        .render()
        .to_u8();
    let _ = reused
        .track(&small.as_view(), &perturbed(&other))
        .expect("tracked");
    assert_eq!(
        reused.track(&img.as_view(), &prior).expect("tracked"),
        first
    );
    assert_eq!(track(BeadConfig::default(), &img, &prior), first);
}

#[test]
fn pixel_types_agree() {
    let ribbon = bead(sine());
    let raster = render(&ribbon);
    let prior = perturbed(&ribbon);
    let base = track(BeadConfig::default(), &raster.to_u8(), &prior);
    // `u16` is DN × 256: the thresholds scale with it.
    let d = BeadConfig::default();
    let scaled = |c: BeadCaliper| BeadCaliper {
        threshold: 256.0 * c.threshold,
        ..c
    };
    let cfg16 = BeadConfig {
        track: scaled(d.track),
        measure: scaled(d.measure),
        ..d
    };
    let wide = BeadTracker::new(cfg16)
        .expect("valid")
        .track(&raster.to_u16().as_view(), &prior)
        .expect("tracked");
    let float = BeadTracker::new(d)
        .expect("valid")
        .track(&raster.to_f32().as_view(), &prior)
        .expect("tracked");
    for (name, other) in [("u16", &wide), ("f32", &float)] {
        assert_eq!(other.samples.len(), base.samples.len());
        let (mut dc, mut dw) = (0.0f32, 0.0f32);
        for (a, b) in base.samples.iter().zip(&other.samples) {
            dc = dc.max((a.point - b.point).norm());
            let (ha, hb) = (a.hit.expect("a hit"), b.hit.expect("a hit"));
            dw = dw.max((ha.pair.width - hb.pair.width).abs());
        }
        eprintln!("u8 vs {name}: centreline {dc:.4} px, width {dw:.4} px");
        // Measured: 0.004 px and 0.014 px against either, u8's rounding.
        assert!(dc < 0.02 && dw < 0.03, "u8 vs {name}: {dc} / {dw}");
    }
}

#[test]
fn a_reversed_prior_gives_the_same_curve_reversed() {
    let ribbon = bead(sine());
    let img = render(&ribbon).to_u8();
    let prior = perturbed(&ribbon);
    let fwd = track(BeadConfig::default(), &img, &prior);
    let rev_prior: Vec<Point2f> = prior.iter().rev().copied().collect();
    let rev = track(BeadConfig::default(), &img, &rev_prior);
    let n = fwd.samples.len();
    assert_eq!(rev.samples.len(), n);
    let mut worst = 0.0f32;
    for (i, a) in fwd.samples.iter().enumerate() {
        let b = &rev.samples[n - 1 - i];
        worst = worst.max((a.point - b.point).norm());
        assert!(
            (a.normal + b.normal).norm() < 1e-3,
            "station {i}: normals not flipped"
        );
        let (ha, hb) = (a.hit.expect("a hit"), b.hit.expect("a hit"));
        assert!(
            (ha.offset + hb.offset).abs() < 2e-3,
            "station {i}: offsets not flipped"
        );
        assert!(
            (ha.pair.width - hb.pair.width).abs() < 2e-3,
            "station {i}: widths differ"
        );
        // −n is +n reversed: the first edge one way is the second the other.
        assert!(
            (ha.pair.first.p - hb.pair.second.p).norm() < 2e-3,
            "station {i}"
        );
    }
    eprintln!("reversed prior: stations agree to {worst:.2e} px");
    assert!(worst < 1e-3, "reversed curve differs by {worst} px");
}

#[test]
fn a_missing_bead_is_a_result_not_an_error() {
    let ribbon = bead(line());
    let prior = perturbed(&ribbon);
    let flat = Image::from_vec(SIZE.0, SIZE.1, vec![100u8; SIZE.0 * SIZE.1]).expect("image");
    let light = render(&ribbon).to_u8();
    let dark = BeadConfig {
        polarity: BeadPolarity::Dark,
        ..BeadConfig::default()
    };
    for (name, img, cfg, reason) in [
        (
            "flat",
            &flat,
            BeadConfig::default(),
            BeadReject::Caliper(RejectReason::NoEdge),
        ),
        ("wrong polarity", &light, dark, BeadReject::NoPair),
    ] {
        let got = track(cfg, img, &prior);
        assert_eq!(got.track.stop, BeadStop::TooFewValid, "{name}");
        assert_eq!(got.track.passes.len(), 1, "{name}");
        assert_eq!(got.track.passes[0].n_valid, 0, "{name}");
        assert_eq!(got.summary.support, 0.0, "{name}");
        assert!(got.summary.stats.is_none(), "{name}");
        assert!(got.samples.iter().all(|s| s.hit == Err(reason)), "{name}");
        assert_eq!(got.summary.rejects, vec![(reason, got.samples.len())]);
        // The curve stays where the prior put it.
        assert!((got.centerline[0] - prior[0]).norm() < 1e-3, "{name}");
    }
}

#[test]
fn a_bead_leaving_the_image_is_rejected_where_it_leaves() {
    // Horizontal, running off the right edge at x = 339 and on to x = 500.
    let ribbon = bead(Curve::line(P2::new(40.0, 140.0), P2::new(500.0, 140.0)));
    let img = render(&ribbon).to_u8();
    let prior = prior(&ribbon, 15.0, 440.0, P2::new(0.0, 4.0), |_| 0.0);
    let got = track(BeadConfig::default(), &img, &prior);
    let mut off = 0;
    for smp in &got.samples {
        let x = smp.point.x;
        if x > SIZE.0 as f32 + 5.0 {
            // Every edge such a strip finds is in border fill.
            off += 1;
            assert_eq!(
                smp.hit,
                Err(BeadReject::Caliper(RejectReason::OffImage)),
                "station at x = {x}"
            );
        } else if x < SIZE.0 as f32 - 60.0 {
            let hit = smp.hit.expect("a hit inside the image");
            assert!((hit.pair.center.y - 140.0).abs() < 0.05, "x = {x}: {hit:?}");
            assert!(
                (smp.point.y - 140.0).abs() < 0.05,
                "x = {x}: {:?}",
                smp.point
            );
        }
    }
    assert!(off > 20, "{off} stations off the image");
    let off_image = BeadReject::Caliper(RejectReason::OffImage);
    assert!(
        got.summary.rejects.iter().any(|&(r, _)| r == off_image),
        "{:?}",
        got.summary.rejects
    );
}

#[test]
fn a_gap_is_bridged_and_reported() {
    let ribbon = bead(line()).gap(100.0, 130.0);
    let img = render(&ribbon).to_u8();
    let got = track(BeadConfig::default(), &img, &perturbed(&ribbon));
    let (center, width, _) = errors(&got, &ribbon);
    let gap = got.summary.longest_gap;
    eprintln!(
        "30 px gap: reported {gap} px (pass 1: {} px), centre max {center:.4}",
        got.track.passes[0].longest_gap
    );
    assert!((gap - 30.0).abs() <= 8.0, "longest gap {gap} px");
    assert!((got.track.passes[0].longest_gap - 30.0).abs() <= 8.0);
    // Inside the gap the curve is bridged, not left behind.
    assert!(center < CENTER_TOL, "centre {center:.4}");
    assert!(width < WIDTH_TOL, "width {width:.4}");
    for smp in &got.samples {
        let (s, _) = ribbon.nearest(smp.point);
        if s > 100.0 + 3.0 && s < 130.0 - 3.0 {
            assert!(smp.hit.is_err(), "a hit inside the gap at s = {s}");
        }
    }
}

/// The line bead with a step 80 DN brighter beyond its +n edge, `near` px outside the edge
/// at the bead's start and diverging to 30 px at its end.
fn line_with_step(near: f64) -> (Ribbon, Image<u8>) {
    let ribbon = bead(line());
    let (a, len) = (ribbon.curve.center(0.0), ribbon.length());
    let (t, n) = (ribbon.curve.tangent(0.0), ribbon.curve.normal(0.0));
    let tilt = ((30.0 - near) / len).atan();
    let dir = t * tilt.cos() + n * tilt.sin();
    let step = Step {
        point: a + n * (0.5 * WIDTH + near),
        normal: dir.perp(),
        contrast: 80.0,
    };
    let scene = Scene::new(SIZE.0, SIZE.1, ribbon.clone(), BG, FG, SIGMA).step(step);
    (ribbon, scene.render().to_u8())
}

#[test]
fn clearance_rejects_a_distractor_beside_an_edge() {
    let (ribbon, img) = line_with_step(3.0);
    let prior = perturbed(&ribbon);
    let clear = BeadConfig {
        clearance: Some(8.0),
        ..BeadConfig::default()
    };
    let got = track(clear, &img, &prior);
    let (mut rejected, mut hits) = (0, 0);
    for smp in &got.samples {
        let (s, _) = ribbon.nearest(smp.point);
        // The step's distance from the +n edge here.
        let gap = 3.0 + 27.0 * s / ribbon.length();
        match smp.hit {
            Err(BeadReject::Clearance) => {
                assert!(gap < 8.0 + 3.0, "rejected with the step {gap:.1} px away");
                rejected += 1;
            }
            Ok(hit) => {
                assert!(gap > 8.0 - 3.0, "a hit with the step {gap:.1} px away");
                let err = (f64::from(hit.pair.width) - WIDTH).abs();
                assert!(
                    err < 0.1,
                    "width off by {err:.3} with the step {gap:.1} px away"
                );
                hits += 1;
            }
            Err(other) => panic!("s = {s:.1}: {other:?}"),
        }
    }
    eprintln!("step beside the edge: {rejected} stations rejected, {hits} hits");
    assert!(
        rejected >= 5 && hits >= 30,
        "{rejected} rejected, {hits} hits"
    );
    // The rejected stations are at the start, so the curve there is extrapolated from the
    // rest; where the bead was measured, the curve is on it.
    let worst = got
        .samples
        .iter()
        .filter(|smp| smp.hit.is_ok())
        .map(|smp| ribbon.nearest(smp.point).1.abs())
        .fold(0.0f64, f64::max);
    assert!(worst < CENTER_TOL, "centre {worst:.4} at a hit");
    assert!(got.summary.support < 1.0 && got.summary.longest_gap > 20.0);

    // Without the gate the same stations measure, beside the distractor.
    let open = track(BeadConfig::default(), &img, &prior);
    assert_eq!(open.summary.support, 1.0);
}

#[test]
fn a_highlight_inside_the_bead_keeps_the_outer_edges() {
    let ribbon = bead(arc());
    let highlight = Stripe {
        offset: 6.0,
        width: 8.0,
        contrast: 50.0,
    };
    let img = Scene::new(SIZE.0, SIZE.1, ribbon.clone(), BG, FG, SIGMA)
        .highlight(highlight)
        .render()
        .to_u8();
    for clearance in [None, Some(8.0)] {
        let cfg = BeadConfig {
            clearance,
            ..BeadConfig::default()
        };
        let got = track(cfg, &img, &perturbed(&ribbon));
        let (center, width, _) = errors(&got, &ribbon);
        eprintln!("highlight, clearance {clearance:?}: centre {center:.4}, width {width:.4}");
        assert_eq!(got.summary.support, 1.0);
        assert!(
            center < CENTER_TOL && width < WIDTH_TOL,
            "{center} / {width}"
        );
    }
}

#[test]
fn without_clearance_a_distractor_near_an_edge_pulls_it() {
    // The step touches the bead's +n edge at its start and diverges to 30 px at its end.
    let (ribbon, img) = line_with_step(0.0);
    let got = track(BeadConfig::default(), &img, &perturbed(&ribbon));
    assert_eq!(got.summary.support, 1.0);
    let (mut near, mut far) = (0.0f64, 0.0f64);
    for smp in &got.samples {
        let (s, _) = ribbon.nearest(smp.point);
        let gap = 30.0 * s / ribbon.length();
        let hit = smp.hit.expect("a hit");
        let width = f64::from(hit.pair.width) - WIDTH;
        let center = ribbon.nearest(hit.pair.center).1;
        if gap < 3.0 {
            // The step's rising edge pushes the bead's falling one inwards.
            near = near.min(width);
            assert!(
                width < 0.0 && center < 0.0,
                "gap {gap:.1}: {width} / {center}"
            );
        } else if gap > 7.0 {
            far = far.max(width.abs()).max(center.abs());
        }
    }
    eprintln!("a step beside the edge: width {near:.3} px within 3 px, {far:.3} beyond 7 px");
    // Measured: up to 0.43 px short within 3 px of the step; beyond 7 px, within 0.017 px.
    // Nearer than about 5 px, use `clearance`.
    assert!(near < -0.25 && near > -0.8, "{near}");
    assert!(far < 0.03, "{far}");
}

#[test]
fn a_background_gradient_leaves_the_bead_in_place() {
    // The background rises from 20 DN to 133 DN across the image, nearly the bead's
    // 120 DN of contrast, and nothing clips.
    for (name, curve) in [("line", line()), ("sine", sine())] {
        let ribbon = bead(curve);
        let img = Scene::new(SIZE.0, SIZE.1, ribbon.clone(), 20.0, 140.0, SIGMA)
            .gradient(P2::new(0.0, 0.0), P2::new(0.25, 0.1))
            .render()
            .to_u8();
        let got = track(BeadConfig::default(), &img, &perturbed(&ribbon));
        let (center, width, _) = errors(&got, &ribbon);
        eprintln!("gradient, {name}: centre {center:.4} px, width {width:.4} px");
        // Measured: 0.021 (line) and 0.028 (sine) px, widths 0.021 and 0.025 px: a linear
        // background adds a constant to the derivative, which moves no peak.
        assert_eq!(got.summary.support, 1.0, "{name}");
        assert!(
            center < CENTER_TOL && width < WIDTH_TOL,
            "{name}: {center} / {width}"
        );
    }
}

#[test]
fn a_varying_width_is_measured_station_by_station() {
    // From 35 px at the start to 65 px at the end.
    for (name, curve) in [("line", line()), ("arc", arc()), ("sine", sine())] {
        let ribbon = Ribbon::new(curve, Width::Linear { w0: 35.0, w1: 65.0 });
        let img = render(&ribbon).to_u8();
        let got = track(BeadConfig::default(), &img, &perturbed(&ribbon));
        let (center, width, n) = errors(&got, &ribbon);
        let stats = got.summary.stats.expect("hits");
        eprintln!(
            "35 to 65 px, {name}: centre {center:.4} px, width {width:.4} px, {:.2} to {:.2}",
            stats.width_min, stats.width_max
        );
        // Measured: centre 0.012, 0.022 and 0.039 px; width 0.019, 0.027 and 0.038 px.
        assert_eq!(got.summary.support, 1.0, "{name}");
        assert!(n > 40 && center < CENTER_TOL, "{name}: {center}");
        assert!(width < 0.06, "{name}: {width}");
        // The prior starts and ends 15 px in.
        assert!(
            stats.width_min < 38.0 && stats.width_max > 62.0,
            "{name}: {stats:?}"
        );
    }
}

/// The curve's signed distance from the true centreline at arc length `s`, linear between
/// the stations of `curve`; `(s, d)` from `nearest`, in increasing `s`.
fn offset_at(curve: &[(f64, f64)], s: f64) -> f64 {
    let k = curve
        .partition_point(|&(sk, _)| sk < s)
        .clamp(1, curve.len() - 1);
    let ((s0, d0), (s1, d1)) = (curve[k - 1], curve[k]);
    d0 + (d1 - d0) * ((s - s0) / (s1 - s0)).clamp(0.0, 1.0)
}

#[test]
fn the_curve_does_not_depend_on_the_spacing() {
    let ribbon = bead(sine());
    let img = render(&ribbon).to_u8();
    let prior = perturbed(&ribbon);
    let at = |spacing: f32| {
        track(
            BeadConfig {
                spacing,
                ..BeadConfig::default()
            },
            &img,
            &prior,
        )
    };
    let (fine, coarse) = (at(2.0), at(4.0));
    assert!(fine.samples.len() > coarse.samples.len() + 40);
    // Compared along the truth's normal: a straight line between stations would cut the
    // bend's chords by up to h²/(8R), about 0.04 px at 4 px on the sine's tightest bend.
    let truth = |b: &TrackedBead| -> Vec<(f64, f64)> {
        b.centerline.iter().map(|&p| ribbon.nearest(p)).collect()
    };
    let coarse = truth(&coarse);
    let worst = truth(&fine)
        .iter()
        .map(|&(s, d)| (d - offset_at(&coarse, s)).abs())
        .fold(0.0f64, f64::max);
    eprintln!("spacing 2 vs 4: curves {worst:.4} px apart");
    // Measured: 0.014 px.
    assert!(worst < 0.05, "spacing 2 and 4 differ by {worst} px");
}

#[test]
fn bad_priors_are_errors() {
    let img = Image::from_vec(8, 8, vec![0u8; 64]).expect("image");
    let mut tracker = BeadTracker::new(BeadConfig::default()).expect("valid");
    let p = Point2f::new(2.0, 3.0);
    let mut fails = |prior: &[Point2f]| tracker.track(&img.as_view(), prior).unwrap_err();
    // Too few points to place stations on.
    assert_eq!(fails(&[]), Error::InsufficientData { need: 2, got: 0 });
    assert_eq!(fails(&[p]), Error::InsufficientData { need: 2, got: 1 });
    // Points enough, but no curve: a prior of length zero, or one with a NaN in it.
    assert_eq!(
        fails(&[p, p, p]),
        Error::Degenerate("bead prior has zero length")
    );
    assert_eq!(
        fails(&[p, Point2f::new(f32::NAN, 1.0)]),
        Error::Degenerate("bead prior has a non-finite point")
    );
    let bad = BeadConfig {
        min_width: 90.0,
        ..BeadConfig::default()
    };
    assert!(BeadTracker::new(bad).is_err());
    assert!(tracker.set_config(bad).is_err());
    assert_eq!(tracker.config(), &BeadConfig::default());
}

/// The default config with `bending_px` and `passes` set, and `tol` tiny when `all_passes`.
fn tuned(bending_px: f32, passes: usize, all_passes: bool) -> BeadConfig {
    let d = BeadConfig::default();
    BeadConfig {
        tuning: BeadTuning {
            bending_px,
            passes: NonZeroUsize::new(passes).expect("at least one pass"),
            tol: if all_passes { 1e-6 } else { d.tuning.tol },
            ..d.tuning
        },
        ..d
    }
}

// The penalties act on each pass's correction, so a deviation of the prior shorter than
// about 2π·bending_px decays over passes rather than in one. The two cases below pin how
// far the defaults get, and that a shorter bending length converges; the final stage's
// `center_max_dev` reports what is left either way.

#[test]
fn a_local_bump_in_the_prior_decays_over_passes() {
    // An 8 px Gaussian bump, σ = 8 px of arc, on the straight bead.
    let ribbon = bead(line());
    let img = render(&ribbon).to_u8();
    let len = ribbon.length();
    let prior = prior(&ribbon, 15.0, len - 15.0, P2::new(0.0, 0.0), |s| {
        8.0 * (-(s - 0.5 * len).powi(2) / (2.0 * 8.0 * 8.0)).exp()
    });
    let run = |cfg: BeadConfig| {
        let got = track(cfg, &img, &prior);
        let (center, _, _) = errors(&got, &ribbon);
        let max_dev = f64::from(got.summary.stats.expect("hits").center_max_dev);
        (got, center, max_dev)
    };

    // Measured: 0.215 px left after the default 3 passes. With up to 6, the corrections
    // stop after 5 with 0.125 px left: `Converged` says they stopped, not that the curve
    // fits.
    let (three, center, max_dev) = run(BeadConfig::default());
    eprintln!("bump, defaults: {center:.3} px after 3 passes");
    assert_eq!(three.track.stop, BeadStop::PassLimit);
    assert!(center > 0.15 && center < 0.32, "{center}");
    assert!(
        (max_dev - center).abs() < 0.1,
        "the final stage sees it: {max_dev}"
    );
    let (six, more, _) = run(tuned(4.0, 6, false));
    eprintln!("bump, defaults: {more:.3} px, {:?}", six.track.stop);
    assert_eq!(six.track.stop, BeadStop::Converged);
    assert!(more > 0.08 && more < 0.19, "{more}");

    // Measured: 1.72 px left after 3 passes with bending_px = 8, 0.76 px after 10.
    let (_, stiff, _) = run(tuned(8.0, 3, false));
    let (_, ten, _) = run(tuned(8.0, 10, false));
    eprintln!("bump, bending_px 8: {stiff:.3} px after 3 passes, {ten:.3} px after 10");
    assert!(stiff > 1.2 && stiff < 2.6, "{stiff}");
    assert!(ten < 0.6 * stiff && ten < 1.15, "{ten}");

    // Measured: 0.056 px with bending_px = 3 and up to 6 passes, converged.
    let (short, center, max_dev) = run(tuned(3.0, 6, false));
    eprintln!("bump, bending_px 3: {center:.3} px, {:?}", short.track.stop);
    assert_eq!(short.track.stop, BeadStop::Converged);
    assert!(center < 0.085 && max_dev < 0.1, "{center} / {max_dev}");
}

#[test]
fn a_coarse_polygon_prior_converges_only_with_a_short_bending_length() {
    // Six vertices on the arc, one every 40 px: the chords' 1.3 px sagitta varies along
    // the curve faster than the default bending length passes.
    let ribbon = bead(arc());
    let img = render(&ribbon).to_u8();
    let len = ribbon.length();
    let n = ((len - 30.0) / 40.0).round() as usize;
    let polygon: Vec<Point2f> = (0..=n)
        .map(|k| {
            let s = 15.0 + (len - 30.0) * k as f64 / n as f64;
            ribbon.curve.center(s).point()
        })
        .collect();
    assert_eq!(polygon.len(), 6);
    let run = |cfg: BeadConfig| {
        let got = track(cfg, &img, &polygon);
        let (center, _, _) = errors(&got, &ribbon);
        let max_dev = f64::from(got.summary.stats.expect("hits").center_max_dev);
        (got, center, max_dev)
    };

    // Measured: with up to 10 passes, the defaults report `Converged` after 3 with the
    // curve still 0.160 px off, and bending_px = 8 after 4 with 0.41 px. Convergence says
    // the corrections stopped, not that the curve fits.
    for (bending_px, lo, hi) in [(4.0, 0.11, 0.24), (8.0, 0.3, 0.6)] {
        let (got, center, max_dev) = run(tuned(bending_px, 10, false));
        eprintln!(
            "40 px polygon, bending_px {bending_px}: {center:.3} px, {:?} after {} passes",
            got.track.stop,
            got.track.passes.len()
        );
        assert_eq!(got.track.stop, BeadStop::Converged);
        assert!(center > lo && center < hi, "{bending_px}: {center}");
        assert!(max_dev > lo, "the final stage sees it: {max_dev}");
    }

    // Measured: 0.044 px with bending_px = 2.
    let (got, center, max_dev) = run(tuned(2.0, 6, false));
    eprintln!(
        "40 px polygon, bending_px 2: {center:.3} px, {:?}",
        got.track.stop
    );
    assert!(center < 0.07 && max_dev < 0.07, "{center} / {max_dev}");
}

#[test]
fn a_kinked_prior_is_stepped_without_folding() {
    // A tent 14 px high and 60 px wide on the straight bead, seen with a 2 px tangent
    // window: the curvature concentrates at the apex, and the full first correction would
    // reverse a segment there, so the fold guard shortens it.
    let ribbon = bead(line());
    let img = render(&ribbon).to_u8();
    let len = ribbon.length();
    let prior = prior(&ribbon, 15.0, len - 15.0, P2::new(0.0, 0.0), |s| {
        14.0 * (1.0 - (s - 0.5 * len).abs() / 30.0).max(0.0)
    });
    let mut last = f64::INFINITY;
    for passes in [1, 2, 3, 6] {
        let mut cfg = tuned(8.0, passes, true);
        cfg.tuning.tangent_window_px = 2.0;
        let got = track(cfg, &img, &prior);
        let (center, _, _) = errors(&got, &ribbon);
        let scale = got.track.passes[0].solve.expect("solved").step_scale;
        // Adjacent segments never turn back on each other.
        let turn = got
            .centerline
            .windows(3)
            .map(|w| (w[1] - w[0]).normalize().dot(&(w[2] - w[1]).normalize()))
            .fold(1.0f32, f32::min);
        eprintln!(
            "kinked prior, {passes} passes: {center:.3} px, first step {scale:.3}, turn cos \
             {turn:.3}"
        );
        // Measured: a first step of 0.668; 5.49, 1.26, 0.75 and 0.42 px; turn cos ≥ 0.95.
        assert!(scale < 0.8, "first step {scale}");
        assert!(
            turn > 0.9,
            "{passes} passes: adjacent segments turn by acos {turn}"
        );
        assert!(center < last, "{passes} passes: {center} after {last}");
        last = center;
    }
    assert!(last < 0.65, "after 6 passes: {last}");
}

// ── explaining a run ─────────────────────────────────────────────────────

/// The line bead with a gap at arc length 150..175, crossed at a shallow angle by a step
/// 80 DN brighter on its +n side, tracked with `min_margin` 0.7:
/// - in the gap, the step's edge has no partner, and the stations find no pair;
/// - where the step runs near the bead's −n edge, its edge and the bead's +n edge make a
///   second pair whose score is close enough to the bead's that the margin gate rejects
///   the station as ambiguous;
/// - everywhere else the bead is found.
fn troubled() -> (Image<u8>, Vec<Point2f>, BeadConfig) {
    let ribbon = bead(line()).gap(150.0, 175.0);
    let (len, t, n) = (
        ribbon.length(),
        ribbon.curve.tangent(0.0),
        ribbon.curve.normal(0.0),
    );
    // The step's edge crosses the bead at its middle, (s − L/2)/3 px along the normal at
    // arc length s.
    let tilt = (1.0f64 / 3.0).atan();
    let dir = t * tilt.cos() + n * tilt.sin();
    let step = Step {
        point: ribbon.curve.center(0.5 * len),
        normal: dir.perp(),
        contrast: 80.0,
    };
    let img = Scene::new(SIZE.0, SIZE.1, ribbon.clone(), BG, FG, SIGMA)
        .step(step)
        .render()
        .to_u8();
    let cfg = BeadConfig {
        min_margin: Some(0.7),
        ..BeadConfig::default()
    };
    (img, perturbed(&ribbon), cfg)
}

/// How many of `stations` found the bead, were rejected by a gate before the margin, and
/// were ambiguous.
fn kinds(stations: &[BeadStationTrace]) -> (usize, usize, usize) {
    let ambiguous = Err(BeadReject::Ambiguous);
    let hits = stations.iter().filter(|s| s.hit.is_ok()).count();
    let unclear = stations.iter().filter(|s| s.hit == ambiguous).count();
    (hits, stations.len() - hits - unclear, unclear)
}

#[test]
fn explaining_returns_what_track_returns() {
    let (img, prior, cfg) = troubled();
    let mut tracker = BeadTracker::new(cfg).expect("valid");
    let trace = explain_bead(&mut tracker, &img.as_view(), &prior).expect("tracked");
    let tracked = track(cfg, &img, &prior);
    assert_eq!(trace.result, tracked);
    // `Debug` prints each float's shortest round-trip form, so equal text is equal bits,
    // signed zeros included.
    assert_eq!(format!("{:?}", trace.result), format!("{tracked:?}"));

    // The scene has every kind of station, in the passes and in the final stage.
    // Measured: 30 hits, 6 rejected and 13 ambiguous in the first pass; 38, 7 and 4 in
    // the final stage.
    let first = kinds(&trace.passes[0].stations);
    let last = kinds(&trace.measure);
    eprintln!("hits, rejected, ambiguous: first pass {first:?}, final stage {last:?}");
    for (hits, rejected, ambiguous) in [first, last] {
        assert!(
            hits > 20 && rejected >= 4 && ambiguous >= 2,
            "{first:?} {last:?}"
        );
    }
}

#[test]
fn the_trace_holds_every_station_of_every_pass() {
    let (img, prior, cfg) = troubled();
    let mut tracker = BeadTracker::new(cfg).expect("valid");
    let trace = explain_bead(&mut tracker, &img.as_view(), &prior).expect("tracked");
    let bead = &trace.result;
    let n = bead.samples.len();
    assert_eq!(trace.passes.len(), bead.track.passes.len());
    assert!(trace.passes.len() > 1);
    assert_eq!(trace.measure.len(), n);

    let stage = |stations: &[BeadStationTrace], half_width: f32| {
        for (i, st) in stations.iter().enumerate() {
            // n = t.perp(), exactly.
            assert_eq!(st.normal, Vec2f::new(-st.tangent.y, st.tangent.x), "{i}");
            assert!(
                st.window.0 < 0.0 && st.window.1 > 0.0,
                "{i}: {:?}",
                st.window
            );
            // The strip runs through the station along its normal, from −n to +n.
            let mid = Point2f::from((st.strip.start.coords + st.strip.end.coords) * 0.5);
            assert!((mid - st.point).norm() < 1e-3, "{i}");
            let along = (st.strip.end - st.strip.start).normalize();
            assert!((along - st.normal).norm() < 1e-5, "{i}");
            assert_eq!(st.strip.half_width, half_width);
            // A caliper rejection is the station's; a hit's edges are the caliper's.
            match st.caliper.reject {
                Some(r) => assert_eq!(st.hit, Err(BeadReject::Caliper(r)), "{i}"),
                None => assert!(!st.caliper.edges.is_empty(), "{i}"),
            }
            if let Ok(hit) = st.hit {
                assert!(st.caliper.edges.contains(&hit.pair.first), "{i}");
                assert!(st.caliper.edges.contains(&hit.pair.second), "{i}");
            }
        }
    };
    for (k, (p, rec)) in trace.passes.iter().zip(&bead.track.passes).enumerate() {
        assert_eq!(p.stations.len(), n, "pass {k}");
        assert_eq!(p.weights.len(), n, "pass {k}");
        assert_eq!(p.corrections.len(), n, "pass {k}");
        stage(&p.stations, cfg.track.half_width);
        let valid = p.stations.iter().filter(|s| s.hit.is_ok()).count();
        assert_eq!(valid, rec.n_valid, "pass {k}");
        for (st, &w) in p.stations.iter().zip(&p.weights) {
            assert!(
                st.hit.is_ok() || w == 0.0,
                "pass {k}: a rejected station weighs {w}"
            );
            assert!((0.0..=1.0).contains(&w), "pass {k}: weight {w}");
        }
        let solve = rec.solve.expect("every pass solved");
        let largest = p.corrections.iter().fold(0.0f32, |m, c| m.max(c.abs()));
        assert_eq!(largest, solve.correction_max, "pass {k}");
    }
    // The next pass measured where this one moved the curve: resampling slides a station
    // along the curve, not across it.
    let (p0, p1, mid) = (&trace.passes[0], &trace.passes[1], n / 2);
    let (before, after) = (&p0.stations[mid], &p1.stations[mid]);
    let moved = before.point + before.normal * p0.corrections[mid];
    let (across, along) = (
        (after.point - moved).dot(&before.normal),
        (after.point - moved).dot(&before.tangent),
    );
    // Measured: a 4.72 px correction, then 0.06 px across and 0.74 px along.
    assert!(p0.corrections[mid].abs() > 3.0);
    assert!(
        across.abs() < 0.15 && along.abs() < 2.0,
        "{across} / {along}"
    );

    // The final stage is the result's samples, with the evidence beside them.
    stage(&trace.measure, cfg.measure.half_width);
    for (st, smp) in trace.measure.iter().zip(&bead.samples) {
        assert_eq!(
            (st.point, st.normal, st.hit),
            (smp.point, smp.normal, smp.hit)
        );
    }
    // Each caliper trace is `explain` of a caliper of the stage's config at that strip.
    for st in trace.measure.iter().step_by(5) {
        let mut alone = Caliper::strip(st.strip, cfg.measure.to_measure_config());
        assert_eq!(explain(&mut alone, &img.as_view()), st.caliper);
    }
}

#[test]
fn explaining_leaves_the_tracker_as_it_was() {
    let (img, prior, cfg) = troubled();
    let fresh = track(cfg, &img, &prior);
    let mut tracker = BeadTracker::new(cfg).expect("valid");
    let first = explain_bead(&mut tracker, &img.as_view(), &prior).expect("tracked");
    let after = tracker.track(&img.as_view(), &prior).expect("tracked");
    assert_eq!(format!("{after:?}"), format!("{fresh:?}"));
    // Explaining again, after tracking, gives the same trace.
    let again = explain_bead(&mut tracker, &img.as_view(), &prior).expect("tracked");
    assert_eq!(again, first);
}

#[test]
fn a_missing_bead_is_explained_station_by_station() {
    let ribbon = bead(line());
    let prior = perturbed(&ribbon);
    let flat = Image::from_vec(SIZE.0, SIZE.1, vec![100u8; SIZE.0 * SIZE.1]).expect("image");
    let mut tracker = BeadTracker::new(BeadConfig::default()).expect("valid");
    let trace = explain_bead(&mut tracker, &flat.as_view(), &prior).expect("a result");
    assert_eq!(trace.result.track.stop, BeadStop::TooFewValid);
    // One pass, which found nothing and so neither weighed nor moved any station.
    let [pass] = trace.passes.as_slice() else {
        panic!("{} passes", trace.passes.len())
    };
    let n = trace.result.samples.len();
    assert_eq!(pass.weights, vec![0.0; n]);
    assert_eq!(pass.corrections, vec![0.0; n]);
    let no_edge = Err(BeadReject::Caliper(RejectReason::NoEdge));
    for st in pass.stations.iter().chain(&trace.measure) {
        assert_eq!(st.hit, no_edge);
        assert_eq!(st.caliper.reject, Some(RejectReason::NoEdge));
        assert!(st.caliper.candidates.is_empty());
    }
    // A bad prior fails as `track` does.
    let p = Point2f::new(2.0, 3.0);
    assert_eq!(
        explain_bead(&mut tracker, &flat.as_view(), &[p]).unwrap_err(),
        Error::InsufficientData { need: 2, got: 1 }
    );
}

#[cfg(feature = "serde")]
#[test]
fn a_bead_trace_serializes() {
    let (img, prior, cfg) = troubled();
    let mut tracker = BeadTracker::new(cfg).expect("valid");
    let trace = explain_bead(&mut tracker, &img.as_view(), &prior).expect("tracked");
    let v = serde_json::to_value(&trace).expect("serializable");
    let n = trace.result.samples.len();
    assert_eq!(v["measure"].as_array().map(Vec::len), Some(n));
    assert_eq!(v["passes"][0]["weights"].as_array().map(Vec::len), Some(n));
    let st = &v["passes"][0]["stations"][0];
    assert!(st["strip"]["start"].is_array() && st["caliper"]["profile"].is_array());
    assert!(st["hit"].get("Ok").is_some() || st["hit"].get("Err").is_some());
}
