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
//!   by the clearance gate; a highlight inside the bead leaves the outer edges chosen.
//! - **Spacing.** The refined curve does not depend on the station spacing.
//!
//! Stations within `END_MARGIN` px of a ribbon end, or of a gap's end, carry no truth
//! (the fixture cuts the ribbon square there) and are left out of accuracy checks.

#[path = "../examples/common/ribbon.rs"]
mod ribbon;

use std::f64::consts::TAU;

use ribbon::{Curve, P2, Raster, Ribbon, Scene, Step, Stripe, Width};
use vision_metrology::measure::{
    BeadCaliper, BeadConfig, BeadPolarity, BeadReject, BeadStop, BeadTracker, RejectReason,
    TrackedBead,
};
use vision_metrology::{Image, Point2f};

const SIZE: (usize, usize) = (340, 280);
const BG: f64 = 40.0;
const FG: f64 = 180.0;
const SIGMA: f64 = 1.2;
const WIDTH: f64 = 50.0;
/// Arc length, px, from a ribbon end or a gap end within which a station carries no truth.
const END_MARGIN: f64 = 10.0;

/// The refined centreline's worst distance from the truth, px, on a noise-free `u8` render
/// with the default config. Measured: 0.023 (line), 0.042 (arc), 0.039 (sine).
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
        assert!(first.correction_max > 5.0, "{name}: {first:?}");
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
        // Measured: centre 0.045 (line) and 0.053 (sine) px, width 0.21 and 0.14 px.
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
        // Measured: 0.003 px and 0.014 px against either, u8's rounding.
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
            off += 1;
            match smp.hit {
                Err(BeadReject::Caliper(_) | BeadReject::NoPair | BeadReject::Width) => {}
                other => panic!("station at x = {x} off the image: {other:?}"),
            }
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
    // Measured: 0.017 px.
    assert!(worst < 0.05, "spacing 2 and 4 differ by {worst} px");
}

#[test]
fn bad_priors_are_errors() {
    let img = Image::from_vec(8, 8, vec![0u8; 64]).expect("image");
    let mut tracker = BeadTracker::new(BeadConfig::default()).expect("valid");
    let p = Point2f::new(2.0, 3.0);
    for prior in [
        vec![],
        vec![p],
        vec![p, p, p],
        vec![p, Point2f::new(f32::NAN, 1.0)],
    ] {
        assert!(tracker.track(&img.as_view(), &prior).is_err(), "{prior:?}");
    }
    let bad = BeadConfig {
        min_width: 90.0,
        ..BeadConfig::default()
    };
    assert!(BeadTracker::new(bad).is_err());
    assert!(tracker.set_config(bad).is_err());
    assert_eq!(tracker.config(), &BeadConfig::default());
}
