//! Coupled pairing (`BeadPairing::Coupled`, ADR-0019) end to end: it converges as
//! per-station pairing does, explaining returns its result to the bit with its margins,
//! a wall that leaves the image is still measured on its own, and a parallel bead lowers
//! the whole call's margin.

use vision_metrology::measure::diagnostics::explain_bead;
use vision_metrology::measure::{
    BeadCaliper, BeadConfig, BeadCoupling, BeadPairing, BeadReject, BeadSide, BeadTracker,
};

use super::ribbon::{Curve, P2, Ribbon, Scene, Stripe, Width};
use super::{
    BG, CENTER_TOL, FG, SIGMA, SIZE, WIDTH_TOL, arc, bead, errors, line, perturbed, prior, render,
    sine, track,
};

fn coupled() -> BeadConfig {
    BeadConfig {
        pairing: BeadPairing::Coupled(BeadCoupling::default()),
        ..BeadConfig::default()
    }
}

#[test]
fn coupled_pairing_converges_onto_the_bead() {
    for (name, curve) in [("line", line()), ("arc", arc()), ("sine", sine())] {
        let ribbon = bead(curve);
        let img = render(&ribbon).to_u8();
        let got = track(coupled(), &img, &perturbed(&ribbon));
        let (center, width, n) = errors(&got, &ribbon);
        let path = got.track.passes[0]
            .path
            .expect("a coupled pass scores its path");
        eprintln!(
            "{name}: centre {center:.4}, width {width:.4} over {n} stations, first-pass margin \
             {:?}, evidence {:.3}",
            path.margin, path.evidence
        );
        assert_eq!(
            got.summary.support, 1.0,
            "{name}: {:?}",
            got.summary.rejects
        );
        assert!(center < CENTER_TOL, "{name}: centre {center}");
        assert!(width < WIDTH_TOL, "{name}: width {width}");
        assert!(
            got.samples
                .iter()
                .all(|s| s.margin.is_some() && s.wall.is_none())
        );
        assert!(got.summary.path.is_some());
    }
}

#[test]
fn per_station_pairing_reports_no_path() {
    let ribbon = bead(line());
    let img = render(&ribbon).to_u8();
    let got = track(BeadConfig::default(), &img, &perturbed(&ribbon));
    assert!(got.summary.path.is_none() && got.track.passes.iter().all(|p| p.path.is_none()));
    assert!(
        got.samples
            .iter()
            .all(|s| s.margin.is_none() && s.wall.is_none())
    );
}

#[test]
fn coupled_explaining_returns_what_track_returns() {
    let ribbon = bead(arc()).gap(80.0, 110.0);
    let img = render(&ribbon).to_u8();
    let prior = perturbed(&ribbon);
    let mut tracker = BeadTracker::new(coupled()).expect("valid");
    let trace = explain_bead(&mut tracker, &img.as_view(), &prior).expect("explained");
    let got = tracker.track(&img.as_view(), &prior).expect("tracked");
    assert_eq!(trace.result, got);
    for (st, smp) in trace.measure.iter().zip(&got.samples) {
        assert_eq!(
            (st.hit, st.wall, st.margin),
            (smp.hit, smp.wall, smp.margin)
        );
    }
    let gap = got
        .samples
        .iter()
        .filter(|s| matches!(s.hit, Err(BeadReject::Absent | BeadReject::Caliper(_))))
        .count();
    assert!(
        gap > 3,
        "the gap's stations are absent: {:?}",
        got.summary.rejects
    );
    // The first pass and the final stage compute margins; the passes between do not.
    assert!(trace.passes[0].stations.iter().any(|s| s.margin.is_some()));
    assert!(
        trace.passes[1..]
            .iter()
            .all(|p| p.stations.iter().all(|s| s.margin.is_none()))
    );
}

#[test]
fn a_wall_that_leaves_the_image_is_measured_alone() {
    // Running down and out of the image's bottom edge at y = 279: the +n wall, 25 px
    // below the centreline, leaves first.
    let (a, b) = (P2::new(30.0, 230.0), P2::new(330.0, 300.0));
    let ribbon = Ribbon::new(Curve::line(a, b), Width::Const(50.0));
    let img = render(&ribbon).to_u8();
    let prior = prior(
        &ribbon,
        10.0,
        ribbon.length() - 10.0,
        P2::new(0.0, 0.0),
        |_| 0.0,
    );
    let got = track(coupled(), &img, &prior);
    let per_station = track(BeadConfig::default(), &img, &prior);
    let bottom = (SIZE.1 - 1) as f64;
    let (mut one_sided, mut worst, mut lost) = (0, 0.0f64, 0);
    for (smp, old) in got.samples.iter().zip(&per_station.samples) {
        let (s, _) = ribbon.nearest(smp.point);
        let c = ribbon.curve.center(s);
        let n = ribbon.curve.normal(s);
        // Where the +n wall is off the image by a margin and the −n wall on it by one.
        let second_y = (c + n * 25.0).y;
        let first_y = (c - n * 25.0).y;
        if second_y < bottom + 4.0 || first_y > bottom - 6.0 {
            continue;
        }
        lost += usize::from(old.hit.is_err());
        assert_eq!(smp.hit, Err(BeadReject::OneSided), "s = {s:.1}");
        let wall = smp.wall.expect("the visible wall");
        assert_eq!(wall.side, BeadSide::First);
        let (_, dw) = ribbon.nearest(wall.edge.p);
        worst = worst.max((dw + 25.0).abs());
        one_sided += 1;
    }
    eprintln!(
        "one wall off the image: {one_sided} one-sided stations (per station: {lost} lost), \
         worst visible-wall error {worst:.4} px"
    );
    assert!(one_sided >= 10, "{one_sided} one-sided stations");
    assert_eq!(
        lost, one_sided,
        "per-station pairing loses every one of them"
    );
    assert!(worst < 0.1, "the visible wall is off by {worst} px");
}

/// A light bead 8 px wide along a line, with an identical one `twin` px beside it when
/// set; the coupled tracker's first-pass margin from a prior `at` px off the first.
fn first_pass_margin(twin: Option<f64>, at: f64) -> f32 {
    let ribbon = Ribbon::new(line(), Width::Const(8.0));
    let mut scene = Scene::new(SIZE.0, SIZE.1, ribbon.clone(), BG, FG, SIGMA);
    if let Some(offset) = twin {
        scene = scene.parallel(Stripe {
            offset,
            width: 8.0,
            contrast: FG - BG,
        });
    }
    let img = scene.render().to_u8();
    let len = ribbon.length();
    let prior = prior(&ribbon, 15.0, len - 15.0, P2::new(0.0, 0.0), |_| at);
    let d = BeadConfig::default();
    let cfg = BeadConfig {
        min_width: 4.0,
        max_width: 12.0,
        track: BeadCaliper {
            max_offset: 10.0,
            ..d.track
        },
        ..coupled()
    };
    let got = track(cfg, &img, &prior);
    got.track.passes[0]
        .path
        .and_then(|p| p.margin)
        .expect("a present path")
}

#[test]
fn a_parallel_bead_midway_lowers_the_first_pass_margin() {
    let alone = first_pass_margin(None, 7.0);
    let twin = first_pass_margin(Some(14.0), 7.0);
    let near = first_pass_margin(Some(14.0), 0.0);
    eprintln!(
        "first-pass margin: {alone:.2} alone, {twin:.2} midway between two, {near:.2} on one"
    );
    assert!(twin < 1.0, "midway between two beads: {twin}");
    assert!(alone > 4.0 && near > twin, "{alone} / {near}");
}
