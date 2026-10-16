//! Unit tests for the coupled path on profiles given directly: the occlusion and one-sided
//! case of issue #73, spacing independence, polarity, and the margins.

use vm_primitives::{Edge1DConfig, Edge1DDetector, EdgePolarity};

use super::config::BeadPolarity;
use super::coupled::{ABSENT, BOTH, CoupledPath, Evidence, FIRST, SECOND, Setup, snap};
use super::coupling::BeadCoupling;

/// The profile grid of #73: `x = −40, −39.75, …, 40`.
const STEP: f64 = 0.25;
const HALF: f64 = 40.0;

fn logistic(z: f64) -> f64 {
    1.0 / (1.0 + (-z / 0.5).exp())
}

/// #73's intensity at `x` on station `k`: a light bead of half-width `12 + 0.04k`, a
/// sharper internal reversal between 2 and 5, and a broad exterior bump between 23 and 35.
fn issue73(x: f64, k: usize) -> f64 {
    let w = 12.0 + 0.04 * k as f64;
    20.0 + 180.0 * logistic(x + w) * logistic(w - x)
        - 140.0 * logistic(2.0 * (x - 2.0)) * logistic(2.0 * (5.0 - x))
        + 40.0 * logistic((x - 23.0) / 4.0) * logistic((35.0 - x) / 4.0)
}

/// One station's profile as a caliper would see it: sampled on the grid, clamped past the
/// last valid sample, differentiated by a DoG of σ 1 px, with the edges above
/// `threshold` DN/px. Returns the response per px, its validity, and the edges as
/// `(offset, polarity)`.
struct Profiled {
    resp: Vec<f32>,
    valid: Vec<bool>,
    edges: Vec<(f64, EdgePolarity)>,
}

fn profile(
    f: impl Fn(f64) -> f64,
    valid: impl Fn(f64) -> bool,
    det: &mut Edge1DDetector,
) -> Profiled {
    let n = (2.0 * HALF / STEP).round() as usize + 1;
    let xs: Vec<f64> = (0..n).map(|i| -HALF + i as f64 * STEP).collect();
    let ok: Vec<bool> = xs.iter().map(|&x| valid(x)).collect();
    // Past an invalid sample the image is not there: clamp to the last valid value, as a
    // caliper's border fill does.
    let mut last = f(xs[0]);
    let signal: Vec<f32> = xs
        .iter()
        .zip(&ok)
        .map(|(&x, &v)| {
            if v {
                last = f(x);
            }
            last as f32
        })
        .collect();
    let cfg = Edge1DConfig {
        sigma: (1.0 / STEP) as f32,
        pos_thresh: (3.0 * STEP) as f32,
        neg_thresh: (3.0 * STEP) as f32,
        ..Edge1DConfig::default()
    };
    let peaks = det.detect_in_ref(&signal, &cfg).to_vec();
    let edges = peaks
        .iter()
        .map(|p| (-HALF + f64::from(p.x) * STEP, p.polarity))
        .filter(|&(o, _)| valid(o))
        .collect();
    let resp = det
        .response()
        .iter()
        .map(|&r| (f64::from(r) / STEP) as f32)
        .collect();
    Profiled {
        resp,
        valid: ok,
        edges,
    }
}

fn setup(h: f64) -> Setup {
    Setup {
        polarity: BeadPolarity::Light,
        min_width: 8.0,
        max_width: 40.0,
        reach: 8.0,
        h,
        clearance: Some(4.0),
        margins: true,
    }
}

/// The walls chosen and refined at each station: `(plane, first, second)`, a wall `None`
/// where it is not measured.
fn solve(
    profiles: &[Profiled],
    s: &Setup,
    cfg: &BeadCoupling,
) -> Vec<(usize, Option<f64>, Option<f64>)> {
    let mut ev = Evidence::default();
    ev.reset(profiles[0].resp.len(), STEP, -HALF);
    for p in profiles {
        ev.resp.extend_from_slice(&p.resp);
        ev.valid.extend_from_slice(&p.valid);
    }
    let windows = vec![(-s.reach, s.reach); profiles.len()];
    let mut path = CoupledPath::default();
    path.extract(&ev, &windows, s, cfg);
    let r = path.snap_radius();
    path.choices
        .iter()
        .zip(profiles)
        .map(|(c, p)| {
            let at = |t: f64, pol| snap(p.edges.iter().copied(), t, pol, r).map(|k| p.edges[k].0);
            let l = (c.plane == BOTH || c.plane == FIRST)
                .then(|| at(c.first, EdgePolarity::Rising))
                .flatten();
            let rr = (c.plane == BOTH || c.plane == SECOND)
                .then(|| at(c.second, EdgePolarity::Falling))
                .flatten();
            (c.plane, l, rr)
        })
        .collect()
}

#[test]
fn issue73_occlusion_and_one_sided_walls() {
    let mut det = Edge1DDetector::new(4.0);
    let profiles: Vec<Profiled> = (0..60)
        .map(|k| {
            let valid = move |x: f64| !(25..=28).contains(&k) && !(k >= 45 && x > 10.0);
            profile(|x| issue73(x, k), valid, &mut det)
        })
        .collect();
    let got = solve(&profiles, &setup(1.0), &BeadCoupling::default());
    let (mut worst, mut worst_one) = (0.0f64, 0.0f64);
    for (k, &(plane, l, r)) in got.iter().enumerate() {
        let w = 12.0 + 0.04 * k as f64;
        if (25..=28).contains(&k) {
            assert_eq!(
                plane, ABSENT,
                "station {k}: every sample is occluded, got plane {plane}"
            );
            continue;
        }
        if k >= 45 {
            assert_eq!(
                plane, FIRST,
                "station {k}: only the −n wall is visible, got plane {plane}"
            );
            assert!(r.is_none());
        } else {
            assert_eq!(
                plane, BOTH,
                "station {k}: both walls are visible, got plane {plane}"
            );
        }
        // Three stations next to each end of the curve and of each occluded run are
        // excluded from the error bound, as the issue's prototype was.
        let edge = [0, 24, 29, 45, 59]
            .iter()
            .any(|&e: &usize| k.abs_diff(e) < 3);
        let el = (l.expect("the −n wall is measured") + w).abs();
        if let Some(r) = r {
            let er = (r - w).abs();
            if !edge {
                worst = worst.max(el).max(er);
            }
        } else if !edge {
            worst_one = worst_one.max(el);
        }
    }
    eprintln!(
        "issue 73: worst wall error {worst:.3} px with both walls, {worst_one:.3} px with one"
    );
    assert!(
        worst <= 0.25 && worst_one <= 0.25,
        "walls off by {worst:.3} / {worst_one:.3} px"
    );
}

/// A clean light bead, its walls at `±w/2` about `c(k)`, on stations `h` px apart.
fn clean(k_count: usize, h: f64, det: &mut Edge1DDetector) -> Vec<Profiled> {
    (0..k_count)
        .map(|k| {
            let s = k as f64 * h;
            let c = 2.0 * (s / 40.0).sin();
            let w = 10.0 + 0.05 * s;
            let f =
                move |x: f64| 30.0 + 150.0 * logistic(x - c + 0.5 * w) * logistic(c + 0.5 * w - x);
            profile(f, |_| true, det)
        })
        .collect()
}

#[test]
fn the_walls_do_not_depend_on_the_station_spacing() {
    let mut det = Edge1DDetector::new(4.0);
    let cfg = BeadCoupling::default();
    let coarse = solve(&clean(41, 2.0, &mut det), &setup(2.0), &cfg);
    let fine = solve(&clean(81, 1.0, &mut det), &setup(1.0), &cfg);
    // The end stations, where the path has one neighbour, are left out.
    let n = coarse.len();
    for (k, (a, b)) in coarse.iter().zip(fine.iter().step_by(2)).enumerate() {
        if k == 0 || k + 1 == n {
            continue;
        }
        assert_eq!(a.0, BOTH, "station {k}");
        assert_eq!(
            a, b,
            "station {k}: spacing 2 px chose {a:?}, spacing 1 px {b:?}"
        );
    }
}

#[test]
fn a_dark_bead_is_the_mirror_of_a_light_one() {
    let mut det = Edge1DDetector::new(4.0);
    let light = clean(30, 2.0, &mut det);
    let dark: Vec<Profiled> = light
        .iter()
        .map(|p| Profiled {
            resp: p.resp.iter().map(|&r| -r).collect(),
            valid: p.valid.clone(),
            edges: p
                .edges
                .iter()
                .map(|&(o, pol)| {
                    (
                        o,
                        match pol {
                            EdgePolarity::Rising => EdgePolarity::Falling,
                            EdgePolarity::Falling => EdgePolarity::Rising,
                        },
                    )
                })
                .collect(),
        })
        .collect();
    let cfg = BeadCoupling::default();
    let a = solve(&light, &setup(2.0), &cfg);
    let mut s = setup(2.0);
    s.polarity = BeadPolarity::Dark;
    let mut ev = Evidence::default();
    ev.reset(dark[0].resp.len(), STEP, -HALF);
    for p in &dark {
        ev.resp.extend_from_slice(&p.resp);
        ev.valid.extend_from_slice(&p.valid);
    }
    let mut path = CoupledPath::default();
    path.extract(&ev, &vec![(-8.0, 8.0); dark.len()], &s, &cfg);
    let mut light_path = CoupledPath::default();
    let mut ev_l = Evidence::default();
    ev_l.reset(light[0].resp.len(), STEP, -HALF);
    for p in &light {
        ev_l.resp.extend_from_slice(&p.resp);
        ev_l.valid.extend_from_slice(&p.valid);
    }
    light_path.extract(&ev_l, &vec![(-8.0, 8.0); light.len()], &setup(2.0), &cfg);
    assert_eq!(path.choices, light_path.choices);
    assert!(a.iter().all(|c| c.0 == BOTH));
}

#[test]
fn a_parallel_line_lowers_the_margin() {
    let mut det = Edge1DDetector::new(4.0);
    let bead = |x: f64, c: f64| logistic(x - c + 4.0) * logistic(c + 4.0 - x);
    let run = |twin: bool, det: &mut Edge1DDetector| {
        let profiles: Vec<Profiled> = (0..40)
            .map(|_| {
                let f = move |x: f64| {
                    30.0 + 150.0 * bead(x, -7.0) + if twin { 150.0 * bead(x, 7.0) } else { 0.0 }
                };
                profile(f, |_| true, det)
            })
            .collect();
        let mut ev = Evidence::default();
        ev.reset(profiles[0].resp.len(), STEP, -HALF);
        for p in &profiles {
            ev.resp.extend_from_slice(&p.resp);
            ev.valid.extend_from_slice(&p.valid);
        }
        let mut s = setup(2.0);
        s.min_width = 4.0;
        s.max_width = 12.0;
        s.reach = 12.0;
        let mut path = CoupledPath::default();
        let out = path.extract(&ev, &vec![(-12.0, 12.0); 40], &s, &BeadCoupling::default());
        out.margin.expect("present")
    };
    let alone = run(false, &mut det);
    let twin = run(true, &mut det);
    eprintln!("whole-call margin: {alone:.2} alone, {twin:.2} with a parallel bead");
    assert!(
        twin < 0.1 * alone,
        "a parallel bead midway must lower the margin: {twin} vs {alone}"
    );
}

#[test]
fn coupled_configs_are_validated() {
    use super::BeadConfig;
    use super::config::validate;
    use super::coupling::BeadPairing;
    let ok = BeadConfig {
        pairing: BeadPairing::Coupled(BeadCoupling::default()),
        clearance: Some(3.0),
        ..BeadConfig::default()
    };
    assert!(
        validate(&ok).is_ok(),
        "clearance is a cost with coupled pairing"
    );
    let bad = |f: fn(&mut BeadConfig, &mut BeadCoupling)| {
        let (mut cfg, mut c) = (ok, BeadCoupling::default());
        f(&mut cfg, &mut c);
        cfg.pairing = BeadPairing::Coupled(c);
        validate(&cfg).is_err()
    };
    assert!(bad(|cfg, _| cfg.min_margin = Some(0.2)));
    assert!(bad(|_, c| c.saturation = 0.0));
    assert!(bad(|_, c| c.absent_level = f32::NAN));
    assert!(bad(|_, c| c.bin_px = -1.0));
    assert!(bad(|_, c| c.margin_quantile = 0.0));
    assert!(bad(|_, c| c.margin_quantile = 1.5));
    assert!(bad(|_, c| c.max_wall_slope = 0.0));
    // A grid with too many cells for its reach and width range.
    assert!(bad(|cfg, _| cfg.max_width = 10_000.0));
}
