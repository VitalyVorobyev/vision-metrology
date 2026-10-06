//! The bead fixture (`examples/common/ribbon.rs`) and the checks that pin it.
//!
//! The fixture renders a ribbon of known width along a centreline: a line, an arc, an
//! S-bend of two tangent arcs, or a sine. The ribbon is a box across the normal, blurred
//! by a 1-D Gaussian and averaged over each pixel by 4 × 4 Gauss–Legendre quadrature, with
//! optional gaps, highlight stripes, parallel distractor stripes, straight distractor steps,
//! a background gradient, and seeded noise quantised to `u8`, `u16` or `f32`. Its truth is
//! continuous: a point maps to an arc length `s` and a signed distance `d` along the normal
//! `(−t_y, t_x)`, and the edges sit at `d = ±w(s)/2` exactly.
//!
//! - **Closed form.** A straight ribbon is a bar, the difference of two straight steps, and
//!   for a straight edge the 1-D normal blur is the isotropic PSF. So it must equal
//!   `strip.rs`'s exact pixel mean, over angles, blurs, widths and subpixel phases.
//! - **Truth geometry.** `nearest` inverts `center(s) + d·normal(s)` for every curve kind,
//!   the sine's arc length matches a dense polyline, and the S-bend is C¹ at its join.
//! - **Rendering.** The bead level, its edges, gaps, square ends and a distractor step land
//!   where the truth says, and a scene renders bit-identically for the same seed.

use std::f64::consts::TAU;

use super::ribbon::{Curve, P2, Ribbon, Scene, Step, Stripe, Width};
use super::strip::{HI, LO, render_steps};

/// The straight-ribbon cross-check's worst pixel, in DN at 160 DN contrast. Measured:
/// 6.6e-5 DN, the 4 × 4 quadrature's error (a 6 × 6 rule gives 1.9e-6 DN). About 1.5× that.
const CLOSED_FORM_TOL_DN: f64 = 1e-4;

#[test]
fn a_straight_ribbon_matches_the_closed_form() {
    const SIZE: usize = 64;
    let mut worst = 0.0f64;
    for &angle in &[0.0f64, 17.0, 45.0, 108.0] {
        let t = P2::polar(angle.to_radians());
        let n = t.perp();
        for &sigma in &[0.6, 1.2] {
            for &w in &[3.0, 8.5] {
                for &phase in &[0.0, 0.3, 0.55, 0.8] {
                    // The ends are 100 px out, well off the image.
                    let c = P2::new(32.0, 32.0) + n * phase;
                    let line = Curve::line(c - t * 100.0, c + t * 100.0);
                    let ribbon = Ribbon::new(line, Width::Const(w));
                    let got = Scene::new(SIZE, SIZE, ribbon, LO, HI, sigma).render();
                    let mid = n.dot(c);
                    let steps = [(mid - 0.5 * w, 1.0), (mid + 0.5 * w, -1.0)];
                    let want = render_steps(SIZE, (n.x, n.y), &steps, sigma);
                    for (g, e) in got.data.iter().zip(&want) {
                        worst = worst.max((g - e).abs());
                    }
                }
            }
        }
    }
    eprintln!(
        "ribbon vs closed form: max |Δ| {worst:.2e} DN at {} DN contrast",
        HI - LO
    );
    assert!(
        worst < CLOSED_FORM_TOL_DN,
        "a straight ribbon must equal the closed-form bar, worst pixel {worst:.2e} DN"
    );
}

// ── truth geometry ───────────────────────────────────────────────────────

fn sine_params() -> (P2, P2, f64, f64, f64) {
    (P2::new(15.0, 90.0), P2::polar(0.15), 230.0, 10.0, 120.0)
}

/// One curve of each kind, with the round-trip tolerance its projection meets.
fn curves() -> Vec<(&'static str, Curve, f64)> {
    let a = P2::new(10.3, 20.7);
    let (origin, dir, span, amplitude, period) = sine_params();
    vec![
        (
            "line",
            Curve::line(a, a + P2::polar(17f64.to_radians()) * 150.0),
            1e-9,
        ),
        (
            "clockwise arc",
            Curve::arc(P2::new(128.4, 127.6), 60.0, 0.3, 2.0),
            1e-9,
        ),
        (
            "anticlockwise arc",
            Curve::arc(P2::new(128.4, 127.6), 45.0, 2.9, -2.5),
            1e-9,
        ),
        (
            "S-bend",
            Curve::s_bend(P2::new(20.0, 60.0), 0.2, 50.0, 1.2),
            1e-9,
        ),
        (
            "sine",
            Curve::sine(origin, dir, span, amplitude, period),
            1e-6,
        ),
    ]
}

#[test]
fn nearest_inverts_the_normal_offset() {
    // Every |d| stays below each curve's smallest radius of curvature (the sine's is 36 px).
    for (name, curve, tol) in curves() {
        let len = curve.length();
        let (mut ds, mut dd) = (0.0f64, 0.0f64);
        for k in 0..=20 {
            let s = len * (0.02 + 0.96 * f64::from(k) / 20.0);
            for &d in &[-15.0, -3.3, 0.0, 2.5, 15.0] {
                let p = curve.center(s) + curve.normal(s) * d;
                let (s2, d2) = curve.nearest(p);
                ds = ds.max((s2 - s).abs());
                dd = dd.max((d2 - d).abs());
            }
        }
        eprintln!("{name}: max |Δs| {ds:.1e}, |Δd| {dd:.1e}");
        assert!(
            ds < tol && dd < tol,
            "{name}: nearest(center(s) + d·normal(s)) must return (s, d), off by |Δs| {ds:.1e}, \
             |Δd| {dd:.1e}"
        );
    }
}

#[test]
fn signs_follow_the_screen() {
    // Travelling +x with y down, +n points down: to the right of travel on screen.
    let line = Curve::line(P2::new(0.0, 0.0), P2::new(10.0, 0.0));
    assert_eq!(line.normal(5.0), P2::new(-0.0, 1.0));
    assert_eq!(line.nearest(P2::new(4.0, 2.0)), (4.0, 2.0));
    // Beyond an end, s clamps and d is the normal component.
    assert_eq!(line.nearest(P2::new(13.0, -1.5)), (10.0, -1.5));
    assert_eq!(line.nearest(P2::new(-2.0, 0.5)), (0.0, 0.5));
    // A positive sweep turns clockwise on screen, with +n towards the centre; a negative
    // one turns anticlockwise, with +n away from it.
    let centre = P2::new(50.0, 50.0);
    for (sweep, inward) in [(1.0, 1.0), (-1.0, -1.0)] {
        let arc = Curve::arc(centre, 20.0, 0.0, sweep);
        let s = 10.0;
        let toward = (centre - arc.center(s)).dot(arc.normal(s));
        assert!(
            (toward - inward * 20.0).abs() < 1e-12,
            "sweep {sweep}: {toward}"
        );
        let below = arc.center(s).y > 50.0;
        assert_eq!(below, sweep > 0.0, "sweep {sweep} turns the wrong way");
    }
}

#[test]
fn the_sine_arc_length_matches_a_dense_polyline() {
    let (origin, dir, span, amplitude, period) = sine_params();
    let curve = Curve::sine(origin, dir, span, amplitude, period);
    let at = |u: f64| origin + dir * u + dir.perp() * (amplitude * (TAU * u / period).sin());
    let n = 200_000;
    let mut len = 0.0;
    let mut prev = at(0.0);
    for i in 1..=n {
        let q = at(span * f64::from(i) / f64::from(n));
        len += (q - prev).norm();
        prev = q;
        // Partway, the arc length of a point on the curve.
        if i % 40_000 == 0 && i < n {
            let (s, d) = curve.nearest(q);
            assert!(
                (s - len).abs() < 1e-6 && d.abs() < 1e-9,
                "u {}: s {s} vs polyline {len}, d {d}",
                span * f64::from(i) / f64::from(n)
            );
        }
    }
    let err = (curve.length() - len).abs();
    eprintln!("sine length {:.6} px, polyline {len:.6}", curve.length());
    assert!(
        err < 1e-6,
        "sine arc length off the polyline's by {err:.1e} px"
    );
}

#[test]
fn the_s_bend_is_c1_at_its_join() {
    let curve = Curve::s_bend(P2::new(20.0, 60.0), 0.2, 50.0, 1.2);
    let Curve::SBend([first, second]) = &curve else {
        panic!("an S-bend is two arcs");
    };
    let join = first.length();
    let gap = (first.at(join) - second.at(0.0)).norm();
    let kink = (first.tangent(join) - second.tangent(0.0)).norm();
    assert!(
        gap < 1e-12 && kink < 1e-12,
        "join: gap {gap:.1e}, kink {kink:.1e}"
    );
    // The two centres of curvature lie on opposite sides of the join.
    let n = first.tangent(join).perp();
    let (c1, c2) = (
        (first.center - first.at(join)).dot(n),
        (second.center - second.at(0.0)).dot(n),
    );
    assert!(c1 * c2 < 0.0, "both arcs turn the same way: {c1}, {c2}");
    // Through the curve's own API, either side of the join.
    let eps = 1e-6;
    let step = (curve.center(join + eps) - curve.center(join - eps)).norm();
    let turn = (curve.tangent(join + eps) - curve.tangent(join - eps)).norm();
    assert!(
        (step - 2.0 * eps).abs() < 1e-9 && turn < 1e-6,
        "step {step}, turn {turn}"
    );
    // The heading comes back.
    let back = (curve.tangent(0.0) - curve.tangent(curve.length())).norm();
    assert!(back < 1e-12, "end heading differs by {back:.1e}");
}

// ── rendering ────────────────────────────────────────────────────────────

#[test]
fn levels_edges_gaps_and_ends_land_where_the_truth_says() {
    // A light bead along y = 40 from x = −10 to 70, so s = x + 10; its width grows from 8 to
    // 16, 14 px at x = 50. No bead over s ∈ [30, 40], x ∈ [20, 30]. A step at x = 85 is
    // brighter to its right; a stripe parallel to the bead sits 25 px below it.
    let line = Curve::line(P2::new(-10.0, 40.0), P2::new(70.0, 40.0));
    let ribbon = Ribbon::new(line, Width::Linear { w0: 8.0, w1: 16.0 }).gap(30.0, 40.0);
    let scene = Scene::new(96, 80, ribbon, 50.0, 180.0, 0.8)
        .gradient(P2::new(48.0, 40.0), P2::new(0.25, -0.1))
        .parallel(Stripe {
            offset: 25.0,
            width: 8.0,
            contrast: 40.0,
        })
        .step(Step {
            point: P2::new(85.0, 0.0),
            normal: P2::new(1.0, 0.0),
            contrast: 25.0,
        });
    let img = scene.render();
    let above = |x: usize, y: usize| img.at(x, y) - scene.background(P2::new(x as f64, y as f64));

    assert!((scene.ribbon.width(60.0) - 14.0).abs() < 1e-12);
    assert!(
        (above(50, 40) - 130.0).abs() < 1e-6,
        "centre {}",
        above(50, 40)
    );
    // A pixel centred on an edge (d = ±7) averages to half the contrast.
    for y in [33, 47] {
        assert!(
            (above(50, y) - 65.0).abs() < 1e-3,
            "edge y {y}: {}",
            above(50, y)
        );
    }
    assert!(
        above(50, 54).abs() < 1e-6,
        "beside the bead: {}",
        above(50, 54)
    );
    // The gap and the square end leave background; the parallel stripe goes on in the gap.
    assert_eq!(above(25, 40), 0.0, "in the gap");
    assert_eq!(above(75, 40), 0.0, "past the end");
    assert!(
        (above(25, 65) - 40.0).abs() < 1e-3,
        "stripe {}",
        above(25, 65)
    );
    // The step: on it, half its contrast; well to either side, all or nothing.
    assert!(
        (above(85, 5) - 12.5).abs() < 1e-6,
        "on the step: {}",
        above(85, 5)
    );
    assert!(
        (above(93, 5) - 25.0).abs() < 1e-9,
        "right of the step: {}",
        above(93, 5)
    );
    assert_eq!(above(77, 5), 0.0, "left of the step");
    // A dark bead is the same with the levels swapped.
    let dark = Scene {
        fg: 10.0,
        ..scene.clone()
    }
    .render();
    let below = dark.at(50, 40) - scene.background(P2::new(50.0, 40.0));
    assert!((below + 40.0).abs() < 1e-6, "dark centre {below}");
}

#[test]
fn a_scene_renders_bit_identically_for_one_seed() {
    // Everything at once on a sine that runs off the image.
    let (_, dir, _, amplitude, period) = sine_params();
    let sine = Curve::sine(P2::new(-20.0, 140.0), dir, 300.0, amplitude, period);
    let ribbon = Ribbon::new(sine, Width::Linear { w0: 6.0, w1: 14.0 }).gap(60.0, 75.0);
    let scene = Scene::new(256, 256, ribbon, 60.0, 30.0, 1.0)
        .gradient(P2::new(128.0, 128.0), P2::new(0.1, -0.05))
        .highlight(Stripe {
            offset: 1.5,
            width: 2.0,
            contrast: 25.0,
        })
        .parallel(Stripe {
            offset: 14.0,
            width: 3.0,
            contrast: 40.0,
        })
        .step(Step {
            point: P2::new(170.0, 0.0),
            normal: P2::polar(0.3),
            contrast: 20.0,
        });
    let clean = scene.render();
    let a = clean.noisy(3.0, 7);
    let b = scene.render().noisy(3.0, 7);
    let bits = |r: &[f64]| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.data), bits(&b.data));
    assert_eq!(a.to_u8(), b.to_u8());
    assert_eq!(a.to_u16(), b.to_u16());
    let f32_bits = |r: &[f32]| r.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(f32_bits(a.to_f32().data()), f32_bits(b.to_f32().data()));
    // Another seed is other noise.
    assert_ne!(a.to_u8(), clean.noisy(3.0, 8).to_u8());
    // The three quantisations agree: u8 rounds the DN, u16 holds 256 × DN.
    let (u8s, u16s, f32s) = (a.to_u8(), a.to_u16(), a.to_f32());
    for ((&q8, &q16), &v) in u8s.data().iter().zip(u16s.data()).zip(f32s.data()) {
        if (1.0..254.0).contains(&v) {
            assert!((f32::from(q8) - v).abs() <= 0.5 && (f32::from(q16) / 256.0 - v).abs() <= 0.01);
        }
    }
}
