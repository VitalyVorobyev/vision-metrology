//! Convergence basins: how far a prior may be off the bead and still converge.
//!
//! The fixtures are the arc (R = 150 px) and the sine of `tests/bead.rs` with a bead 30 px
//! wide, blurred by σ = 1.2 px, under 2 DN of seeded noise. They are tracked with the
//! default config, except that `min_width` is 20 px: the default 30 px would sit on the
//! bead's own width. A prior is the truth from 15 px in from each end, a vertex every
//! 4 px, perturbed by one of:
//! - a translation of `m` px along the true normal (a parallel curve);
//! - a rotation of `m` degrees about the prior's midpoint;
//! - a sine of amplitude `m` px along the normal, at a wavelength of `L/2` or `L/4`;
//! - a Gaussian bump of `m` px at the middle, σ_b = 5 or 15 px of arc;
//! - a 5 px translation plus the rotation, or plus the σ_b = 15 bump.
//!
//! A prior has converged when every refined station with truth is within
//! [`CONVERGED_PX`] of the true centreline after the configured passes. A basin is the
//! largest magnitude that converges: [`basin`] walks up in coarse steps to the first
//! failure, then bisects. The pinned basins must not shrink by more than 10%.
//!
//! - **Reach.** A translation within `track.max_offset` converges; one beyond
//!   `max_offset + max_width/2` plus a margin finds nothing, and at no translation does a
//!   station take anything but the bead for a hit.
//! - **The bending length.** An ignored measurement, `measure_the_bending_length`, sweeps
//!   `bending_px` and the pass count, and a coarse-to-fine schedule of bending lengths, over
//!   the local-deformation basins and the noise they let through. Run it in release with
//!   `-- --ignored --nocapture`.

use std::f64::consts::TAU;
use std::num::NonZeroUsize;

use super::ribbon::{Curve, P2, Ribbon, Scene, Width};
use super::{BG, FG, SIZE, arc, errors, line, prior, sine};
use vision_metrology::measure::{BeadConfig, BeadStop, BeadTracker, BeadTuning, TrackedBead};
use vision_metrology::{Image, Point2f};

const WIDTH: f64 = 30.0;
const SIGMA: f64 = 1.2;
const NOISE_DN: f64 = 2.0;
/// A prior has converged when no refined station with truth is further than this, px,
/// from the true centreline.
const CONVERGED_PX: f64 = 0.1;
/// The fixed translation of the combined perturbations, px.
const SHIFT: f64 = 5.0;

struct Fixture {
    name: &'static str,
    ribbon: Ribbon,
    img: Image<u8>,
}

fn fixture(name: &'static str, curve: Curve, noise_dn: f64, seed: u64) -> Fixture {
    let ribbon = Ribbon::new(curve, Width::Const(WIDTH));
    let img = Scene::new(SIZE.0, SIZE.1, ribbon.clone(), BG, FG, SIGMA)
        .render()
        .noisy(noise_dn, seed)
        .to_u8();
    Fixture { name, ribbon, img }
}

fn fixtures() -> [Fixture; 2] {
    [
        fixture("arc", arc(), NOISE_DN, 23),
        fixture("sine", sine(), NOISE_DN, 29),
    ]
}

/// The default config, with a width range that holds the 30 px bead with room.
fn config() -> BeadConfig {
    BeadConfig {
        min_width: 20.0,
        ..BeadConfig::default()
    }
}

/// [`config`] with `bending_px` and `passes`.
fn bending(bending_px: f32, passes: usize) -> BeadConfig {
    let d = config();
    BeadConfig {
        tuning: BeadTuning {
            bending_px,
            passes: NonZeroUsize::new(passes).expect("at least one pass"),
            ..d.tuning
        },
        ..d
    }
}

/// How a prior is moved off the truth; the magnitude `m` is the basin's variable.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Perturbation {
    /// `m` px along the true normal.
    Translate,
    /// `m` degrees about the prior's midpoint.
    Rotate,
    /// `m·sin(2π·waves·s/L)` px along the normal.
    Sine { waves: f64 },
    /// `m·exp(−(s − L/2)²/(2σ_b²))` px along the normal.
    Bump { sigma: f64 },
    /// [`SHIFT`] px along the normal, then `m` degrees about the midpoint.
    ShiftRotate,
    /// [`SHIFT`] px along the normal plus a σ_b = 15 px bump of `m` px.
    ShiftBump,
}

impl Perturbation {
    /// The perturbed prior, from 15 px in from each end.
    fn prior(self, ribbon: &Ribbon, m: f64) -> Vec<Point2f> {
        let len = ribbon.length();
        let bump = |a: f64, sigma: f64, s: f64| {
            a * (-(s - 0.5 * len).powi(2) / (2.0 * sigma * sigma)).exp()
        };
        let bend = |s: f64| match self {
            Self::Translate => m,
            Self::Rotate => 0.0,
            Self::Sine { waves } => m * (TAU * waves * s / len).sin(),
            Self::Bump { sigma } => bump(m, sigma, s),
            Self::ShiftRotate => SHIFT,
            Self::ShiftBump => SHIFT + bump(m, 15.0, s),
        };
        let pts = prior(ribbon, 15.0, len - 15.0, P2::new(0.0, 0.0), bend);
        if !matches!(self, Self::Rotate | Self::ShiftRotate) {
            return pts;
        }
        let mid = pts[pts.len() / 2];
        let (s, c) = (m.to_radians() as f32).sin_cos();
        pts.iter()
            .map(|p| {
                let v = p - mid;
                Point2f::new(mid.x + c * v.x - s * v.y, mid.y + s * v.x + c * v.y)
            })
            .collect()
    }

    /// The walk: coarse step, resolution and the largest magnitude tried.
    fn walk(self) -> (f64, f64, f64) {
        match self {
            Self::Translate | Self::Rotate | Self::ShiftRotate => (1.0, 0.1, 40.0),
            Self::Sine { .. } | Self::Bump { .. } | Self::ShiftBump => (0.5, 0.05, 30.0),
        }
    }
}

/// The truth as a coarse polygon: a vertex every `spacing` px of arc from 15 px in.
fn polygon(ribbon: &Ribbon, spacing: f64) -> Vec<Point2f> {
    let len = ribbon.length();
    let n = ((len - 30.0) / spacing).round().max(1.0) as usize;
    (0..=n)
        .map(|k| {
            ribbon
                .curve
                .center(15.0 + (len - 30.0) * k as f64 / n as f64)
                .point()
        })
        .collect()
}

/// A way to track: one `track` call, or a schedule of calls, each result the next prior.
struct Method {
    label: String,
    trackers: Vec<BeadTracker>,
}

impl Method {
    fn single(label: impl Into<String>, cfg: BeadConfig) -> Self {
        Self::schedule(label, &[cfg])
    }

    fn schedule(label: impl Into<String>, cfgs: &[BeadConfig]) -> Self {
        Self {
            label: label.into(),
            trackers: cfgs
                .iter()
                .map(|&c| BeadTracker::new(c).expect("a valid config"))
                .collect(),
        }
    }

    fn track(&mut self, img: &Image<u8>, prior: &[Point2f]) -> TrackedBead {
        let mut prior = prior.to_vec();
        let mut last = None;
        for t in &mut self.trackers {
            let got = t.track(&img.as_view(), &prior).expect("a trackable prior");
            prior.clone_from(&got.centerline);
            last = Some(got);
        }
        last.expect("at least one stage")
    }

    /// Whether this method brings `prior` onto the bead of `f`.
    fn converges(&mut self, f: &Fixture, prior: &[Point2f]) -> bool {
        let got = self.track(&f.img, prior);
        let (center, _, n) = errors(&got, &f.ribbon);
        n > 0 && got.track.stop != BeadStop::TooFewValid && center < CONVERGED_PX
    }
}

/// The largest magnitude that converges with every coarse step below it: a walk up by
/// `coarse` to `max`, then a bisection between the last success and the first failure
/// down to `resolution`.
fn basin(coarse: f64, resolution: f64, max: f64, mut ok: impl FnMut(f64) -> bool) -> f64 {
    let mut good = 0.0;
    let mut m = coarse;
    let mut bad = loop {
        if m > max + 1e-9 {
            return good;
        }
        if !ok(m) {
            break m;
        }
        good = m;
        m += coarse;
    };
    while bad - good > resolution + 1e-9 {
        let mid = 0.5 * (good + bad);
        if ok(mid) {
            good = mid;
        } else {
            bad = mid;
        }
    }
    good
}

/// The basin of `p` on `f` for `method`.
fn basin_of(f: &Fixture, method: &mut Method, p: Perturbation) -> f64 {
    let (coarse, resolution, max) = p.walk();
    basin(coarse, resolution, max, |m| {
        method.converges(f, &p.prior(&f.ribbon, m))
    })
}

/// Each perturbation with its measured basin on the arc and on the sine, with the default
/// config; px, or degrees for a rotation. Their docs table is in `docs/performance.md`.
const PINNED: [(Perturbation, f64, f64); 8] = [
    (Perturbation::Translate, 15.0, 15.0),
    (Perturbation::Rotate, 21.0, 15.9),
    (Perturbation::Sine { waves: 2.0 }, 7.22, 12.78),
    (Perturbation::Sine { waves: 4.0 }, 3.31, 4.12),
    (Perturbation::Bump { sigma: 5.0 }, 0.91, 0.69),
    (Perturbation::Bump { sigma: 15.0 }, 15.47, 16.41),
    (Perturbation::ShiftRotate, 16.9, 16.3),
    (Perturbation::ShiftBump, 10.34, 10.03),
];

#[test]
fn the_basins_hold() {
    for (k, f) in fixtures().iter().enumerate() {
        let mut method = Method::single("defaults", config());
        for (p, arc, sine) in PINNED {
            let pinned = [arc, sine][k];
            let got = basin_of(f, &mut method, p);
            eprintln!("{} {p:?}: basin {got:.2} (pinned {pinned})", f.name);
            assert!(
                got >= 0.9 * pinned,
                "{} {p:?}: the basin shrank to {got:.2} from {pinned}",
                f.name
            );
        }
    }
}

#[test]
fn a_translation_within_the_reach_converges() {
    // ±14 px, inside the tracking reach of 15 px, both ways.
    for f in fixtures() {
        let mut method = Method::single("defaults", config());
        for k in -14..=14 {
            let d = f64::from(k);
            assert!(
                method.converges(&f, &Perturbation::Translate.prior(&f.ribbon, d)),
                "{}: a {d} px translation did not converge",
                f.name
            );
        }
    }
}

#[test]
fn a_translation_beyond_the_reach_never_locks_on() {
    // Beyond max_offset + max_width/2 + 5 px, no strip holds both of the bead's edges.
    let cfg = config();
    let beyond = f64::from(cfg.track.max_offset + 0.5 * cfg.max_width) + 5.0;
    let mut tracker = BeadTracker::new(cfg).expect("a valid config");
    for f in fixtures() {
        for d in [16.0, 20.0, 30.0, 45.0, beyond, beyond + 20.0] {
            for d in [d, -d] {
                let prior = Perturbation::Translate.prior(&f.ribbon, d);
                let got = tracker.track(&f.img.as_view(), &prior).expect("tracked");
                eprintln!(
                    "{} at {d} px: {:?}, support {}, rejects {:?}",
                    f.name, got.track.stop, got.summary.support, got.summary.rejects
                );
                // A hit, if any, is the bead: its centre on the true centreline.
                for smp in &got.samples {
                    if let Ok(hit) = smp.hit {
                        let off = f.ribbon.nearest(hit.pair.center).1;
                        assert!(off.abs() < 0.5, "{} at {d} px: a hit {off} px off", f.name);
                    }
                }
                if d.abs() >= beyond {
                    assert_eq!(got.track.stop, BeadStop::TooFewValid, "{} at {d}", f.name);
                    assert_eq!(got.summary.support, 0.0, "{} at {d}", f.name);
                }
            }
        }
    }
}

// ── the bending length ───────────────────────────────────────────────────

/// The methods the bending-length sweep compares.
fn methods() -> Vec<Method> {
    let mut out = Vec::new();
    for passes in [3, 6] {
        for b in [8.0, 4.0, 3.0, 2.0] {
            out.push(Method::single(
                format!("bending_px {b}, {passes} passes"),
                bending(b, passes),
            ));
        }
    }
    // Coarse to fine: each call's result is the next one's prior.
    let schedules: [(&str, &[(f32, usize)]); 3] = [
        ("8, 4, 2; 1 pass each", &[(8.0, 1), (4.0, 1), (2.0, 1)]),
        ("8, 4, 2; 2 passes each", &[(8.0, 2), (4.0, 2), (2.0, 2)]),
        ("8, 4; 3 passes each", &[(8.0, 3), (4.0, 3)]),
    ];
    for (label, stages) in schedules {
        let cfgs: Vec<BeadConfig> = stages.iter().map(|&(b, p)| bending(b, p)).collect();
        out.push(Method::schedule(format!("schedule {label}"), &cfgs));
    }
    out
}

/// The refined centreline's RMS and largest distance from the truth on the straight bead
/// under `noise_dn` of noise, the prior 2 px off and bent by a further 1 px over its
/// length; each averaged over 8 seeds.
fn noise_cost(method: &mut Method, noise_dn: f64) -> (f64, f64) {
    let (mut rms, mut max) = (0.0, 0.0);
    let seeds = 8;
    for seed in 0..seeds {
        let f = fixture("line", line(), noise_dn, 100 + seed);
        let len = f.ribbon.length();
        let p = prior(&f.ribbon, 15.0, len - 15.0, P2::new(0.0, 0.0), |s| {
            2.0 + (TAU * s / len).sin()
        });
        let got = method.track(&f.img, &p);
        let (mut ss, mut n, mut mx) = (0.0, 0, 0.0f64);
        for smp in &got.samples {
            let (s, d) = f.ribbon.nearest(smp.point);
            if s > 10.0 && s < len - 10.0 {
                ss += d * d;
                n += 1;
                mx = mx.max(d.abs());
            }
        }
        rms += (ss / f64::from(n)).sqrt();
        max += mx;
    }
    (rms / seeds as f64, max / seeds as f64)
}

#[test]
#[ignore = "a measurement: run in release with --ignored --nocapture"]
fn measure_the_bending_length() {
    let arc_f = fixture("arc", arc(), NOISE_DN, 23);
    eprintln!(
        "| Method | bump σ_b 5 | bump σ_b 15 | sine L/4 | polygon spacing | \
         noise 2 DN rms / max | noise 5 DN rms / max |"
    );
    for mut m in methods() {
        let b5 = basin_of(&arc_f, &mut m, Perturbation::Bump { sigma: 5.0 });
        let b15 = basin_of(&arc_f, &mut m, Perturbation::Bump { sigma: 15.0 });
        let l4 = basin_of(&arc_f, &mut m, Perturbation::Sine { waves: 4.0 });
        let poly = basin(5.0, 1.0, 200.0, |v| {
            m.converges(&arc_f, &polygon(&arc_f.ribbon, v))
        });
        let (r2, m2) = noise_cost(&mut m, 2.0);
        let (r5, m5) = noise_cost(&mut m, 5.0);
        eprintln!(
            "| {} | {b5:.2} | {b15:.2} | {l4:.2} | {poly:.0} | {r2:.4} / {m2:.4} | \
             {r5:.4} / {m5:.4} |",
            m.label
        );
    }
}
