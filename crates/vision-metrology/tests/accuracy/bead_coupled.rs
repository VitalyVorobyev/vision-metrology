//! Coupled pairing's walls on the bead fixture's hard cases (ADR-0019).
//!
//! A light bead 8 px wide, 60 DN on 80 DN, blurred by σ 1 px, under 2 DN of seeded noise
//! (five trials), along the fixture's line, arc and sine, as `tools/bead_eval/synthetic.py`
//! renders it. The tracker runs with coupled pairing, a width range of 2–16 px, a reach of
//! 8 px and stations 2 px apart, from the truth moved 2 px and bent by a further 1 px. Each
//! row pools both walls of every final hit over a cell's trials, as signed errors along
//! the normal (the `−n` wall's `d + w/2`, the `+n` wall's `d − w/2`), and reports the worst
//! cell over the three curves:
//! - `bead_coupled_walls`: the clean bead;
//! - `bead_coupled_walls_continuous`: the same, with both stages on
//!   `Derivative::Continuous` (σ 1 px), whose edges the walls then snap to;
//! - `bead_coupled_varying_width`: the width runs from 6 to 12 px along the curve;
//! - `bead_coupled_distractor_2px`: a 3 px stripe, 40 DN, 2 px outside the `+n` wall,
//!   which merges with it in the derivative and pulls it inwards; only that wall pools;
//! - `bead_coupled_gap`: no bead over 30 px in the middle; no station inside the gap may
//!   hit, and the walls pool outside it;
//! - `bead_coupled_highlight`: a 1.5 px stripe 40 DN brighter inside the bead, off centre.
//!
//! The parallel line, which the whole call's margin flags, is checked in `tests/bead.rs`
//! (`coupled::a_parallel_bead_midway_lowers_the_first_pass_margin`).

use std::f64::consts::TAU;
use std::sync::OnceLock;

use vision_metrology::Point2f;
use vision_metrology::measure::{
    BeadCaliper, BeadConfig, BeadCoupling, BeadPairing, BeadTracker, Derivative, ProfileConfig,
};

use super::ribbon::{Curve, P2, Ribbon, Scene, Stripe, Width};
use super::{Measured, mean_std};

const BG: f64 = 80.0;
const CONTRAST: f64 = 60.0;
const PSF: f64 = 1.0;
const NOISE_DN: f64 = 2.0;
const TRIALS: u64 = 5;
const SIZE: (usize, usize) = (360, 240);

/// The three curves, inside the image with room for the bead and its distractor.
fn curves() -> [(&'static str, Curve); 3] {
    [
        (
            "line",
            Curve::line(P2::new(40.0, 95.0), P2::new(320.0, 145.0)),
        ),
        ("arc", Curve::arc(P2::new(180.0, 330.0), 250.0, -2.2, 1.15)),
        (
            "sine",
            Curve::sine(P2::new(40.0, 120.0), P2::polar(0.0), 280.0, 15.0, 200.0),
        ),
    ]
}

/// One hard case: the bead's width, a stripe or none, a gap or none, and the stages'
/// derivative.
#[derive(Clone, Copy)]
struct Case {
    width: Width,
    stripe: Option<(Stripe, bool)>,
    gap: bool,
    continuous: bool,
    /// Pool only the `+n` wall.
    second_only: bool,
}

const CLEAN: Case = Case {
    width: Width::Const(8.0),
    stripe: None,
    gap: false,
    continuous: false,
    second_only: false,
};

fn config(continuous: bool) -> BeadConfig {
    let d = BeadConfig::default();
    let stage = |c: BeadCaliper| BeadCaliper {
        profile: ProfileConfig {
            derivative: if continuous {
                Derivative::Continuous
            } else {
                c.profile.derivative
            },
            ..c.profile
        },
        ..c
    };
    BeadConfig {
        min_width: 2.0,
        max_width: 16.0,
        spacing: 2.0,
        track: stage(BeadCaliper {
            max_offset: 8.0,
            ..d.track
        }),
        measure: stage(d.measure),
        pairing: BeadPairing::Coupled(BeadCoupling::default()),
        ..d
    }
}

/// The worst cell of `case` over the curves.
fn run(case: Case) -> Measured {
    let mut cells = Vec::new();
    for (name, curve) in curves() {
        let len = curve.length();
        let mut ribbon = Ribbon::new(curve, case.width);
        if case.gap {
            ribbon = ribbon.gap(0.5 * len - 15.0, 0.5 * len + 15.0);
        }
        let mut scene = Scene::new(SIZE.0, SIZE.1, ribbon.clone(), BG, BG + CONTRAST, PSF);
        if let Some((stripe, inside)) = case.stripe {
            scene = if inside {
                scene.highlight(stripe)
            } else {
                scene.parallel(stripe)
            };
        }
        let clean = scene.render();
        let end = 8.0;
        let n = ((len - 2.0 * end) / 4.0).ceil() as usize;
        let prior: Vec<Point2f> = (0..=n)
            .map(|k| {
                let s = end + (len - 2.0 * end) * k as f64 / n as f64;
                let bend = 2.0 + (TAU * s / len).sin();
                (ribbon.curve.center(s) + ribbon.curve.normal(s) * bend).point()
            })
            .collect();
        let mut tracker = BeadTracker::new(config(case.continuous)).expect("valid");
        let mut errs = Vec::new();
        let (mut hits, mut stations) = (0usize, 0usize);
        for trial in 0..TRIALS {
            let img = clean.noisy(NOISE_DN, 0xC0 << 8 | trial).to_u8();
            let got = tracker.track(&img.as_view(), &prior).expect("tracked");
            for smp in &got.samples {
                let (s, _) = ribbon.nearest(smp.point);
                let in_gap = ribbon.gaps.iter().any(|&(a, b)| s > a - 6.0 && s < b + 6.0);
                if s < end + 4.0 || s > len - end - 4.0 || in_gap {
                    if ribbon.gaps.iter().any(|&(a, b)| s > a + 3.0 && s < b - 3.0) {
                        assert!(
                            smp.hit.is_err(),
                            "{name}: a hit inside the gap at s = {s:.1}"
                        );
                    }
                    continue;
                }
                stations += 1;
                let Ok(hit) = smp.hit else { continue };
                hits += 1;
                let (s1, d1) = ribbon.nearest(hit.pair.first.p);
                let (s2, d2) = ribbon.nearest(hit.pair.second.p);
                if !case.second_only {
                    errs.push((d1 + 0.5 * ribbon.width(s1)) as f32);
                }
                errs.push((d2 - 0.5 * ribbon.width(s2)) as f32);
            }
        }
        assert!(
            hits * 20 >= stations * 19,
            "{name}: {hits} hits of {stations} stations"
        );
        let (bias, sigma) = mean_std(&errs);
        eprintln!(
            "coupled walls, {name}{}: {bias:+.4} ± {sigma:.4} px ({hits} hits of {stations})",
            if case.continuous { ", continuous" } else { "" }
        );
        cells.push((bias, sigma));
    }
    Measured::worst_of(cells.into_iter())
}

fn cached(cell: &'static OnceLock<Measured>, case: Case) -> Measured {
    *cell.get_or_init(|| run(case))
}

pub(super) fn walls() -> Measured {
    static C: OnceLock<Measured> = OnceLock::new();
    cached(&C, CLEAN)
}

pub(super) fn walls_continuous() -> Measured {
    static C: OnceLock<Measured> = OnceLock::new();
    cached(
        &C,
        Case {
            continuous: true,
            ..CLEAN
        },
    )
}

pub(super) fn varying_width() -> Measured {
    static C: OnceLock<Measured> = OnceLock::new();
    cached(
        &C,
        Case {
            width: Width::Linear { w0: 6.0, w1: 12.0 },
            ..CLEAN
        },
    )
}

pub(super) fn distractor_2px() -> Measured {
    static C: OnceLock<Measured> = OnceLock::new();
    let stripe = Stripe {
        offset: 4.0 + 2.0 + 1.5,
        width: 3.0,
        contrast: 40.0,
    };
    cached(
        &C,
        Case {
            stripe: Some((stripe, false)),
            second_only: true,
            ..CLEAN
        },
    )
}

pub(super) fn gap() -> Measured {
    static C: OnceLock<Measured> = OnceLock::new();
    cached(&C, Case { gap: true, ..CLEAN })
}

pub(super) fn highlight() -> Measured {
    static C: OnceLock<Measured> = OnceLock::new();
    let stripe = Stripe {
        offset: 1.25,
        width: 1.5,
        contrast: 40.0,
    };
    cached(
        &C,
        Case {
            stripe: Some((stripe, true)),
            ..CLEAN
        },
    )
}
