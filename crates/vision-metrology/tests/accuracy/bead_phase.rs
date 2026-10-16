//! The bead tracker over the phase sweep: does `BeadTracker` lock to the pixel grid?
//!
//! The tracker locates its edges with strip calipers (`phase.rs` measures those alone).
//! These rows run the whole tracker on a straight light bar 16 px wide, the
//! pixel-integrated closed form of `strip.rs` at no blur, PSF σ 0.5 and σ 1.0 px, levels
//! 40 and 200 DN under 2 DN of seeded noise. Its normal is +x, the angle at which the
//! strip rows lock most. The prior is the centreline moved 2 px along the normal, 48 px
//! long.
//!
//! The sweep moves the bar's normal offset in 200 equal steps that move each edge's
//! crossing of a fixed line along the normal one pixel in `x`. The final stage's edges
//! (both, at every station that hit) give:
//! - the worst |bias| over phases of the edges' signed distance from the true edge along
//!   the normal, per phase pooled over stations;
//! - the locking index of their `x` coordinates, as in `phase.rs`.
//!
//! Two configs: the default, and both stages on `Derivative::Continuous` at σ 2.5 px.

use vision_metrology::measure::{BeadConfig, BeadTracker, Derivative};
use vision_metrology::{Image, Point2f};

use super::phase::{BINS, bin, locking_index, seed};
use super::strip::{normal, quantize, render_steps};

const SIZE: usize = 96;
const WIDTH: f64 = 16.0;
const BLURS: [f64; 3] = [0.0, 0.5, 1.0];
const ANGLES_DEG: [f64; 1] = [0.0];
const PHASES: usize = 200;
const TRIALS: u64 = 1;
const NOISE_DN: f64 = 2.0;

/// A tracker config, its name, and its envelopes: |bias| in px, the locking index.
struct Variant {
    name: &'static str,
    sigma: Option<f32>,
    bias_envelope: f64,
    locking_envelope: f64,
}

fn config(v: &Variant) -> BeadConfig {
    let mut cfg = BeadConfig {
        min_width: (0.5 * WIDTH) as f32,
        max_width: (1.5 * WIDTH) as f32,
        ..BeadConfig::default()
    };
    if let Some(sigma) = v.sigma {
        for stage in [&mut cfg.track, &mut cfg.measure] {
            stage.profile.sigma = sigma;
            stage.profile.derivative = Derivative::Continuous;
        }
    }
    cfg
}

/// Per blur: the worst |bias| over angles and phases, and the worst locking index.
fn sweep(v: &Variant) -> ([f64; 3], [f64; 3]) {
    let mut tracker = BeadTracker::new(config(v)).expect("a valid config");
    let (mut bias, mut locking) = ([0.0f64; 3], [0.0f64; 3]);
    for (bi, &blur) in BLURS.iter().enumerate() {
        for (ai, &angle) in ANGLES_DEG.iter().enumerate() {
            let n = normal(angle);
            let t = (-n.1, n.0);
            let c = 0.5 * (SIZE - 1) as f64;
            let d0 = c * (n.0 + n.1);
            let mut hist = [0u64; BINS];
            for k in 0..PHASES {
                let d = d0 + k as f64 / (PHASES as f64 * n.0);
                let edges = [d - 0.5 * WIDTH, d + 0.5 * WIDTH];
                let clean = render_steps(SIZE, n, &[(edges[0], 1.0), (edges[1], -1.0)], blur);
                // The prior: 2 px off the centreline, along it through the image centre.
                let mid = (c + (d - d0 + 2.0) * n.0, c + (d - d0 + 2.0) * n.1);
                let prior = [-24.0, 24.0]
                    .map(|s: f64| Point2f::new((mid.0 + s * t.0) as f32, (mid.1 + s * t.1) as f32));
                let (mut sum, mut count) = (0.0f64, 0usize);
                for trial in 0..TRIALS {
                    let px = quantize(
                        &clean,
                        NOISE_DN,
                        seed([7, bi as u64, ai as u64, k as u64, trial]),
                    );
                    let img = Image::from_vec(SIZE, SIZE, px).expect("valid image");
                    let got = tracker
                        .track(&img.as_view(), &prior)
                        .expect("a valid prior");
                    for hit in got.samples.iter().filter_map(|s| s.hit.ok()) {
                        for (e, truth) in [hit.pair.first, hit.pair.second].iter().zip(edges) {
                            let (x, y) = (f64::from(e.p.x), f64::from(e.p.y));
                            sum += n.0 * x + n.1 * y - truth;
                            count += 1;
                            hist[bin(x)] += 1;
                        }
                    }
                }
                assert!(
                    count > 0,
                    "{}: no station hit at {blur} {angle}° phase {k}",
                    v.name
                );
                bias[bi] = bias[bi].max((sum / count as f64).abs());
            }
            locking[bi] = locking[bi].max(locking_index(&hist));
        }
    }
    (bias, locking)
}

/// Envelopes measured 2026-10-16 (`cargo test -p vision-metrology --test accuracy
/// --all-features bead_phase -- --nocapture`): |bias| at ~1.5× the measured worst and the
/// locking index `L` at `1 + 1.5 (L − 1)`, except that the continuous row's is pinned at the
/// acceptance bar of 1.2 (measured 1.17); the numbers are in `docs/performance.md`.
const VARIANTS: &[Variant] = &[
    Variant {
        name: "bead_default",
        sigma: None,
        bias_envelope: 0.067,
        locking_envelope: 2.15,
    },
    Variant {
        name: "bead_continuous_sigma2.5",
        sigma: Some(2.5),
        bias_envelope: 0.022,
        locking_envelope: 1.2,
    },
];

#[test]
fn bead_phase_sweep_envelopes_hold() {
    let mut failures = Vec::new();
    for v in VARIANTS {
        let (bias, locking) = sweep(v);
        let worst = |a: [f64; 3]| a.iter().copied().fold(0.0, f64::max);
        eprintln!(
            "{}: bias={:.4} locking={:.3} (envelope {:.4}/{:.3}); per blur {BLURS:?}: bias \
             {bias:.4?} locking {locking:.3?}",
            v.name,
            worst(bias),
            worst(locking),
            v.bias_envelope,
            v.locking_envelope
        );
        if worst(bias) > v.bias_envelope || worst(locking) > v.locking_envelope {
            failures.push(v.name);
        }
    }
    assert!(failures.is_empty(), "outside their envelopes: {failures:?}");
}
