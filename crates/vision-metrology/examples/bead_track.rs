//! Example: track a bead along a perturbed prior, and see where and why it was rejected.
//!
//! Renders a light S-shaped bead 40 px wide on a sloping background, with a gap, a
//! brighter step that starts 4 px outside one edge and slowly leaves it, and a highlight
//! stripe inside. The prior is the true centreline moved 5 px sideways with a 5 px bump in it.
//! The tracker runs with `clearance` on, so the stations beside the step are rejected
//! rather than measured against it.
//!
//! The example checks its own result (the support, why the loop stopped, the refined
//! curve against the truth, and the rejections at the gap and the step), prints the
//! quality summary, and writes an overlay:
//! - the prior, thin grey;
//! - the final stage's strips, faint;
//! - the refined centreline, green;
//! - the final edges, yellow;
//! - each rejected station, coloured by its reason (the legend is printed).
//!
//! ```text
//! cargo run --release -p vision-metrology --example bead_track -- --out bead.png
//! ```

#[path = "common/bead_draw.rs"]
mod bead_draw;
#[path = "common/overlay.rs"]
mod overlay;
#[path = "common/ribbon.rs"]
mod ribbon;

use std::collections::BTreeMap;
use std::path::PathBuf;

use anyhow::{Context, Result, ensure};
use clap::Parser;
use ribbon::{Curve, P2, Ribbon, Scene, Step, Stripe, Width};
use vision_metrology::measure::diagnostics::explain_bead;
use vision_metrology::measure::{BeadConfig, BeadReject, BeadStop, BeadTracker};
use vision_metrology::{Image, Point2f};

#[derive(Parser, Debug)]
#[command(about = "Track a synthetic bead from a perturbed prior and draw the result")]
struct Args {
    /// Where to write the overlay PNG.
    #[arg(long, default_value = "target/bead_track.png")]
    out: PathBuf,
}

const SIZE: (usize, usize) = (640, 400);
const WIDTH: f64 = 40.0;
/// The gap, in px of arc length along the bead.
const GAP: (f64, f64) = (300.0, 335.0);
/// Where the step passes the bead's +n edge, in px of arc length, how far outside it, and
/// the angle at which it leaves the bead.
const STEP_AT: f64 = 10.0;
const STEP_GAP: f64 = 4.0;
const STEP_TILT_DEG: f64 = 8.0;

/// The scene: the bead, its truth, and the rendered `u8` image.
fn scene() -> (Ribbon, Image<u8>) {
    let ribbon = Ribbon::new(
        Curve::s_bend(P2::new(80.0, 300.0), 0.0, 260.0, -0.9),
        Width::Const(WIDTH),
    )
    .gap(GAP.0, GAP.1);
    // A straight step 60 DN brighter on its far side, 4 px outside the bead's +n edge at
    // the prior's start, leaving it at 8°; the bend carries the bead away from it too.
    let (t, n) = (ribbon.curve.tangent(STEP_AT), ribbon.curve.normal(STEP_AT));
    let tilt = STEP_TILT_DEG.to_radians();
    let dir = t * tilt.cos() + n * tilt.sin();
    let step = Step {
        point: ribbon.curve.center(STEP_AT) + n * (0.5 * WIDTH + STEP_GAP),
        normal: dir.perp(),
        contrast: 60.0,
    };
    let img = Scene::new(SIZE.0, SIZE.1, ribbon.clone(), 30.0, 160.0, 1.2)
        .gradient(P2::new(0.0, 0.0), P2::new(0.06, 0.04))
        .highlight(Stripe {
            offset: -7.0,
            width: 6.0,
            contrast: 45.0,
        })
        .step(step)
        .render()
        .noisy(2.0, 1)
        .to_u8();
    (ribbon, img)
}

/// The prior: the truth from 10 px in from each end, a vertex every 8 px, moved 5 px along
/// the normal, with a 5 px Gaussian bump (σ 15 px of arc) at 75% of its length.
fn prior(ribbon: &Ribbon) -> Vec<Point2f> {
    let len = ribbon.length();
    let n = ((len - 20.0) / 8.0).ceil() as usize;
    (0..=n)
        .map(|k| {
            let s = 10.0 + (len - 20.0) * k as f64 / n as f64;
            let bump = 5.0 * (-(s - 0.75 * len).powi(2) / (2.0 * 15.0 * 15.0)).exp();
            (ribbon.curve.center(s) + ribbon.curve.normal(s) * (5.0 + bump)).point()
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (ribbon, img) = scene();
    let prior = prior(&ribbon);
    let cfg = BeadConfig {
        clearance: Some(8.0),
        ..BeadConfig::default()
    };
    let mut tracker = BeadTracker::new(cfg)?;
    let bead = tracker.track(&img.as_view(), &prior)?;

    // ── the quality summary ──
    let stats = bead.summary.stats.context("no station found the bead")?;
    println!(
        "{} stations {:.2} px apart; stopped {} after {} passes",
        bead.samples.len(),
        bead.spacing,
        bead.track.stop.as_str(),
        bead.track.passes.len()
    );
    for (k, pass) in bead.track.passes.iter().enumerate() {
        let solve = pass.solve.context("every pass solved")?;
        println!(
            "  pass {}: {} of {} stations found the bead, correction up to {:.3} px",
            k + 1,
            pass.n_valid,
            bead.samples.len(),
            solve.correction_max
        );
    }
    println!(
        "support {:.3}, longest gap {:.1} px; centre rms {:.4} px, max {:.4} px; width \
         {:.3} ± {:.3} px ({:.3} to {:.3})",
        bead.summary.support,
        bead.summary.longest_gap,
        stats.center_rms,
        stats.center_max_dev,
        stats.width_mean,
        stats.width_std,
        stats.width_min,
        stats.width_max
    );
    let rejects: BTreeMap<&str, usize> = bead
        .summary
        .rejects
        .iter()
        .map(|&(r, n)| (r.as_str(), n))
        .collect();
    println!("rejected: {rejects:?}");

    // ── the checks, against the truth ──
    ensure!(
        bead.track.stop != BeadStop::TooFewValid,
        "the tracking gave up"
    );
    ensure!(
        (0.75..0.95).contains(&bead.summary.support),
        "support {}",
        bead.summary.support
    );
    let (mut curve_err, mut width_err) = (0.0f64, 0.0f64);
    let (mut in_gap, mut beside_step) = (0, 0);
    for smp in &bead.samples {
        let (s, d) = ribbon.nearest(smp.point);
        let near_gap = s > GAP.0 - 8.0 && s < GAP.1 + 8.0;
        match smp.hit {
            Ok(hit) => {
                // The gap's ends are square cuts, whose pixels carry no truth.
                if (s - GAP.0).abs() > 10.0 && (s - GAP.1).abs() > 10.0 {
                    curve_err = curve_err.max(d.abs());
                    width_err = width_err.max((f64::from(hit.pair.width) - WIDTH).abs());
                }
                ensure!(
                    !(s > GAP.0 + 3.0 && s < GAP.1 - 3.0),
                    "a hit in the gap at s = {s:.0}"
                );
            }
            // At the gap's square ends a strip can also see the cut, too oblique.
            Err(BeadReject::Caliper(_)) if near_gap => in_gap += 1,
            Err(BeadReject::Clearance) if s < STEP_AT + 40.0 => beside_step += 1,
            Err(other) => anyhow::bail!("s = {s:.0}: unexpected rejection {}", other.as_str()),
        }
    }
    println!(
        "against the truth: refined curve within {curve_err:.4} px, widths within \
         {width_err:.4} px; {in_gap} stations rejected in the gap, {beside_step} beside \
         the step"
    );
    ensure!(curve_err < 0.1, "the refined curve is {curve_err} px off");
    ensure!(width_err < 0.2, "a width is {width_err} px off");
    ensure!(in_gap >= 6, "{in_gap} stations rejected in the gap");
    ensure!(
        beside_step >= 3,
        "{beside_step} stations rejected beside the step"
    );

    // ── the overlay ──
    let trace = explain_bead(&mut tracker, &img.as_view(), &prior)?;
    ensure!(
        trace.result == bead,
        "explain_bead returns what track returns"
    );
    let canvas = bead_draw::draw(&img, &prior, &trace);
    println!("legend: prior grey, strips faint blue, refined centreline green, edges yellow");
    let mut legend = BTreeMap::new();
    for smp in &bead.samples {
        if let Err(r) = smp.hit {
            legend.insert(r.as_str(), bead_draw::reason_colour(r));
        }
    }
    for (name, c) in &legend {
        println!("  rejected, {name}: rgb{c:?}");
    }
    if let Some(dir) = args.out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    }
    canvas
        .save(&args.out)
        .with_context(|| format!("writing {}", args.out.display()))?;
    println!("overlay -> {}", args.out.display());
    Ok(())
}
