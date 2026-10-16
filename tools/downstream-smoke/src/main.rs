//! Passes nalgebra points across the two crates' public APIs.
//!
//! `Pt2::cast` returns calibration-rs's `nalgebra::Point2<f32>`, and the
//! assignment to `Point2f` only type-checks when that is the same nalgebra
//! this workspace uses. The run itself checks the round trip numerically.

use vision_calibration_core::{Mat3, Pt2, pixel_to_normalized};
use vision_metrology::Point2f;
use vision_metrology::fit::{FitConfig, fit_circle};

fn main() {
    let (cx, cy, r) = (320.0_f64, 240.0_f64, 50.0_f64);

    // Points on a circle, built as calibration-rs points.
    let ring: Vec<Pt2> = (0..36)
        .map(|i| {
            let t = f64::from(i) * std::f64::consts::TAU / 36.0;
            Pt2::new(cx + r * t.cos(), cy + r * t.sin())
        })
        .collect();

    // calibration-rs -> vision-metrology.
    let pts: Vec<Point2f> = ring.iter().map(|p| p.cast::<f32>()).collect();
    let fit = fit_circle(&pts, &FitConfig::default()).expect("circle fit");

    // vision-metrology -> calibration-rs.
    let center: Pt2 = fit.model.center.cast::<f64>();
    let k = Mat3::new(1000.0, 0.0, cx, 0.0, 1000.0, cy, 0.0, 0.0, 1.0);
    let n = pixel_to_normalized(center, &k);

    assert!((center - Pt2::new(cx, cy)).norm() < 1e-3, "centre {center}");
    assert!((f64::from(fit.model.radius) - r).abs() < 1e-3);
    assert!(n.coords.norm() < 1e-6, "normalized centre {n}");
    println!("ok: centre {center} -> normalized {n}");
}
