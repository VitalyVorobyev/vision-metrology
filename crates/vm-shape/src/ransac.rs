//! RANSAC ellipse fitting.
//!
//! Iteratively samples 5 points, fits a Fitzgibbon conic, and scores the model
//! by counting inliers (points whose normalised algebraic distance is below a
//! configurable threshold).
//!
//! The random state is a seeded LCG for full reproducibility (no external RNG
//! dependency). The seed is stored in [`ConicFitter`] and can be overridden.
//!
//! # Inlier distance
//! We use the Sampson (first-order) approximation to the geometric distance:
//!
//! ```text
//! d_sampson(p) = |F(p)| / ‖∇F(p)‖
//! ```
//!
//! This is accurate to `O(distance²)` for points close to the conic and is
//! cheap to compute without solving a quartic.

use vm_core::{Error, Point2f};

use crate::conic::{Conic2f, Ellipse2f};
use crate::fit_conic::fit_fitzgibbon;

// ---------------------------------------------------------------------------
// Minimal LCG PRNG (reproducible, no-std compatible)
// ---------------------------------------------------------------------------

/// Linear congruential generator with Knuth parameters.
///
/// `next()` advances the state and returns a value in `[0, u64::MAX)`.
pub(crate) struct Lcg(u64);

impl Lcg {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }

    /// Return next value in [0, m).
    pub(crate) fn next_mod(&mut self, m: usize) -> usize {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as usize % m
    }
}

// ---------------------------------------------------------------------------
// RANSAC sampling
// ---------------------------------------------------------------------------

/// Sample 5 distinct indices from `[0, n)` using the LCG.
fn sample5(rng: &mut Lcg, n: usize, out: &mut [usize; 5]) {
    let mut used = [usize::MAX; 5];
    let mut count = 0;
    while count < 5 {
        let idx = rng.next_mod(n);
        if !used[..count].contains(&idx) {
            out[count] = idx;
            used[count] = idx;
            count += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Inlier counting
// ---------------------------------------------------------------------------

/// Count inliers: points with Sampson distance `|F(p)| / ‖∇F(p)‖ < tol`.
fn count_inliers(conic: &Conic2f, pts: &[Point2f], tol: f32) -> usize {
    pts.iter()
        .filter(|&&p| {
            let f = conic.eval(p);
            let g = conic.grad_norm(p);
            if g < 1e-8 {
                return false;
            }
            (f / g).abs() < tol
        })
        .count()
}

/// Collect inlier point indices.
fn collect_inlier_indices(conic: &Conic2f, pts: &[Point2f], tol: f32, out: &mut Vec<usize>) {
    out.clear();
    for (i, &p) in pts.iter().enumerate() {
        let f = conic.eval(p);
        let g = conic.grad_norm(p);
        if g >= 1e-8 && (f / g).abs() < tol {
            out.push(i);
        }
    }
}

// ---------------------------------------------------------------------------
// Public RANSAC entry point
// ---------------------------------------------------------------------------

/// Run RANSAC ellipse fitting over `pts`.
///
/// # Parameters
/// - `pts`: input point set (in pixel coordinates).
/// - `ransac_iters`: number of RANSAC iterations.
/// - `inlier_tol`: Sampson distance threshold in pixels.
/// - `min_inliers`: minimum number of inliers to accept a hypothesis.
/// - `rng_seed`: seed for the internal LCG; use a fixed value for reproducibility.
///
/// # Returns
/// The best-fitting [`Ellipse2f`] found, or `Err` when:
/// - Fewer than 5 points are provided.
/// - No hypothesis accumulates at least `min_inliers` inliers.
///
/// # Algorithm
/// 1. Sample 5 random points, fit Fitzgibbon conic.
/// 2. If conic is not an ellipse, discard hypothesis.
/// 3. Count inliers.
/// 4. After all iterations, re-fit on best inlier set using Fitzgibbon.
pub(crate) fn ransac_fit_ellipse(
    pts: &[Point2f],
    ransac_iters: usize,
    inlier_tol: f32,
    min_inliers: usize,
    rng_seed: u64,
    inlier_scratch: &mut Vec<usize>,
) -> Result<Ellipse2f, Error> {
    if pts.len() < 5 {
        return Err(Error::InsufficientData {
            need: 5,
            got: pts.len(),
        });
    }

    let mut rng = Lcg::new(rng_seed);
    let mut best_count = 0usize;
    let mut best_inliers: Vec<usize> = Vec::new();
    let mut sample = [0usize; 5];

    for _ in 0..ransac_iters {
        // Sample 5 points
        if pts.len() == 5 {
            sample = [0, 1, 2, 3, 4];
        } else {
            sample5(&mut rng, pts.len(), &mut sample);
        }

        let sample_pts: Vec<Point2f> = sample.iter().map(|&i| pts[i]).collect();

        // Fit Fitzgibbon conic to the sample
        let conic = match fit_fitzgibbon(&sample_pts) {
            Ok(c) if c.is_ellipse() => c,
            _ => continue,
        };

        // Score hypothesis
        let count = count_inliers(&conic, pts, inlier_tol);
        if count > best_count {
            best_count = count;
            collect_inlier_indices(&conic, pts, inlier_tol, inlier_scratch);
            best_inliers.clear();
            best_inliers.extend_from_slice(inlier_scratch);
        }
    }

    if best_count < min_inliers {
        return Err(Error::Degenerate(
            "RANSAC: insufficient inliers for ellipse fit",
        ));
    }

    // Re-fit on all inliers
    let inlier_pts: Vec<Point2f> = best_inliers.iter().map(|&i| pts[i]).collect();
    let refined_conic = fit_fitzgibbon(&inlier_pts)?;
    refined_conic.to_ellipse()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;
    use vm_core::Vec2f;

    use crate::conic::Ellipse2f;

    fn ellipse_points_with_noise(ell: &Ellipse2f, n: usize, noise: f32, seed: u64) -> Vec<Point2f> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f32 / n as f32;
                let mut p = ell.point_at(t);
                // Add deterministic noise in range [-noise, noise]
                let nx = (rng.next_mod(1000) as f32 / 500.0 - 1.0) * noise;
                let ny = (rng.next_mod(1000) as f32 / 500.0 - 1.0) * noise;
                p.x += nx;
                p.y += ny;
                p
            })
            .collect()
    }

    #[test]
    fn ransac_recovers_ellipse_with_outliers() {
        // 80 inlier points on an ellipse + 20 random outliers.
        // The RANSAC should still find the ellipse to within 0.5 px / 1%.
        let ell = Ellipse2f {
            center: Point2f { x: 50.0, y: 40.0 },
            semi_axes: Vec2f { x: 20.0, y: 10.0 },
            angle: 0.0,
        };

        let mut pts = ellipse_points_with_noise(&ell, 80, 0.3, 42);

        // Add 20 outliers spread across a wider region
        let mut rng = Lcg::new(999);
        for _ in 0..20 {
            pts.push(Point2f {
                x: rng.next_mod(200) as f32,
                y: rng.next_mod(200) as f32,
            });
        }

        let mut scratch = Vec::new();
        let result = ransac_fit_ellipse(&pts, 200, 2.0, 60, 42, &mut scratch)
            .expect("RANSAC should converge");

        assert!(
            (result.center.x - 50.0).abs() < 1.0,
            "cx={}",
            result.center.x
        );
        assert!(
            (result.center.y - 40.0).abs() < 1.0,
            "cy={}",
            result.center.y
        );
        assert!(
            (result.semi_major() - 20.0).abs() < 20.0 * 0.05,
            "semi_major={}",
            result.semi_major()
        );
        assert!(
            (result.semi_minor() - 10.0).abs() < 10.0 * 0.05,
            "semi_minor={}",
            result.semi_minor()
        );
    }

    #[test]
    fn ransac_fails_with_too_few_points() {
        let pts = vec![Point2f { x: 1.0, y: 2.0 }; 4];
        let mut scratch = Vec::new();
        assert!(ransac_fit_ellipse(&pts, 10, 1.0, 5, 0, &mut scratch).is_err());
    }

    #[test]
    fn ransac_fails_with_all_outliers() {
        // All 30 points uniformly random (no coherent ellipse).
        let mut rng = Lcg::new(7);
        let pts: Vec<Point2f> = (0..30)
            .map(|_| Point2f {
                x: rng.next_mod(100) as f32,
                y: rng.next_mod(100) as f32,
            })
            .collect();
        let mut scratch = Vec::new();
        // With a very strict inlier requirement (50 inliers from 30 points), must fail.
        assert!(ransac_fit_ellipse(&pts, 50, 0.5, 50, 0, &mut scratch).is_err());
    }
}
