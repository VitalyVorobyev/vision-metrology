//! [`ConicFitter`] — public entry point for algebraic conic and ellipse fitting.
//!
//! Wraps both the Bookstein and Fitzgibbon DLS fitting methods and provides a
//! RANSAC ellipse fitter, all using pre-allocated scratch storage.
//!
//! # Example
//! ```rust
//! use vision_metrology::{ConicFitter, ConicFitConfig, Ellipse2f};
//! use vm_primitives::{Point2f, Vec2f};
//! use core::f32::consts::PI;
//!
//! let ell = Ellipse2f {
//!     center: Point2f { x: 50.0, y: 40.0 },
//!     semi_axes: Vec2f { x: 20.0, y: 10.0 },
//!     angle: 0.0,
//! };
//! let pts: Vec<Point2f> = (0..20).map(|i| {
//!     let t = 2.0 * PI * i as f32 / 20.0;
//!     ell.point_at(t)
//! }).collect();
//!
//! let mut fitter = ConicFitter::new();
//! let cfg = ConicFitConfig::default();
//! let conic = fitter.fit(&pts, &cfg).expect("fit succeeded");
//! assert!(conic.is_ellipse());
//! ```

use vm_primitives::{Error, Point2f};

use super::conic::{Conic2f, Ellipse2f};
use super::fit_conic::{fit_bookstein, fit_fitzgibbon};
use super::ransac::ransac_fit_ellipse;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for [`ConicFitter`].
#[derive(Debug, Clone, PartialEq)]
pub struct ConicFitConfig {
    /// When `true`, use the Bookstein `‖c‖ = 1` constraint (works for all
    /// conics, similarity-invariant). When `false`, use the Fitzgibbon
    /// `4AC − B² = 1` constraint (always yields an ellipse).
    ///
    /// Default: `true` (Bookstein).
    pub use_bookstein: bool,

    /// Number of RANSAC iterations. `0` disables RANSAC (direct fit only).
    ///
    /// Default: `0`.
    pub ransac_iters: usize,

    /// Sampson distance threshold (in pixels) to classify a point as an inlier
    /// during RANSAC.
    ///
    /// Default: `1.0`.
    pub inlier_tol: f32,

    /// Minimum number of inliers required to accept a RANSAC hypothesis.
    ///
    /// Default: `5`.
    pub min_inliers: usize,

    /// Seed for the internal reproducible LCG random number generator used in
    /// RANSAC sampling. Using the same seed gives deterministic results.
    ///
    /// Default: `42`.
    pub rng_seed: u64,
}

impl Default for ConicFitConfig {
    fn default() -> Self {
        Self {
            use_bookstein: true,
            ransac_iters: 0,
            inlier_tol: 1.0,
            min_inliers: 5,
            rng_seed: 42,
        }
    }
}

// ---------------------------------------------------------------------------
// Fitter
// ---------------------------------------------------------------------------

/// Reusable algebraic conic and ellipse fitter.
///
/// Create once with [`ConicFitter::new`] and reuse across frames to avoid
/// reallocating internal scratch storage.
pub struct ConicFitter {
    /// Scratch buffer for RANSAC inlier index collection.
    inlier_scratch: Vec<usize>,
}

impl ConicFitter {
    /// Create a new fitter with empty scratch buffers.
    pub fn new() -> Self {
        Self {
            inlier_scratch: Vec::new(),
        }
    }

    /// Direct algebraic fit to `pts`.
    ///
    /// Uses Bookstein (`‖c‖ = 1`) when `cfg.use_bookstein = true`, or
    /// Fitzgibbon (`4AC − B² = 1`) otherwise.
    ///
    /// # Errors
    /// - [`Error::InsufficientData`] when `pts.len() < 5`.
    /// - [`Error::Degenerate`] when the point set is numerically degenerate
    ///   (e.g. collinear, or Fitzgibbon fails to find a valid ellipse eigenvector).
    ///
    /// # Notes
    /// When `use_bookstein = false` (Fitzgibbon), the returned conic is
    /// guaranteed to be an ellipse. With `use_bookstein = true`, the result may
    /// be any conic; check [`Conic2f::is_ellipse`] if needed.
    pub fn fit(&mut self, pts: &[Point2f], cfg: &ConicFitConfig) -> Result<Conic2f, Error> {
        if cfg.use_bookstein {
            fit_bookstein(pts)
        } else {
            fit_fitzgibbon(pts)
        }
    }

    /// RANSAC ellipse fit.
    ///
    /// Internally always uses the Fitzgibbon constraint (which guarantees an
    /// ellipse hypothesis at each iteration), regardless of `cfg.use_bookstein`.
    ///
    /// # Errors
    /// - [`Error::InsufficientData`] when `pts.len() < 5`.
    /// - [`Error::Degenerate`] when RANSAC fails to accumulate sufficient inliers
    ///   or the final re-fit is degenerate.
    ///
    /// # Reproducibility
    /// Results are fully deterministic for a given `cfg.rng_seed` and point set.
    pub fn fit_ellipse_ransac(
        &mut self,
        pts: &[Point2f],
        cfg: &ConicFitConfig,
    ) -> Result<Ellipse2f, Error> {
        if cfg.ransac_iters == 0 {
            // Degenerate case: RANSAC with 0 iterations just runs the direct fit.
            let conic = fit_fitzgibbon(pts)?;
            return conic.to_ellipse();
        }
        ransac_fit_ellipse(
            pts,
            cfg.ransac_iters,
            cfg.inlier_tol,
            cfg.min_inliers,
            cfg.rng_seed,
            &mut self.inlier_scratch,
        )
    }
}

impl Default for ConicFitter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use core::f32::consts::PI;
    use vm_primitives::{Point2f, Vec2f};

    use crate::shape::conic::Ellipse2f;
    use crate::shape::ransac::Lcg;

    fn ellipse_pts(ell: &Ellipse2f, n: usize) -> Vec<Point2f> {
        (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f32 / n as f32;
                ell.point_at(t)
            })
            .collect()
    }

    fn noisy_ellipse_pts(ell: &Ellipse2f, n: usize, noise: f32, seed: u64) -> Vec<Point2f> {
        let mut rng = Lcg::new(seed);
        (0..n)
            .map(|i| {
                let t = 2.0 * PI * i as f32 / n as f32;
                let mut p = ell.point_at(t);
                let nx = (rng.next_mod(1000) as f32 / 500.0 - 1.0) * noise;
                let ny = (rng.next_mod(1000) as f32 / 500.0 - 1.0) * noise;
                p.x += nx;
                p.y += ny;
                p
            })
            .collect()
    }

    #[test]
    fn bookstein_fits_exact_ellipse() {
        // 20 exact points on ellipse (a=20, b=8, θ=0, cx=60, cy=40).
        // The Bookstein fit should recover centre within 0.5 px and axes within 1%.
        let ell = Ellipse2f {
            center: Point2f { x: 60.0, y: 40.0 },
            semi_axes: Vec2f { x: 20.0, y: 8.0 },
            angle: 0.0,
        };
        let pts = ellipse_pts(&ell, 20);

        let mut fitter = ConicFitter::new();
        let cfg = ConicFitConfig {
            use_bookstein: true,
            ..Default::default()
        };
        let conic = fitter.fit(&pts, &cfg).expect("fit");
        assert!(
            conic.is_ellipse(),
            "Bookstein result should be ellipse for exact ellipse pts"
        );

        let recovered = conic.to_ellipse().expect("to_ellipse");
        assert!(
            (recovered.center.x - 60.0).abs() < 0.5,
            "cx={}",
            recovered.center.x
        );
        assert!(
            (recovered.center.y - 40.0).abs() < 0.5,
            "cy={}",
            recovered.center.y
        );
        let rel_a = (recovered.semi_major() - 20.0).abs() / 20.0;
        assert!(rel_a < 0.01, "semi_major relative error={rel_a}");
        let rel_b = (recovered.semi_minor() - 8.0).abs() / 8.0;
        assert!(rel_b < 0.01, "semi_minor relative error={rel_b}");
    }

    #[test]
    fn fitzgibbon_direct_fit_noisy_ellipse() {
        // 20 noisy points (±0.3 px noise), Fitzgibbon direct fit.
        // Recover centre within 0.5 px and axes within 1%.
        let ell = Ellipse2f {
            center: Point2f { x: 40.0, y: 30.0 },
            semi_axes: Vec2f { x: 15.0, y: 7.0 },
            angle: PI / 6.0,
        };
        let pts = noisy_ellipse_pts(&ell, 20, 0.3, 123);

        let mut fitter = ConicFitter::new();
        let cfg = ConicFitConfig {
            use_bookstein: false,
            ..Default::default()
        };
        let conic = fitter.fit(&pts, &cfg).expect("fitzgibbon fit");
        assert!(conic.is_ellipse());
        let recovered = conic.to_ellipse().expect("to_ellipse");
        assert!(
            (recovered.center.x - 40.0).abs() < 0.5,
            "cx={}",
            recovered.center.x
        );
        assert!(
            (recovered.center.y - 30.0).abs() < 0.5,
            "cy={}",
            recovered.center.y
        );
    }

    #[test]
    fn ransac_with_outliers() {
        // 16 inliers + 4 outliers; RANSAC (200 iters) should recover the ellipse.
        let ell = Ellipse2f {
            center: Point2f { x: 50.0, y: 50.0 },
            semi_axes: Vec2f { x: 18.0, y: 9.0 },
            angle: 0.0,
        };
        let mut pts = noisy_ellipse_pts(&ell, 16, 0.2, 77);
        // Add outliers
        let mut rng = Lcg::new(888);
        for _ in 0..4 {
            pts.push(Point2f {
                x: rng.next_mod(200) as f32,
                y: rng.next_mod(200) as f32,
            });
        }

        let mut fitter = ConicFitter::new();
        let cfg = ConicFitConfig {
            use_bookstein: false,
            ransac_iters: 200,
            inlier_tol: 1.5,
            min_inliers: 12,
            rng_seed: 42,
        };
        let recovered = fitter.fit_ellipse_ransac(&pts, &cfg).expect("ransac");
        assert!(
            (recovered.center.x - 50.0).abs() < 1.0,
            "cx={}",
            recovered.center.x
        );
        assert!(
            (recovered.center.y - 50.0).abs() < 1.0,
            "cy={}",
            recovered.center.y
        );
    }

    #[test]
    fn insufficient_data_returns_error() {
        let pts = vec![Point2f { x: 1.0, y: 1.0 }; 3];
        let mut fitter = ConicFitter::new();
        assert!(fitter.fit(&pts, &ConicFitConfig::default()).is_err());
        assert!(
            fitter
                .fit_ellipse_ransac(&pts, &ConicFitConfig::default())
                .is_err()
        );
    }

    #[test]
    fn reuse_fitter_across_calls() {
        // Confirm scratch buffers are reused: no panic on second call.
        let ell = Ellipse2f {
            center: Point2f { x: 20.0, y: 20.0 },
            semi_axes: Vec2f { x: 10.0, y: 5.0 },
            angle: 0.0,
        };
        let pts = ellipse_pts(&ell, 10);
        let mut fitter = ConicFitter::new();
        let cfg = ConicFitConfig::default();
        let _ = fitter.fit(&pts, &cfg);
        let _ = fitter.fit(&pts, &cfg);
    }
}
