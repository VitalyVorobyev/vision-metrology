//! Robust fitting of geometric primitives to point sets.
//!
//! Every fit returns a [`Fit<M>`] carrying not just the model but the
//! **residual statistics** a metrology user actually gates on: `rms` and
//! `max_dev` in pixels, and which input points were used. A measurement without
//! a residual is not a measurement — it is a number.
//!
//! # What separates these from an algebraic fit
//!
//! Algebraic fits minimise `Σ F(pᵢ)²` for some implicit polynomial `F`, which
//! is fast, closed-form, and **biased**: points where `‖∇F‖` is large are
//! weighted more heavily, so a partial arc pulls the centre in. Each fitter here
//! uses the algebraic solution as an *initial guess* and then refines it against
//! the true orthogonal (geometric) distance. On a 90° arc of a 100 px circle
//! with 0.2 px noise that is the difference between ~0.4 px and ~0.02 px of
//! radius bias.
//!
//! # Robustness
//!
//! [`RobustLoss`] applies iteratively-reweighted least squares on top of the
//! geometric refinement — Huber for mild contamination, Tukey to reject
//! outliers outright. For gross contamination (a wrong contour, a second edge),
//! enable [`RansacConfig`] as well: RANSAC finds the consensus set, then the
//! robust refinement polishes it.
//!
//! # Numerics
//!
//! Points are stored as `f32` (pixel coordinates never need more), but every
//! accumulation — normal equations, moments, residual sums — runs in `f64`.
//! Summing 10 000 squared coordinates near 2000 px overflows `f32`'s 24-bit
//! mantissa long before the fit converges.
//!
//! # Example
//! ```
//! use vision_metrology::fit::{FitConfig, fit_circle};
//! use vision_metrology::{Circle2f, Point2f};
//!
//! let truth = Circle2f { center: Point2f::new(50.0, 40.0), radius: 25.0 };
//! let pts: Vec<Point2f> = (0..40)
//!     .map(|i| truth.point_at(i as f32 * core::f32::consts::TAU / 40.0))
//!     .collect();
//!
//! let fit = fit_circle(&pts, &FitConfig::default()).unwrap();
//! assert!((fit.model.radius - 25.0).abs() < 1e-3);
//! assert!(fit.rms < 1e-3);
//! ```

mod circle;
mod conic;
mod ellipse;
mod line;
mod ransac;

pub use circle::fit_circle;
pub use ellipse::{fit_conic, fit_ellipse};
pub use line::fit_line;

use vm_primitives::Point2f;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Down-weighting applied to large residuals during refinement.
///
/// The tuning constant is in **pixels**, not in units of σ: metrology residuals
/// have a physical scale, and a fixed pixel threshold is easier to reason about
/// than one that moves with the data.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RobustLoss {
    /// Plain least squares. Every point counts fully.
    #[default]
    None,
    /// Quadratic within `k` px, linear beyond. Reduces the pull of moderate
    /// outliers without discarding them.
    Huber {
        /// Transition radius in pixels.
        k: f32,
    },
    /// Zero weight beyond `c` px. Rejects outliers outright, at the cost of a
    /// non-convex problem — only sound from a good starting guess, which is why
    /// it runs after the algebraic initialisation.
    Tukey {
        /// Rejection radius in pixels.
        c: f32,
    },
}

impl RobustLoss {
    /// The loss to use on IRLS iteration `iter`, given the largest residual
    /// seen at the start.
    ///
    /// Tukey is non-convex, so it only converges from a starting guess that is
    /// already close. But the algebraic initialisation is computed from *all*
    /// the points, outliers included — so a small rejection radius applied
    /// immediately throws away the inliers and keeps whatever the corrupted fit
    /// happened to pass through. (Measured: one outlier 70 px off a 40-point
    /// circle left a Tukey fit standing on 2 points.)
    ///
    /// Graduated non-convexity fixes it: start with a radius that admits every
    /// point, shrink geometrically, and stop at the configured value. Huber is
    /// convex and needs none of this, so it is returned unchanged.
    #[inline]
    pub(crate) fn annealed(self, iter: usize, max_residual: f32) -> Self {
        match self {
            Self::Tukey { c } => {
                let start = max_residual.max(c);
                Self::Tukey {
                    c: (start * 0.6f32.powi(iter as i32)).max(c),
                }
            }
            other => other,
        }
    }

    /// IRLS weight for a residual of `r` pixels.
    #[inline]
    fn weight(self, r: f32) -> f64 {
        let r = r.abs() as f64;
        match self {
            Self::None => 1.0,
            Self::Huber { k } => {
                let k = k.max(1e-6) as f64;
                if r <= k { 1.0 } else { k / r }
            }
            Self::Tukey { c } => {
                let c = c.max(1e-6) as f64;
                if r >= c {
                    0.0
                } else {
                    let u = 1.0 - (r / c) * (r / c);
                    u * u
                }
            }
        }
    }
}

/// Optional RANSAC pre-stage for grossly contaminated point sets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RansacConfig {
    /// Hypotheses to draw.
    pub iters: usize,
    /// Orthogonal distance, in pixels, within which a point is an inlier.
    pub inlier_tol: f32,
    /// Reject the fit unless at least this many points agree.
    pub min_inliers: usize,
    /// Seed for the internal LCG. The same seed always gives the same answer —
    /// library code carries no ambient randomness (invariant 12).
    pub seed: u64,
}

impl Default for RansacConfig {
    fn default() -> Self {
        Self {
            iters: 200,
            inlier_tol: 1.0,
            min_inliers: 5,
            seed: 42,
        }
    }
}

/// How a primitive is fitted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitConfig {
    /// Down-weighting applied during geometric refinement.
    pub loss: RobustLoss,
    /// RANSAC consensus stage before refinement. `None` fits all points.
    pub ransac: Option<RansacConfig>,
    /// Maximum refinement iterations. `0` returns the algebraic solution.
    pub max_iters: usize,
    /// Stop when the parameter update falls below this, in pixels.
    pub tol: f32,
}

impl Default for FitConfig {
    fn default() -> Self {
        Self {
            loss: RobustLoss::None,
            ransac: None,
            max_iters: 20,
            tol: 1e-4,
        }
    }
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

/// A fitted model together with the residual statistics that qualify it.
#[derive(Debug, Clone, PartialEq)]
pub struct Fit<M> {
    /// The fitted primitive.
    pub model: M,
    /// Root-mean-square orthogonal distance over the points used, in pixels.
    pub rms: f32,
    /// Largest orthogonal distance over the points used, in pixels.
    ///
    /// A form-tolerance check reads this, not `rms`: roundness is a worst-case
    /// property.
    pub max_dev: f32,
    /// How many points the final fit used.
    pub n_used: usize,
    /// Indices into the input slice of the points that were used.
    ///
    /// Empty when every point was used and no loss discarded any — check
    /// `n_used` against the input length rather than assuming.
    pub inliers: Vec<u32>,
}

impl<M> Fit<M> {
    /// Build the residual summary for `model` over `pts`, given a distance fn.
    pub(crate) fn summarize(
        model: M,
        pts: &[Point2f],
        inliers: Vec<u32>,
        dist: impl Fn(&M, Point2f) -> f32,
    ) -> Self {
        let idx: Vec<u32> = if inliers.is_empty() {
            (0..pts.len() as u32).collect()
        } else {
            inliers
        };
        let mut sum_sq = 0.0f64;
        let mut max_dev = 0.0f32;
        for &i in &idx {
            let d = dist(&model, pts[i as usize]).abs();
            sum_sq += (d as f64) * (d as f64);
            max_dev = max_dev.max(d);
        }
        let n = idx.len().max(1);
        Self {
            model,
            rms: (sum_sq / n as f64).sqrt() as f32,
            max_dev,
            n_used: idx.len(),
            inliers: if idx.len() == pts.len() {
                Vec::new()
            } else {
                idx
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RobustLoss;

    #[test]
    fn plain_least_squares_weights_everything_equally() {
        for r in [0.0f32, 1.0, 100.0] {
            assert_eq!(RobustLoss::None.weight(r), 1.0);
        }
    }

    #[test]
    fn huber_is_flat_inside_and_decays_outside() {
        let h = RobustLoss::Huber { k: 2.0 };
        assert_eq!(h.weight(0.5), 1.0);
        assert_eq!(h.weight(2.0), 1.0);
        // Beyond k the weight is k/r, so the *influence* r*w stays constant.
        assert!((h.weight(4.0) - 0.5).abs() < 1e-12);
        assert!((h.weight(8.0) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn tukey_rejects_beyond_its_radius() {
        let t = RobustLoss::Tukey { c: 3.0 };
        assert_eq!(t.weight(0.0), 1.0);
        assert_eq!(t.weight(3.0), 0.0);
        assert_eq!(t.weight(10.0), 0.0);
        // Strictly decreasing in between.
        assert!(t.weight(1.0) > t.weight(2.0));
        assert!(t.weight(2.0) > 0.0);
    }

    /// Annealing must start permissive and finish at the configured radius.
    #[test]
    fn tukey_annealing_starts_wide_and_lands_on_the_target() {
        let t = RobustLoss::Tukey { c: 2.0 };
        // Iteration 0 admits the worst point seen.
        let RobustLoss::Tukey { c: c0 } = t.annealed(0, 70.0) else {
            panic!("stays Tukey")
        };
        assert!(
            c0 >= 70.0,
            "first radius {c0} should admit a 70 px residual"
        );
        // It shrinks monotonically...
        let radii: Vec<f32> = (0..12)
            .map(|i| match t.annealed(i, 70.0) {
                RobustLoss::Tukey { c } => c,
                _ => unreachable!(),
            })
            .collect();
        for w in radii.windows(2) {
            assert!(w[1] <= w[0], "radii must not grow: {radii:?}");
        }
        // ...and lands exactly on the configured radius, never below it.
        assert_eq!(*radii.last().unwrap(), 2.0);
        assert!(radii.iter().all(|&c| c >= 2.0));
    }

    /// Huber is convex; annealing it would be pointless and is not done.
    #[test]
    fn huber_and_none_are_not_annealed() {
        let h = RobustLoss::Huber { k: 1.5 };
        assert_eq!(h.annealed(0, 99.0), h);
        assert_eq!(RobustLoss::None.annealed(3, 99.0), RobustLoss::None);
    }

    /// Sign must not matter — residuals are two-sided.
    #[test]
    fn weights_are_symmetric_in_the_residual() {
        for loss in [
            RobustLoss::Huber { k: 1.5 },
            RobustLoss::Tukey { c: 4.0 },
            RobustLoss::None,
        ] {
            for r in [0.3f32, 1.0, 2.5, 9.0] {
                assert_eq!(loss.weight(r), loss.weight(-r));
            }
        }
    }
}
