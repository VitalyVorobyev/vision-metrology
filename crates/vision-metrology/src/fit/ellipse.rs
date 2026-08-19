//! Ellipse fitting: algebraic initialisation followed by geometric refinement.

use vm_primitives::{Conic2f, Ellipse2f, Error, Point2f};

use super::conic::{fit_bookstein, fit_fitzgibbon};
use super::ransac::ransac_fit_ellipse;
use super::{Fit, FitConfig, RobustLoss};

/// Fit a general conic `A x² + B x y + C y² + D x + E y + F = 0`.
///
/// Uses the Bookstein `‖c‖ = 1` constraint, which is similarity-invariant and
/// admits any conic — use it when the data may not be an ellipse. For an
/// ellipse specifically, prefer [`fit_ellipse`].
///
/// # Errors
/// - [`Error::InsufficientData`] with fewer than 5 points.
/// - [`Error::Degenerate`] when the point set is numerically degenerate.
pub fn fit_conic(pts: &[Point2f]) -> Result<Conic2f, Error> {
    fit_bookstein(pts)
}

/// Fit an ellipse, minimising approximate orthogonal distance.
///
/// Two stages, mirroring [`fit_circle`](super::fit_circle):
///
/// 1. **Fitzgibbon** — the `4AC − B² = 1` constraint guarantees the algebraic
///    solution is an ellipse rather than a hyperbola, which is what makes it
///    usable as a starting point without a validity check.
/// 2. **Refinement** — iteratively reweighted re-fitting against the Sampson
///    distance `|F(p)| / ‖∇F(p)‖`, a first-order approximation to the true
///    orthogonal distance that is accurate to well under a tenth of a pixel at
///    metrology noise levels and costs no root-finding.
///
/// `Fit::rms` and `Fit::max_dev` are reported in those same units, so they are
/// directly comparable to a circle fit's.
///
/// # Errors
/// - [`Error::InsufficientData`] with fewer than 5 points.
/// - [`Error::Degenerate`] when the fit is not an ellipse, or RANSAC finds no
///   consensus.
///
/// # Example
/// ```
/// use vision_metrology::fit::{FitConfig, fit_ellipse};
/// use vision_metrology::{Ellipse2f, Point2f, Vec2f};
///
/// let truth = Ellipse2f {
///     center: Point2f::new(60.0, 45.0),
///     semi_axes: Vec2f::new(30.0, 18.0),
///     angle: 0.4,
/// };
/// let pts: Vec<Point2f> = (0..40)
///     .map(|i| truth.point_at(i as f32 * core::f32::consts::TAU / 40.0))
///     .collect();
///
/// let fit = fit_ellipse(&pts, &FitConfig::default()).unwrap();
/// assert!((fit.model.semi_major() - 30.0).abs() < 0.05);
/// assert!(fit.rms < 0.05);
/// ```
pub fn fit_ellipse(pts: &[Point2f], cfg: &FitConfig) -> Result<Fit<Ellipse2f>, Error> {
    if pts.len() < 5 {
        return Err(Error::InsufficientData {
            need: 5,
            got: pts.len(),
        });
    }

    let (subset, inliers) = match cfg.ransac {
        Some(rc) => {
            let mut scratch = Vec::new();
            let e = ransac_fit_ellipse(
                pts,
                rc.iters,
                rc.inlier_tol,
                rc.min_inliers,
                rc.seed,
                &mut scratch,
            )?;
            let idx: Vec<u32> = pts
                .iter()
                .enumerate()
                .filter(|&(_, &p)| sampson(&e, p).is_some_and(|d| d.abs() <= rc.inlier_tol))
                .map(|(i, _)| i as u32)
                .collect();
            if idx.len() < 5 {
                return Err(Error::Degenerate("ellipse fit: RANSAC consensus too small"));
            }
            (
                idx.iter().map(|&i| pts[i as usize]).collect::<Vec<_>>(),
                idx,
            )
        }
        None => (pts.to_vec(), Vec::new()),
    };

    let mut ellipse = fit_fitzgibbon(&subset)?.to_ellipse()?;
    let mut weights = vec![1.0f64; subset.len()];

    if cfg.loss != RobustLoss::None {
        for _ in 0..cfg.max_iters {
            for (w, &p) in weights.iter_mut().zip(&subset) {
                *w = match sampson(&ellipse, p) {
                    Some(d) => cfg.loss.weight(d),
                    None => 0.0,
                };
            }
            // Re-fit on the points that still carry weight. Fitzgibbon has no
            // weighted form, so approximate IRLS by dropping zero-weight points
            // — which is exactly what Tukey means, and is a no-op for Huber
            // until a residual exceeds its knee.
            let kept: Vec<Point2f> = subset
                .iter()
                .zip(&weights)
                .filter(|&(_, &w)| w > 0.0)
                .map(|(&p, _)| p)
                .collect();
            if kept.len() < 5 {
                break;
            }
            let Ok(next) = fit_fitzgibbon(&kept).and_then(|c| c.to_ellipse()) else {
                break;
            };
            let moved =
                (next.center - ellipse.center).norm() + (next.semi_axes - ellipse.semi_axes).norm();
            ellipse = next;
            if moved < cfg.tol {
                break;
            }
        }
    }

    let used: Vec<u32> = if cfg.loss == RobustLoss::None {
        inliers
    } else if inliers.is_empty() {
        weights
            .iter()
            .enumerate()
            .filter(|&(_, &w)| w > 0.0)
            .map(|(i, _)| i as u32)
            .collect()
    } else {
        inliers
            .iter()
            .zip(&weights)
            .filter(|&(_, &w)| w > 0.0)
            .map(|(&i, _)| i)
            .collect()
    };

    Ok(Fit::summarize(ellipse, pts, used, |e, p| {
        sampson(e, p).unwrap_or(0.0)
    }))
}

/// Sampson distance from `p` to the ellipse: `F(p) / ‖∇F(p)‖`.
///
/// `None` when the ellipse cannot be expressed as a conic or the gradient
/// vanishes (`p` at the centre).
pub(crate) fn sampson(e: &Ellipse2f, p: Point2f) -> Option<f32> {
    let c = e.to_conic().ok()?;
    let g = c.grad_norm(p);
    (g > 1e-8).then(|| c.eval(p) / g)
}

#[cfg(test)]
mod tests {
    use super::{fit_conic, fit_ellipse, sampson};
    use crate::fit::{FitConfig, RansacConfig, RobustLoss};
    use vm_primitives::{Ellipse2f, Point2f, Vec2f};

    fn on_ellipse(e: &Ellipse2f, n: usize) -> Vec<Point2f> {
        (0..n)
            .map(|i| e.point_at(i as f32 * core::f32::consts::TAU / n as f32))
            .collect()
    }

    #[test]
    fn recovers_an_exact_ellipse() {
        let truth = Ellipse2f {
            center: Point2f::new(80.0, 60.0),
            semi_axes: Vec2f::new(40.0, 25.0),
            angle: 0.7,
        };
        let pts = on_ellipse(&truth, 48);
        let fit = fit_ellipse(&pts, &FitConfig::default()).expect("fit");

        assert!((fit.model.center - truth.center).norm() < 0.05);
        assert!((fit.model.semi_major() - 40.0).abs() < 0.05);
        assert!((fit.model.semi_minor() - 25.0).abs() < 0.05);
        assert!(fit.rms < 0.05, "rms = {}", fit.rms);
    }

    /// A circle is an ellipse with equal axes; the fit must not diverge there.
    #[test]
    fn a_circle_is_a_valid_ellipse() {
        let truth = Ellipse2f {
            center: Point2f::new(30.0, 30.0),
            semi_axes: Vec2f::new(20.0, 20.0),
            angle: 0.0,
        };
        let fit = fit_ellipse(&on_ellipse(&truth, 36), &FitConfig::default()).expect("fit");
        assert!((fit.model.semi_major() - 20.0).abs() < 0.05);
        assert!((fit.model.semi_minor() - 20.0).abs() < 0.05);
    }

    #[test]
    fn ransac_plus_tukey_rejects_a_contaminating_arc() {
        let truth = Ellipse2f {
            center: Point2f::new(150.0, 120.0),
            semi_axes: Vec2f::new(60.0, 35.0),
            angle: 0.3,
        };
        let mut pts = on_ellipse(&truth, 40);
        for i in 0..14 {
            pts.push(Point2f::new(40.0 + i as f32 * 3.0, 30.0));
        }

        let fit = fit_ellipse(
            &pts,
            &FitConfig {
                ransac: Some(RansacConfig {
                    iters: 500,
                    inlier_tol: 1.0,
                    min_inliers: 20,
                    seed: 11,
                }),
                loss: RobustLoss::Tukey { c: 2.0 },
                ..FitConfig::default()
            },
        )
        .expect("fit");

        assert!(
            (fit.model.center - truth.center).norm() < 1.0,
            "c={:?}",
            fit.model.center
        );
        assert!((fit.model.semi_major() - 60.0).abs() < 1.0);
        assert!(
            fit.n_used >= 35 && fit.n_used <= 45,
            "n_used = {}",
            fit.n_used
        );
    }

    #[test]
    fn sampson_is_zero_on_the_curve_and_grows_off_it() {
        let e = Ellipse2f {
            center: Point2f::new(0.0, 0.0),
            semi_axes: Vec2f::new(10.0, 5.0),
            angle: 0.0,
        };
        assert!(sampson(&e, Point2f::new(10.0, 0.0)).unwrap().abs() < 1e-3);
        assert!(sampson(&e, Point2f::new(0.0, 5.0)).unwrap().abs() < 1e-3);
        let d = sampson(&e, Point2f::new(12.0, 0.0)).unwrap().abs();
        assert!((d - 2.0).abs() < 0.2, "expected ~2 px, got {d}");
    }

    #[test]
    fn too_few_points_is_an_error() {
        let pts = vec![Point2f::new(0.0, 0.0); 4];
        assert!(fit_ellipse(&pts, &FitConfig::default()).is_err());
        assert!(fit_conic(&pts).is_err());
    }

    #[test]
    fn a_general_conic_fit_accepts_non_ellipses() {
        // Points on the hyperbola x*y = 1.
        let pts: Vec<Point2f> = (1..8)
            .map(|i| {
                let x = i as f32 * 0.5;
                Point2f::new(x, 1.0 / x)
            })
            .collect();
        let c = fit_conic(&pts).expect("bookstein handles any conic");
        assert!(!c.is_ellipse(), "x*y = 1 is a hyperbola");
    }
}
