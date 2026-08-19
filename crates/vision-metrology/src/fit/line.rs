//! Total-least-squares line fitting with optional robust reweighting.

use vm_primitives::{Error, Line2f, Point2f, Vec2f, Vec2fExt};

use super::ransac::Lcg;
use super::{Fit, FitConfig, RobustLoss};

/// Orthogonal distance from `p` to the line, signed by which side it falls on.
#[inline]
pub(crate) fn line_distance(l: &Line2f, p: Point2f) -> f32 {
    // `dir` is kept unit-length by the fitter, so the cross product is already
    // the perpendicular distance.
    let d = p - l.p;
    l.dir.x * d.y - l.dir.y * d.x
}

/// Fit a line by total least squares, optionally reweighted.
///
/// Minimises the sum of squared **orthogonal** distances, not vertical ones, so
/// the result does not depend on how the point set is oriented — a near-vertical
/// edge fits as well as a near-horizontal one. This is the eigenvector of the
/// scatter matrix belonging to its smaller eigenvalue, which for 2×2 has a
/// closed form.
///
/// With a [`RobustLoss`](super::RobustLoss) the solve is repeated with IRLS
/// weights until the direction stops moving.
///
/// The returned `dir` is unit length, and `p` is the (weighted) centroid, so
/// `Fit::rms` is directly the RMS orthogonal deviation.
///
/// # Errors
/// - [`Error::InsufficientData`] with fewer than 2 points.
/// - [`Error::Degenerate`] when the points are coincident, or when a robust loss
///   has down-weighted all but one of them.
///
/// # Example
/// ```
/// use vision_metrology::fit::{FitConfig, fit_line};
/// use vision_metrology::Point2f;
///
/// // y = 2 with a little scatter.
/// let pts: Vec<Point2f> = (0..10)
///     .map(|i| Point2f::new(i as f32, 2.0 + if i % 2 == 0 { 0.01 } else { -0.01 }))
///     .collect();
///
/// let fit = fit_line(&pts, &FitConfig::default()).unwrap();
/// assert!(fit.model.dir.y.abs() < 1e-3, "should be horizontal");
/// assert!((fit.rms - 0.01).abs() < 1e-3);
/// ```
pub fn fit_line(pts: &[Point2f], cfg: &FitConfig) -> Result<Fit<Line2f>, Error> {
    if pts.len() < 2 {
        return Err(Error::InsufficientData {
            need: 2,
            got: pts.len(),
        });
    }

    // A single gross outlier can flip the principal axis outright: 30 points
    // on y = 10 plus one at (15, 60) makes the y-spread exceed the x-spread, so
    // total least squares returns a near-*vertical* line. No amount of
    // reweighting recovers from that — the starting guess is not merely
    // inaccurate, it is orthogonal. RANSAC is the tool for that case, which is
    // why `cfg.ransac` is honoured here and not quietly ignored.
    let consensus = match cfg.ransac {
        Some(rc) => Some(ransac_consensus(pts, &rc)?),
        None => None,
    };
    let subset: Vec<Point2f> = match &consensus {
        Some(idx) => idx.iter().map(|&i| pts[i as usize]).collect(),
        None => pts.to_vec(),
    };

    let mut weights = vec![1.0f64; subset.len()];
    let mut line = solve_weighted(&subset, &weights)?;

    let max_resid = subset
        .iter()
        .map(|&p| line_distance(&line, p).abs())
        .fold(0.0f32, f32::max);

    for iter in 0..cfg.max_iters {
        if cfg.loss == RobustLoss::None {
            break;
        }
        // Annealed — see `RobustLoss::annealed`: a fixed small Tukey radius
        // applied to a contaminated initial fit rejects the inliers.
        let loss = cfg.loss.annealed(iter, max_resid);
        for (w, &p) in weights.iter_mut().zip(&subset) {
            *w = loss.weight(line_distance(&line, p));
        }
        let next = solve_weighted(&subset, &weights)?;
        // Converged when the direction and the anchor both stop moving — and
        // only once annealing has reached the configured radius.
        let moved = (next.p - line.p).norm() + (next.dir - line.dir).norm();
        line = next;
        if moved < cfg.tol && loss == cfg.loss {
            break;
        }
    }

    if cfg.loss != RobustLoss::None {
        for (w, &p) in weights.iter_mut().zip(&subset) {
            *w = cfg.loss.weight(line_distance(&line, p));
        }
    }

    // A zero-weight point took no part in the fit and must not count towards
    // the reported residuals either. Indices are into the *original* slice.
    let used: Vec<u32> = match &consensus {
        Some(idx) => idx
            .iter()
            .zip(&weights)
            .filter(|&(_, &w)| w > 0.0)
            .map(|(&i, _)| i)
            .collect(),
        None => weights
            .iter()
            .enumerate()
            .filter(|&(_, &w)| w > 0.0)
            .map(|(i, _)| i as u32)
            .collect(),
    };

    Ok(Fit::summarize(line, pts, used, line_distance))
}

/// One weighted total-least-squares solve.
///
/// The scatter matrix is accumulated in `f64`: for a 2000 px image the raw
/// second moments reach 4·10⁶ per point, and `f32` loses the variation that
/// carries the direction.
fn solve_weighted(pts: &[Point2f], w: &[f64]) -> Result<Line2f, Error> {
    let (mut sw, mut sx, mut sy) = (0.0f64, 0.0f64, 0.0f64);
    for (&p, &wi) in pts.iter().zip(w) {
        sw += wi;
        sx += wi * p.x as f64;
        sy += wi * p.y as f64;
    }
    if sw <= 0.0 {
        return Err(Error::Degenerate("line fit: all points were down-weighted"));
    }
    let (cx, cy) = (sx / sw, sy / sw);

    let (mut sxx, mut sxy, mut syy) = (0.0f64, 0.0f64, 0.0f64);
    for (&p, &wi) in pts.iter().zip(w) {
        let dx = p.x as f64 - cx;
        let dy = p.y as f64 - cy;
        sxx += wi * dx * dx;
        sxy += wi * dx * dy;
        syy += wi * dy * dy;
    }

    if sxx + syy <= f64::EPSILON {
        return Err(Error::Degenerate("line fit: points are coincident"));
    }

    // Principal direction: eigenvector of [[sxx, sxy], [sxy, syy]] for the
    // larger eigenvalue. The line runs along it; the residual is along the other.
    let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
    let dir = Vec2f::new(theta.cos() as f32, theta.sin() as f32);

    Ok(Line2f {
        p: Point2f::new(cx as f32, cy as f32),
        dir,
    })
}

/// RANSAC over lines through two points, returning the consensus indices.
fn ransac_consensus(pts: &[Point2f], rc: &super::RansacConfig) -> Result<Vec<u32>, Error> {
    let n = pts.len();
    let mut rng = Lcg::new(rc.seed);
    let mut best: Vec<u32> = Vec::new();

    for _ in 0..rc.iters {
        let (i, j) = (rng.next_mod(n), rng.next_mod(n));
        if i == j {
            continue;
        }
        let dir = (pts[j] - pts[i]).normalized_or_zero();
        if dir.norm() < 0.5 {
            continue;
        }
        let hyp = Line2f { p: pts[i], dir };
        let inl: Vec<u32> = pts
            .iter()
            .enumerate()
            .filter(|&(_, &p)| line_distance(&hyp, p).abs() <= rc.inlier_tol)
            .map(|(k, _)| k as u32)
            .collect();
        if inl.len() > best.len() {
            best = inl;
        }
    }

    if best.len() < rc.min_inliers.max(2) {
        return Err(Error::Degenerate("line fit: RANSAC found no consensus"));
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::{fit_line, line_distance};
    use crate::fit::{FitConfig, RansacConfig, RobustLoss};
    use vm_primitives::{Line2f, Point2f, Vec2f};

    fn on_line(a: Point2f, dir: Vec2f, n: usize, step: f32) -> Vec<Point2f> {
        (0..n).map(|i| a + dir * (i as f32 * step)).collect()
    }

    #[test]
    fn recovers_an_exact_line_at_any_orientation() {
        for deg in [0.0f32, 17.0, 45.0, 89.0, 90.0, 134.0, 179.0] {
            let th = deg.to_radians();
            let dir = Vec2f::new(th.cos(), th.sin());
            let anchor = Point2f::new(37.0, -12.0);
            let pts = on_line(anchor, dir, 25, 3.0);

            let fit = fit_line(&pts, &FitConfig::default()).expect("fit");
            // Direction is defined up to sign.
            let dot = fit.model.dir.dot(&dir).abs();
            assert!((dot - 1.0).abs() < 1e-4, "deg={deg}: dir dot = {dot}");
            assert!(fit.rms < 1e-3, "deg={deg}: rms = {}", fit.rms);
            assert!(fit.max_dev < 1e-3);
            assert_eq!(fit.n_used, 25);
        }
    }

    /// A near-vertical line is the classic failure of an `y = mx + c` fit.
    /// Total least squares has no preferred axis.
    #[test]
    fn vertical_lines_are_not_special() {
        let pts: Vec<Point2f> = (0..20).map(|i| Point2f::new(5.0, i as f32)).collect();
        let fit = fit_line(&pts, &FitConfig::default()).expect("fit");
        assert!(fit.model.dir.x.abs() < 1e-5, "dir = {:?}", fit.model.dir);
        assert!(fit.rms < 1e-5);
        assert!((fit.model.p.x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn rms_tracks_the_injected_deviation() {
        // Two points at every x, one 0.25 px above y = 0 and one below. Pairing
        // them at the same x makes the cross-moment exactly zero, so the fitted
        // line is exactly y = 0 and every residual is exactly 0.25. Alternating
        // the sign by index instead leaves a small correlation that tilts the
        // line by ~0.001 rad — a real effect, not a fitter bug, and the reason
        // this fixture is paired.
        let pts: Vec<Point2f> = (0..20)
            .flat_map(|i| [Point2f::new(i as f32, 0.25), Point2f::new(i as f32, -0.25)])
            .collect();
        let fit = fit_line(&pts, &FitConfig::default()).expect("fit");
        assert!(fit.model.dir.y.abs() < 1e-6, "dir = {:?}", fit.model.dir);
        assert!((fit.rms - 0.25).abs() < 1e-4, "rms = {}", fit.rms);
        assert!(
            (fit.max_dev - 0.25).abs() < 1e-4,
            "max_dev = {}",
            fit.max_dev
        );
    }

    /// Mild contamination: several points scattered a few pixels off. IRLS
    /// alone handles this, because the starting fit is still roughly right.
    #[test]
    fn tukey_suppresses_mild_contamination() {
        let mut pts: Vec<Point2f> = (0..40).map(|i| Point2f::new(i as f32, 10.0)).collect();
        for (k, i) in [3usize, 11, 27].into_iter().enumerate() {
            pts[i] = Point2f::new(i as f32, 10.0 + 4.0 + k as f32);
        }

        let plain = fit_line(&pts, &FitConfig::default()).expect("fit");
        assert!(plain.rms > 0.5, "plain fit should feel them: {}", plain.rms);

        let robust = fit_line(
            &pts,
            &FitConfig {
                loss: RobustLoss::Tukey { c: 2.0 },
                ..FitConfig::default()
            },
        )
        .expect("robust fit");
        assert!(robust.model.dir.y.abs() < 1e-3, "should be horizontal");
        assert!(robust.rms < 1e-3, "rms = {}", robust.rms);
        assert_eq!(robust.n_used, 37, "the three strays must be excluded");
    }

    /// Gross contamination flips the principal axis, and IRLS cannot recover:
    /// 30 points on y = 10 plus one at (15, 60) gives the outlier enough
    /// y-variance that total least squares returns a near-*vertical* line. The
    /// starting guess is not inaccurate, it is orthogonal. That is what RANSAC
    /// is for, and it is why `FitConfig::ransac` is honoured by `fit_line`.
    #[test]
    fn a_single_gross_outlier_needs_ransac_not_reweighting() {
        let mut pts: Vec<Point2f> = (0..30).map(|i| Point2f::new(i as f32, 10.0)).collect();
        pts.push(Point2f::new(15.0, 60.0));

        // The unweighted fit really does flip to vertical.
        let plain = fit_line(&pts, &FitConfig::default()).expect("fit");
        assert!(
            plain.model.dir.x.abs() < 0.2,
            "expected a near-vertical fit, got dir {:?}",
            plain.model.dir
        );

        let ransac = fit_line(
            &pts,
            &FitConfig {
                ransac: Some(RansacConfig {
                    iters: 200,
                    inlier_tol: 1.0,
                    min_inliers: 10,
                    seed: 3,
                }),
                loss: RobustLoss::Tukey { c: 2.0 },
                ..FitConfig::default()
            },
        )
        .expect("ransac fit");
        assert!(
            ransac.model.dir.y.abs() < 1e-3,
            "should be horizontal, got {:?}",
            ransac.model.dir
        );
        assert!(ransac.rms < 1e-3, "rms = {}", ransac.rms);
        assert_eq!(ransac.n_used, 30, "the outlier must be excluded");
    }

    #[test]
    fn distance_is_orthogonal_and_signed() {
        let l = Line2f {
            p: Point2f::new(0.0, 0.0),
            dir: Vec2f::new(1.0, 0.0),
        };
        assert!((line_distance(&l, Point2f::new(5.0, 3.0)) - 3.0).abs() < 1e-6);
        assert!((line_distance(&l, Point2f::new(5.0, -3.0)) + 3.0).abs() < 1e-6);
        assert!(line_distance(&l, Point2f::new(100.0, 0.0)).abs() < 1e-6);
    }

    #[test]
    fn too_few_points_is_an_error() {
        assert!(fit_line(&[], &FitConfig::default()).is_err());
        assert!(fit_line(&[Point2f::new(0.0, 0.0)], &FitConfig::default()).is_err());
    }

    #[test]
    fn coincident_points_are_degenerate() {
        let pts = vec![Point2f::new(3.0, 3.0); 8];
        assert!(fit_line(&pts, &FitConfig::default()).is_err());
    }
}
