//! Circle fitting: Taubin initialisation followed by geometric refinement.
//!
//! Hole diameters, boss positions and rim fits are the most common measurements
//! in industrial metrology, and this is the fit that produces them.

use vm_primitives::{Circle2f, Error, Point2f};

use super::ransac::Lcg;
use super::{Fit, FitConfig, RobustLoss};

/// Fit a circle, minimising orthogonal distance.
///
/// Two stages:
///
/// 1. **Taubin** — a closed-form algebraic fit that is very nearly unbiased even
///    on short arcs, unlike the simpler Kåsa/Delogne fit which collapses towards
///    the chord. No iteration, no starting guess.
/// 2. **Geometric refinement** — Gauss–Newton on the true residual
///    `‖p − c‖ − r`, with IRLS weights from `cfg.loss`. This is what makes the
///    result a *measurement*: the algebraic stage minimises the wrong quantity,
///    and on a partial arc the difference is tens of times the noise.
///
/// With `cfg.ransac` set, a consensus stage runs first and the refinement then
/// polishes only the inliers.
///
/// # Errors
/// - [`Error::InsufficientData`] with fewer than 3 points.
/// - [`Error::Degenerate`] when the points are collinear or coincident, or when
///   RANSAC never reaches `min_inliers`.
///
/// # Example
/// ```
/// use vision_metrology::fit::{FitConfig, fit_circle};
/// use vision_metrology::{Circle2f, Point2f};
///
/// // A 60 degree arc — the case that exposes a biased fitter.
/// let truth = Circle2f { center: Point2f::new(120.0, 90.0), radius: 60.0 };
/// let pts: Vec<Point2f> = (0..25)
///     .map(|i| truth.point_at(i as f32 * 60.0f32.to_radians() / 24.0))
///     .collect();
///
/// let fit = fit_circle(&pts, &FitConfig::default()).unwrap();
/// assert!((fit.model.radius - 60.0).abs() < 0.01, "r = {}", fit.model.radius);
/// ```
pub fn fit_circle(pts: &[Point2f], cfg: &FitConfig) -> Result<Fit<Circle2f>, Error> {
    if pts.len() < 3 {
        return Err(Error::InsufficientData {
            need: 3,
            got: pts.len(),
        });
    }

    let (subset, inliers) = match cfg.ransac {
        Some(rc) => {
            let idx = ransac_consensus(pts, &rc)?;
            (
                idx.iter().map(|&i| pts[i as usize]).collect::<Vec<_>>(),
                idx,
            )
        }
        None => (pts.to_vec(), Vec::new()),
    };

    let mut circle = taubin(&subset)?;
    let mut weights = vec![1.0f64; subset.len()];

    // Largest residual of the algebraic fit — the starting radius for Tukey's
    // annealing, so the first iteration cannot reject the inliers.
    let max_resid = subset
        .iter()
        .map(|&p| circle.signed_distance(p).abs())
        .fold(0.0f32, f32::max);

    for iter in 0..cfg.max_iters {
        if cfg.loss != RobustLoss::None {
            let loss = cfg.loss.annealed(iter, max_resid);
            for (w, &p) in weights.iter_mut().zip(&subset) {
                *w = loss.weight(circle.signed_distance(p));
            }
        }
        let next = gauss_newton_step(&subset, &weights, circle)?;
        let moved = (next.center - circle.center).norm() + (next.radius - circle.radius).abs();
        circle = next;
        // Only stop once the annealing has reached the configured radius,
        // otherwise a converged *intermediate* problem ends the loop early.
        let annealed_done = cfg.loss.annealed(iter, max_resid) == cfg.loss;
        if moved < cfg.tol && annealed_done {
            break;
        }
    }

    // Final weights at the configured radius, so `n_used` reflects the loss the
    // caller asked for rather than whatever the last annealing step used.
    if cfg.loss != RobustLoss::None {
        for (w, &p) in weights.iter_mut().zip(&subset) {
            *w = cfg.loss.weight(circle.signed_distance(p));
        }
    }

    // Points a robust loss zeroed out did not shape the fit and must not shape
    // its reported residuals either.
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

    Ok(Fit::summarize(circle, pts, used, |c, p| {
        c.signed_distance(p)
    }))
}

/// Taubin's algebraic circle fit.
///
/// Solves `Σ (z − z̄ + …)` in the normalised form that makes the constraint
/// scale-invariant; the practical effect is a fit that stays centred on short
/// arcs where the Kåsa fit is badly biased. Everything accumulates in `f64`,
/// and coordinates are centred first so the fourth moments stay conditioned.
fn taubin(pts: &[Point2f]) -> Result<Circle2f, Error> {
    let n = pts.len() as f64;
    let (mut mx, mut my) = (0.0f64, 0.0f64);
    for p in pts {
        mx += p.x as f64;
        my += p.y as f64;
    }
    mx /= n;
    my /= n;

    let (mut mzz, mut mxx, mut myy, mut mxy, mut mxz, mut myz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in pts {
        let x = p.x as f64 - mx;
        let y = p.y as f64 - my;
        let z = x * x + y * y;
        mxx += x * x;
        myy += y * y;
        mxy += x * y;
        mxz += x * z;
        myz += y * z;
        mzz += z * z;
    }
    mxx /= n;
    myy /= n;
    mxy /= n;
    mxz /= n;
    myz /= n;
    mzz /= n;

    // Characteristic polynomial of the Taubin constraint, solved by Newton
    // from 0 — the root of interest is the smallest positive one.
    let cov_xy = mxx + myy;
    let a0 = mxz * (mxz * myy - myz * mxy) + myz * (myz * mxx - mxz * mxy)
        - (mzz - cov_xy * cov_xy) * (mxx * myy - mxy * mxy);
    let a1 = (mzz - cov_xy * cov_xy) * cov_xy + 4.0 * cov_xy * (mxx * myy - mxy * mxy)
        - mxz * mxz
        - myz * myz;
    let a2 = -3.0 * cov_xy * cov_xy - (mzz - cov_xy * cov_xy) - cov_xy * cov_xy;
    let a22 = a2 + a2;

    let mut x = 0.0f64;
    let mut y = a0;
    for _ in 0..64 {
        let dy = a1 + x * (a22 + 16.0 * x * x);
        if dy.abs() < 1e-300 {
            break;
        }
        let x_new = x - y / dy;
        if x_new == x || !x_new.is_finite() {
            break;
        }
        let y_new = a0 + x_new * (a1 + x_new * (a2 + 4.0 * x_new * x_new));
        if y_new.abs() >= y.abs() {
            break;
        }
        x = x_new;
        y = y_new;
    }

    let det = 2.0 * (mxx * myy - mxy * mxy) - 2.0 * x * (mxx + myy) + 4.0 * x * x;
    if det.abs() < 1e-12 {
        return Err(Error::Degenerate(
            "circle fit: points are collinear or coincident",
        ));
    }
    let cx = (mxz * (myy - x) - myz * mxy) / det;
    let cy = (myz * (mxx - x) - mxz * mxy) / det;
    let r = (cx * cx + cy * cy + cov_xy - x - x).max(0.0).sqrt();

    let circle = Circle2f {
        center: Point2f::new((cx + mx) as f32, (cy + my) as f32),
        radius: r as f32,
    };
    if !circle.center.x.is_finite() || !circle.center.y.is_finite() || !circle.radius.is_finite() {
        return Err(Error::Degenerate(
            "circle fit: non-finite algebraic solution",
        ));
    }
    Ok(circle)
}

/// One Gauss–Newton step on `Σ wᵢ (‖pᵢ − c‖ − r)²`.
///
/// The Jacobian of the residual w.r.t. `(cx, cy, r)` is `(−ux, −uy, −1)` where
/// `u` is the unit vector from the centre to the point, which makes the normal
/// equations a well-conditioned 3×3 that is worth solving in closed form.
fn gauss_newton_step(pts: &[Point2f], w: &[f64], c: Circle2f) -> Result<Circle2f, Error> {
    let (cx, cy, r) = (c.center.x as f64, c.center.y as f64, c.radius as f64);
    // Symmetric 3x3 normal matrix, upper triangle, plus the rhs.
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [0.0f64; 3];
    let mut sw = 0.0f64;

    for (&p, &wi) in pts.iter().zip(w) {
        if wi <= 0.0 {
            continue;
        }
        let dx = p.x as f64 - cx;
        let dy = p.y as f64 - cy;
        let d = (dx * dx + dy * dy).sqrt();
        if d < 1e-12 {
            // A point at the centre has no defined direction; skip it rather
            // than emit a NaN row.
            continue;
        }
        let (ux, uy) = (dx / d, dy / d);
        let res = d - r;
        // J = (-ux, -uy, -1)
        let j = [-ux, -uy, -1.0];
        for i in 0..3 {
            for k in i..3 {
                a[i][k] += wi * j[i] * j[k];
            }
            b[i] -= wi * j[i] * res;
        }
        sw += wi;
    }
    if sw <= 0.0 {
        return Err(Error::Degenerate(
            "circle fit: all points were down-weighted",
        ));
    }
    // Mirror the upper triangle into the lower one.
    a[1][0] = a[0][1];
    a[2][0] = a[0][2];
    a[2][1] = a[1][2];

    let Some(delta) = solve3(a, b) else {
        // Singular normal equations mean the step is unobservable; keeping the
        // current estimate is the right answer, not an error.
        return Ok(c);
    };

    let next = Circle2f {
        center: Point2f::new((cx + delta[0]) as f32, (cy + delta[1]) as f32),
        radius: (r + delta[2]) as f32,
    };
    if !next.radius.is_finite() || next.radius <= 0.0 || !next.center.x.is_finite() {
        return Ok(c);
    }
    Ok(next)
}

/// Solve a symmetric positive-definite 3×3 by Gaussian elimination with partial
/// pivoting. Returns `None` when singular.
fn solve3(mut a: [[f64; 3]; 3], mut b: [f64; 3]) -> Option<[f64; 3]> {
    for col in 0..3 {
        let mut piv = col;
        for r in col + 1..3 {
            if a[r][col].abs() > a[piv][col].abs() {
                piv = r;
            }
        }
        if a[piv][col].abs() < 1e-14 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for r in col + 1..3 {
            let f = a[r][col] / a[col][col];
            let (done, rest) = a.split_at_mut(r);
            let pivot_row = &done[col];
            for (dst, &src) in rest[0][col..].iter_mut().zip(&pivot_row[col..]) {
                *dst -= f * src;
            }
            b[r] -= f * b[col];
        }
    }
    let mut x = [0.0f64; 3];
    for i in (0..3).rev() {
        let mut s = b[i];
        for k in i + 1..3 {
            s -= a[i][k] * x[k];
        }
        x[i] = s / a[i][i];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// RANSAC over circles through 3 points, returning the consensus indices.
fn ransac_consensus(pts: &[Point2f], rc: &super::RansacConfig) -> Result<Vec<u32>, Error> {
    let n = pts.len();
    let mut rng = Lcg::new(rc.seed);
    let mut best: Vec<u32> = Vec::new();

    for _ in 0..rc.iters {
        let (i, j, k) = (rng.next_mod(n), rng.next_mod(n), rng.next_mod(n));
        if i == j || j == k || i == k {
            continue;
        }
        let Some(c) = circle_through(pts[i], pts[j], pts[k]) else {
            continue;
        };
        let inl: Vec<u32> = pts
            .iter()
            .enumerate()
            .filter(|&(_, &p)| c.signed_distance(p).abs() <= rc.inlier_tol)
            .map(|(idx, _)| idx as u32)
            .collect();
        if inl.len() > best.len() {
            best = inl;
        }
    }

    if best.len() < rc.min_inliers.max(3) {
        return Err(Error::Degenerate("circle fit: RANSAC found no consensus"));
    }
    Ok(best)
}

/// The unique circle through three points, or `None` when they are collinear.
fn circle_through(a: Point2f, b: Point2f, c: Point2f) -> Option<Circle2f> {
    let (ax, ay) = (a.x as f64, a.y as f64);
    let (bx, by) = (b.x as f64, b.y as f64);
    let (cx, cy) = (c.x as f64, c.y as f64);
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-10 {
        return None;
    }
    let a2 = ax * ax + ay * ay;
    let b2 = bx * bx + by * by;
    let c2 = cx * cx + cy * cy;
    let ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d;
    let uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d;
    let r = ((ax - ux).powi(2) + (ay - uy).powi(2)).sqrt();
    if !r.is_finite() {
        return None;
    }
    Some(Circle2f {
        center: Point2f::new(ux as f32, uy as f32),
        radius: r as f32,
    })
}

#[cfg(test)]
mod tests {
    use super::{circle_through, fit_circle, taubin};
    use crate::fit::{FitConfig, RansacConfig, RobustLoss};
    use vm_primitives::{Circle2f, Point2f};

    fn arc(c: &Circle2f, n: usize, span_deg: f32, start_deg: f32) -> Vec<Point2f> {
        (0..n)
            .map(|i| {
                let t = (start_deg + span_deg * i as f32 / (n - 1).max(1) as f32).to_radians();
                c.point_at(t)
            })
            .collect()
    }

    #[test]
    fn recovers_an_exact_full_circle() {
        let truth = Circle2f {
            center: Point2f::new(100.0, 80.0),
            radius: 45.0,
        };
        let pts = arc(&truth, 64, 359.0, 0.0);
        let fit = fit_circle(&pts, &FitConfig::default()).expect("fit");

        assert!(
            (fit.model.radius - 45.0).abs() < 1e-3,
            "r={}",
            fit.model.radius
        );
        assert!((fit.model.center - truth.center).norm() < 1e-3);
        assert!(fit.rms < 1e-3);
        assert_eq!(fit.n_used, 64);
    }

    /// The case that separates a geometric fit from an algebraic one. An
    /// algebraic-only fit pulls the centre towards the chord on a short arc.
    #[test]
    fn short_arcs_stay_unbiased() {
        let truth = Circle2f {
            center: Point2f::new(300.0, 200.0),
            radius: 150.0,
        };
        for span in [30.0f32, 45.0, 90.0, 180.0] {
            let pts = arc(&truth, 40, span, 20.0);
            let fit = fit_circle(&pts, &FitConfig::default()).expect("fit");
            let dr = (fit.model.radius - truth.radius).abs();
            let dc = (fit.model.center - truth.center).norm();
            assert!(dr < 0.05, "span={span}: radius off by {dr}");
            assert!(dc < 0.05, "span={span}: centre off by {dc}");
        }
    }

    #[test]
    fn rms_tracks_injected_radial_noise() {
        let truth = Circle2f {
            center: Point2f::new(50.0, 50.0),
            radius: 30.0,
        };
        // Alternate the radius by exactly +-0.2 px: RMS radial deviation is 0.2.
        let pts: Vec<Point2f> = (0..60)
            .map(|i| {
                let t = i as f32 * core::f32::consts::TAU / 60.0;
                let r = truth.radius + if i % 2 == 0 { 0.2 } else { -0.2 };
                Circle2f {
                    center: truth.center,
                    radius: r,
                }
                .point_at(t)
            })
            .collect();
        let fit = fit_circle(&pts, &FitConfig::default()).expect("fit");
        assert!((fit.rms - 0.2).abs() < 0.01, "rms = {}", fit.rms);
        assert!(
            (fit.max_dev - 0.2).abs() < 0.01,
            "max_dev = {}",
            fit.max_dev
        );
        // The circle itself is unmoved by symmetric deviation.
        assert!((fit.model.radius - 30.0).abs() < 0.01);
    }

    #[test]
    fn ransac_survives_gross_contamination() {
        let truth = Circle2f {
            center: Point2f::new(200.0, 150.0),
            radius: 70.0,
        };
        let mut pts = arc(&truth, 40, 359.0, 0.0);
        // 30% outliers on a second, wrong circle.
        let decoy = Circle2f {
            center: Point2f::new(260.0, 150.0),
            radius: 70.0,
        };
        pts.extend(arc(&decoy, 17, 120.0, 200.0));

        let fit = fit_circle(
            &pts,
            &FitConfig {
                ransac: Some(RansacConfig {
                    iters: 400,
                    inlier_tol: 1.0,
                    min_inliers: 20,
                    seed: 7,
                }),
                loss: RobustLoss::Tukey { c: 2.0 },
                ..FitConfig::default()
            },
        )
        .expect("fit");

        assert!(
            (fit.model.radius - 70.0).abs() < 0.2,
            "r = {}",
            fit.model.radius
        );
        assert!(
            (fit.model.center - truth.center).norm() < 0.2,
            "c = {:?}",
            fit.model.center
        );
        assert!(fit.n_used >= 38, "should keep the true arc: {}", fit.n_used);
        assert!(fit.n_used <= 45, "should drop the decoy: {}", fit.n_used);
    }

    /// Determinism is a contract (invariant 12), not an accident.
    #[test]
    fn ransac_is_reproducible() {
        let truth = Circle2f {
            center: Point2f::new(80.0, 60.0),
            radius: 25.0,
        };
        let mut pts = arc(&truth, 30, 359.0, 0.0);
        pts.push(Point2f::new(200.0, 200.0));
        let cfg = FitConfig {
            ransac: Some(RansacConfig::default()),
            ..FitConfig::default()
        };
        let a = fit_circle(&pts, &cfg).expect("a");
        let b = fit_circle(&pts, &cfg).expect("b");
        assert_eq!(a.model, b.model);
        assert_eq!(a.inliers, b.inliers);
    }

    #[test]
    fn collinear_points_are_degenerate() {
        let pts: Vec<Point2f> = (0..10).map(|i| Point2f::new(i as f32, 3.0)).collect();
        assert!(taubin(&pts).is_err() || fit_circle(&pts, &FitConfig::default()).is_err());
    }

    #[test]
    fn too_few_points_is_an_error() {
        let pts = vec![Point2f::new(0.0, 0.0), Point2f::new(1.0, 1.0)];
        assert!(fit_circle(&pts, &FitConfig::default()).is_err());
    }

    #[test]
    fn three_point_circle_is_exact_and_rejects_collinear() {
        let c = circle_through(
            Point2f::new(0.0, 0.0),
            Point2f::new(10.0, 0.0),
            Point2f::new(5.0, 5.0),
        )
        .expect("valid");
        assert!((c.center.x - 5.0).abs() < 1e-4);
        assert!((c.radius - 5.0).abs() < 1e-4);

        assert!(
            circle_through(
                Point2f::new(0.0, 0.0),
                Point2f::new(1.0, 0.0),
                Point2f::new(2.0, 0.0)
            )
            .is_none()
        );
    }
}
