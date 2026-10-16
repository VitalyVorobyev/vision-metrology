//! One step's rigid increment from the lifted views.
//!
//! The model is `D_v = (R(θ) − I)(a_v − c) + u` for each used view `v`, with `a_v` the
//! view's lift point, `c` the mean lift point (the lever centre) and `u` the translation
//! of the lever centre. Solved by Gauss–Newton from `θ = 0`, whose first iteration is the
//! small-angle weighted least squares, with Huber weights on each view's Mahalanobis
//! residual. The weights start from the cross-view median displacement.

use nalgebra::{Matrix2, Matrix3, Vector2, Vector3};

use super::lift::Lift;

// Invariant 20: every quantity here is f64.

/// A lifted view as the solve sees it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Obs {
    pub lift: Lift,
    /// Huber weight, relative to the view's own information.
    pub weight: f64,
    /// Still in the solve.
    pub active: bool,
    /// Given zero weight by the biweight.
    pub rejected: bool,
}

/// What a step needs from its configuration.
#[derive(Debug, Clone, Copy)]
pub(super) struct SolveParams {
    pub huber_k: f64,
    pub tukey_c: Option<f64>,
    pub iterations: usize,
    pub tolerance_mm: f64,
    /// The rotation a translation-only solve holds, rad.
    pub prior_theta: f64,
    /// Its variance, rad².
    pub prior_var: f64,
    /// The constant-velocity prediction of this step, about the plane origin, with its
    /// covariance. It chooses between the biweight's local minima and never moves one.
    pub predicted: Option<Increment>,
}

/// A plane-frame increment `p' = R(θ) p + t`, with its covariance over `(tx, ty, θ)`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Increment {
    pub t: Vector2<f64>,
    pub theta: f64,
    pub cov: Matrix3<f64>,
}

/// The solved step.
#[derive(Debug, Clone, Copy)]
pub(super) struct Solved {
    pub inc: Increment,
    /// The lever centre, mm.
    pub pivot: Vector2<f64>,
    /// The residual scale the covariance was inflated by (1 without redundancy).
    pub scale: f64,
    /// Rotation was held at the prior.
    pub held: bool,
}

/// Lift points closer than this to their centre carry no rotation, mm.
const MIN_SPREAD_MM: f64 = 1e-6;

/// `dR(θ)/dθ · b`.
#[inline]
pub(super) fn d_rot(theta: f64, b: Vector2<f64>) -> Vector2<f64> {
    let (s, c) = theta.sin_cos();
    Vector2::new(-s * b.x - c * b.y, c * b.x - s * b.y)
}

/// `(R(θ) − I) · b`.
#[inline]
pub(super) fn rot_minus_identity(theta: f64, b: Vector2<f64>) -> Vector2<f64> {
    let (s, c) = theta.sin_cos();
    Vector2::new((c - 1.0) * b.x - s * b.y, s * b.x + (c - 1.0) * b.y)
}

/// The Jacobian of `t = c − R(θ)c + u` with respect to `(u, θ)` (and of the inverse
/// map with `c → −c`): identity, plus `∂t/∂θ = −R'(θ)c`.
pub(super) fn pivot_jacobian(theta: f64, c: Vector2<f64>) -> Matrix3<f64> {
    let dt = -d_rot(theta, c);
    Matrix3::new(1.0, 0.0, dt.x, 0.0, 1.0, dt.y, 0.0, 0.0, 1.0)
}

#[inline]
fn mahalanobis(r: Vector2<f64>, info: &Matrix2<f64>) -> f64 {
    (r.dot(&(info * r))).max(0.0).sqrt()
}

#[inline]
fn huber(m: f64, k: f64) -> f64 {
    if m <= k { 1.0 } else { k / m }
}

/// The median of `v`, which it sorts.
fn median(v: &mut [f64]) -> f64 {
    v.sort_unstable_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Solve the step over the active observations, or `None` when none is active.
/// `scratch` holds at least `obs.len()` values.
pub(super) fn solve(obs: &mut [Obs], p: &SolveParams, scratch: &mut [f64]) -> Option<Solved> {
    let n = obs.iter().filter(|o| o.active).count();
    if n == 0 {
        return None;
    }
    let pivot = obs
        .iter()
        .filter(|o| o.active)
        .fold(Vector2::zeros(), |s, o| s + o.lift.lever)
        / n as f64;
    let spread = obs
        .iter()
        .filter(|o| o.active)
        .map(|o| (o.lift.lever - pivot).norm())
        .fold(0.0, f64::max);
    let (u, theta, cov, scale, held) = if n == 1 || spread < MIN_SPREAD_MM {
        let (u, cov) = held_rotation(obs, pivot, p);
        (u, p.prior_theta, cov, 1.0, true)
    } else {
        match robust(obs, pivot, spread, p, scratch) {
            Some((u, theta, cov, s)) => (u, theta, cov, s, false),
            None => {
                let (u, cov) = held_rotation(obs, pivot, p);
                (u, p.prior_theta, cov, 1.0, true)
            }
        }
    };
    // From the lever centre to the plane origin: t = c − R c + u.
    let t = u - rot_minus_identity(theta, pivot);
    let j = pivot_jacobian(theta, pivot);
    Some(Solved {
        inc: Increment {
            t,
            theta,
            cov: j * cov * j.transpose(),
        },
        pivot,
        scale,
        held,
    })
}

/// Translation only, with rotation at the prior: the information-weighted mean of what
/// each view leaves after the held rotation. Covariance over `(u, θ)`.
fn held_rotation(
    obs: &mut [Obs],
    pivot: Vector2<f64>,
    p: &SolveParams,
) -> (Vector2<f64>, Matrix3<f64>) {
    let mut info = Matrix2::zeros();
    let mut rhs = Vector2::zeros();
    for o in obs.iter_mut().filter(|o| o.active) {
        o.weight = 1.0;
        let left = o.lift.d - rot_minus_identity(p.prior_theta, o.lift.lever - pivot);
        info += o.lift.info;
        rhs += o.lift.info * left;
    }
    // `lift` returns positive definite information, so the sum is invertible.
    let cov_u = info.try_inverse().unwrap_or_else(Matrix2::zeros);
    let mut cov = Matrix3::zeros();
    cov.fixed_view_mut::<2, 2>(0, 0).copy_from(&cov_u);
    cov[(2, 2)] = p.prior_var;
    (cov_u * rhs, cov)
}

/// A robust stage's estimate: `(u, θ)`, the covariance over them, and the residual scale.
type Stage = (Vector2<f64>, f64, Matrix3<f64>, f64);

/// The robust solve over at least two spread views: Huber from the median start, then,
/// when configured, the Tukey biweight from the Huber solution. `None` when the normal
/// equations are singular.
fn robust(
    obs: &mut [Obs],
    pivot: Vector2<f64>,
    spread: f64,
    p: &SolveParams,
    scratch: &mut [f64],
) -> Option<Stage> {
    let huber_stage = irls(obs, pivot, spread, p, scratch)?;
    let Some(c) = p.tukey_c else {
        return Some(huber_stage);
    };
    // The biweight from the Huber solution and from the prediction. With three views a
    // gross one can be traded for a rotation that makes a good view look gross, at the
    // same cost; only the motion tells the two apart.
    let (u, theta, _, _) = huber_stage;
    let starts = [
        Some((u, theta)),
        p.predicted
            .map(|q| (q.t + rot_minus_identity(q.theta, pivot), q.theta)),
    ];
    let mut best: Option<(usize, f64)> = None;
    for (i, start) in starts.iter().enumerate() {
        let Some((u0, t0)) = *start else { continue };
        let Some(found) = tukey(obs, pivot, u0, t0, c, p, scratch) else {
            continue;
        };
        let score = tukey_cost(obs, pivot, &found, c) + prior_cost(&found, pivot, p);
        if best.is_none_or(|(_, b)| score < b) {
            best = Some((i, score));
        }
    }
    if let Some((i, _)) = best {
        // Re-run the winner so the weights are its own (deterministic).
        let (u0, t0) = starts[i].expect("a scored start exists");
        let found = tukey(obs, pivot, u0, t0, c, p, scratch).expect("it succeeded before");
        for o in obs.iter_mut().filter(|o| o.active) {
            o.rejected = o.weight == 0.0;
        }
        return Some(found);
    }
    // Too little support survived the biweight: keep the Huber solution and its weights.
    for o in obs.iter_mut().filter(|o| o.active) {
        let r = residual(&o.lift, pivot, u, theta);
        o.weight = huber(mahalanobis(r, &o.lift.info), p.huber_k);
    }
    Some(huber_stage)
}

/// The biweight's cost at a solution, in σ²: `c²/6 · (1 − (1 − (m/c)²)³)` per view.
fn tukey_cost(obs: &[Obs], pivot: Vector2<f64>, at: &Stage, c: f64) -> f64 {
    let (u, theta, _, _) = *at;
    obs.iter()
        .filter(|o| o.active)
        .map(|o| {
            let m = mahalanobis(residual(&o.lift, pivot, u, theta), &o.lift.info);
            let q = (1.0 - (m / c).powi(2)).max(0.0);
            c * c / 6.0 * (1.0 - q * q * q)
        })
        .sum()
}

/// Half the squared Mahalanobis distance of a solution from the prediction; 0 without
/// one.
fn prior_cost(at: &Stage, pivot: Vector2<f64>, p: &SolveParams) -> f64 {
    let Some(q) = p.predicted else { return 0.0 };
    let (u, theta, _, _) = *at;
    let t = u - rot_minus_identity(theta, pivot);
    let d = Vector3::new(t.x - q.t.x, t.y - q.t.y, theta - q.theta);
    q.cov
        .try_inverse()
        .map_or(0.0, |inv| 0.5 * d.dot(&(inv * d)))
}

/// Tukey's biweight on a Mahalanobis residual `m` at radius `c`.
#[inline]
fn biweight(m: f64, c: f64) -> f64 {
    if m >= c {
        0.0
    } else {
        let q = 1.0 - (m / c) * (m / c);
        q * q
    }
}

/// The residual scale: the median Mahalanobis residual of the views `keep` selects, and
/// never below 1. Views whose covariance is honest give about 1; optimistic covariances
/// give their real spread. It inflates the reported covariance only: with a handful of
/// views and one of them gross, a scale estimated alongside the weights is pulled up by
/// the gross view and the weights lose it.
fn scale(
    obs: &[Obs],
    pivot: Vector2<f64>,
    u: Vector2<f64>,
    theta: f64,
    keep: impl Fn(&Obs) -> bool,
    scratch: &mut [f64],
) -> f64 {
    let mut n = 0;
    for o in obs.iter().filter(|o| o.active && keep(o)) {
        scratch[n] = mahalanobis(residual(&o.lift, pivot, u, theta), &o.lift.info);
        n += 1;
    }
    if n == 0 {
        return 1.0;
    }
    median(&mut scratch[..n]).max(1.0)
}

/// Graduated Tukey reweighting from `(u, θ)`: the radius starts at twice the largest
/// residual and shrinks geometrically to `c`,
/// so the start keeps every view and the non-convex loss only bites once the estimate is
/// close. `None` when fewer than two spread views keep weight, or the normal equations
/// are singular.
fn tukey(
    obs: &mut [Obs],
    pivot: Vector2<f64>,
    mut u: Vector2<f64>,
    mut theta: f64,
    c: f64,
    p: &SolveParams,
    scratch: &mut [f64],
) -> Option<Stage> {
    let largest = obs
        .iter()
        .filter(|o| o.active)
        .map(|o| mahalanobis(residual(&o.lift, pivot, u, theta), &o.lift.info))
        .fold(0.0, f64::max);
    let mut radius = largest.max(c);
    let mut settled = 0;
    // The annealing steps, then up to `iterations` at the final radius.
    while settled < p.iterations {
        let normal = weighted_normal(obs, pivot, u, theta, radius)?;
        let mut g = Vector3::zeros();
        for o in obs.iter().filter(|o| o.active && o.weight > 0.0) {
            g += o.weight * normal_terms(&o.lift, pivot, u, theta).1;
        }
        let step = normal.cholesky()?.solve(&g);
        u += Vector2::new(step.x, step.y);
        theta += step.z;
        if radius > c {
            radius = (radius * 0.6).max(c);
            continue;
        }
        settled += 1;
        let spread = obs
            .iter()
            .filter(|o| o.active)
            .map(|o| (o.lift.lever - pivot).norm())
            .fold(0.0, f64::max);
        if step.x.hypot(step.y) + step.z.abs() * spread < p.tolerance_mm {
            break;
        }
    }
    let normal = weighted_normal(obs, pivot, u, theta, c)?;
    let kept = scale(obs, pivot, u, theta, |o| o.weight > 0.0, scratch);
    Some((u, theta, normal.cholesky()?.inverse() * kept * kept, kept))
}

/// Set the biweights at `radius` and return the weighted normal matrix, or `None` when
/// fewer than two spread views keep weight.
fn weighted_normal(
    obs: &mut [Obs],
    pivot: Vector2<f64>,
    u: Vector2<f64>,
    theta: f64,
    radius: f64,
) -> Option<Matrix3<f64>> {
    let mut normal = Matrix3::zeros();
    let (mut kept, mut centre) = (0usize, Vector2::zeros());
    for o in obs.iter_mut().filter(|o| o.active) {
        let r = residual(&o.lift, pivot, u, theta);
        o.weight = biweight(mahalanobis(r, &o.lift.info), radius);
        if o.weight > 0.0 {
            normal += o.weight * normal_terms(&o.lift, pivot, u, theta).0;
            kept += 1;
            centre += o.lift.lever;
        }
    }
    if kept < 2 {
        return None;
    }
    let centre = centre / kept as f64;
    let spread = obs
        .iter()
        .filter(|o| o.active && o.weight > 0.0)
        .map(|o| (o.lift.lever - centre).norm())
        .fold(0.0, f64::max);
    (spread >= MIN_SPREAD_MM).then_some(normal)
}

/// `D − (R − I)(a − c) − u`.
#[inline]
fn residual(l: &Lift, pivot: Vector2<f64>, u: Vector2<f64>, theta: f64) -> Vector2<f64> {
    l.d - rot_minus_identity(theta, l.lever - pivot) - u
}

/// Iteratively reweighted Gauss–Newton from the median start, with Huber weights.
fn irls(
    obs: &mut [Obs],
    pivot: Vector2<f64>,
    spread: f64,
    p: &SolveParams,
    scratch: &mut [f64],
) -> Option<Stage> {
    // Robust start: weights from each view's distance to the cross-view median.
    let mut med = Vector2::zeros();
    for axis in 0..2 {
        let mut n = 0;
        for o in obs.iter().filter(|o| o.active) {
            scratch[n] = o.lift.d[axis];
            n += 1;
        }
        med[axis] = median(&mut scratch[..n]);
    }
    for o in obs.iter_mut().filter(|o| o.active) {
        o.weight = huber(mahalanobis(o.lift.d - med, &o.lift.info), p.huber_k);
    }

    let mut u = Vector2::zeros();
    let mut theta = 0.0;
    for _ in 0..p.iterations {
        let mut normal = Matrix3::zeros();
        let mut g = Vector3::zeros();
        for o in obs.iter().filter(|o| o.active) {
            let (n_v, g_v) = normal_terms(&o.lift, pivot, u, theta);
            normal += o.weight * n_v;
            g += o.weight * g_v;
        }
        let step = normal.cholesky()?.solve(&g);
        u += Vector2::new(step.x, step.y);
        theta += step.z;
        for o in obs.iter_mut().filter(|o| o.active) {
            let r = residual(&o.lift, pivot, u, theta);
            o.weight = huber(mahalanobis(r, &o.lift.info), p.huber_k);
        }
        if step.x.hypot(step.y) + step.z.abs() * spread < p.tolerance_mm {
            break;
        }
    }
    // The covariance at the final weights and estimate, inflated by the scale.
    let mut normal = Matrix3::zeros();
    for o in obs.iter().filter(|o| o.active) {
        normal += o.weight * normal_terms(&o.lift, pivot, u, theta).0;
    }
    let s = scale(obs, pivot, u, theta, |_| true, scratch);
    let cov = normal.cholesky()?.inverse() * s * s;
    Some((u, theta, cov, s))
}

/// `Jᵀ Λ J` and `Jᵀ Λ r` of one view at `(u, θ)`, with `J = [I | R'(θ)(a − c)]`.
fn normal_terms(
    l: &Lift,
    pivot: Vector2<f64>,
    u: Vector2<f64>,
    theta: f64,
) -> (Matrix3<f64>, Vector3<f64>) {
    let b = l.lever - pivot;
    let j = d_rot(theta, b);
    let r = residual(l, pivot, u, theta);
    let lj = l.info * j;
    let lr = l.info * r;
    let mut n = Matrix3::zeros();
    n.fixed_view_mut::<2, 2>(0, 0).copy_from(&l.info);
    n[(0, 2)] = lj.x;
    n[(1, 2)] = lj.y;
    n[(2, 0)] = lj.x;
    n[(2, 1)] = lj.y;
    n[(2, 2)] = j.dot(&lj);
    (n, Vector3::new(lr.x, lr.y, j.dot(&lr)))
}

/// The residual of a lifted view against a solved increment about the plane origin:
/// `D − ((R − I) a + t)`, mm.
pub(super) fn increment_residual(l: &Lift, inc: &Increment) -> f64 {
    (l.d - rot_minus_identity(inc.theta, l.lever) - inc.t).norm()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(lever: (f64, f64), d: Vector2<f64>, sigma: f64) -> Obs {
        Obs {
            lift: Lift {
                lever: Vector2::new(lever.0, lever.1),
                d,
                info: Matrix2::identity() / (sigma * sigma),
            },
            weight: 1.0,
            active: true,
            rejected: false,
        }
    }

    fn params() -> SolveParams {
        SolveParams {
            huber_k: 2.0,
            tukey_c: None,
            iterations: 20,
            tolerance_mm: 1e-12,
            prior_theta: 0.0,
            prior_var: 1e-6,
            predicted: None,
        }
    }

    /// Displacements generated from an exact rigid increment are recovered to rounding,
    /// for a rotation large enough that the small-angle form alone would be visibly off.
    #[test]
    fn exact_rigid_increment_is_recovered() {
        let (theta, t) = (0.05f64, Vector2::new(0.7, -0.3));
        let levers = [(30.0, 0.0), (-15.0, 26.0), (-15.0, -26.0)];
        let mut o: Vec<Obs> = levers
            .iter()
            .map(|&(x, y)| {
                let a = Vector2::new(x, y);
                obs((x, y), rot_minus_identity(theta, a) + t, 0.01)
            })
            .collect();
        let s = solve(&mut o, &params(), &mut [0.0; 3]).unwrap();
        assert!(!s.held);
        assert!((s.inc.theta - theta).abs() < 1e-12, "θ {}", s.inc.theta);
        assert!((s.inc.t - t).norm() < 1e-10, "t {:?}", s.inc.t);
        assert!(o.iter().all(|o| (o.weight - 1.0).abs() < 1e-12));
    }

    /// One view 50σ off: Huber leaves it a small weight, and the biweight none.
    #[test]
    fn a_gross_view_is_bounded_by_huber_and_removed_by_the_biweight() {
        let t = Vector2::new(0.5, 0.2);
        let mut o = vec![
            obs((30.0, 0.0), t, 0.01),
            obs((-15.0, 26.0), t + Vector2::new(0.5, 0.0), 0.01),
            obs((-15.0, -26.0), t, 0.01),
        ];
        let s = solve(&mut o, &params(), &mut [0.0; 3]).unwrap();
        assert!(o[1].weight < 0.1, "gross weight {}", o[1].weight);
        // Huber bounds the gross view's pull, it does not remove it: with rotation free
        // to absorb part of it, a few σ remain (measured 3.4σ here).
        assert!((s.inc.t - t).norm() < 0.05, "Huber pull {:?}", s.inc.t - t);

        let p = SolveParams {
            tukey_c: Some(6.0),
            ..params()
        };
        let s = solve(&mut o, &p, &mut [0.0; 3]).unwrap();
        assert!(o[1].rejected && o[1].weight == 0.0);
        assert!(
            o[0].weight > 0.99 && o[2].weight > 0.99,
            "{:?}",
            [o[0].weight, o[2].weight]
        );
        assert!((s.inc.t - t).norm() < 1e-9, "biweight {:?}", s.inc.t - t);
    }

    #[test]
    fn one_view_holds_the_prior_rotation() {
        let p = SolveParams {
            prior_theta: 0.01,
            ..params()
        };
        let a = Vector2::new(30.0, 0.0);
        let d = rot_minus_identity(0.01, a) + Vector2::new(0.4, 0.1);
        let mut o = vec![obs((30.0, 0.0), d, 0.01)];
        let s = solve(&mut o, &p, &mut [0.0; 1]).unwrap();
        assert!(s.held);
        assert_eq!(s.inc.theta, 0.01);
        assert!((s.inc.t - Vector2::new(0.4, 0.1)).norm() < 1e-12);
        assert!((s.inc.cov[(2, 2)] - 1e-6).abs() < 1e-18);
        assert!(solve(&mut [], &p, &mut []).is_none());
    }
}
