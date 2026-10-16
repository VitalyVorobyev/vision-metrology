use crate::corr::plane::Plane;
use crate::corr::{TranslationStatus, TranslationTuning};
use nalgebra::{SMatrix, SVector};

type Matrix = SMatrix<f64, 6, 6>;
type Vector = SVector<f64, 6>;

#[derive(Clone, Copy)]
pub(super) struct Pose {
    pub center: [f64; 2],
    pub radius: f64,
    // Centre displacement, linear coefficients times radius, gain and offset.
    pub p: Vector,
}
impl Pose {
    pub fn map(&self, x: f64, y: f64) -> [f64; 2] {
        let (u, v) = (
            (x - self.center[0]) / self.radius,
            (y - self.center[1]) / self.radius,
        );
        [
            self.center[0] + self.p[0] + self.p[2] * u - self.p[3] * v,
            self.center[1] + self.p[1] + self.p[3] * u + self.p[2] * v,
        ]
    }
    /// The same map, parameterized about `center` and normalized by `radius`.
    pub fn recentred(&self, center: [f64; 2], radius: f64) -> Self {
        let q = self.map(center[0], center[1]);
        let (a, b) = (self.p[2] / self.radius, self.p[3] / self.radius);
        Self {
            center,
            radius,
            p: Vector::new(
                q[0] - center[0],
                q[1] - center[1],
                a * radius,
                b * radius,
                self.p[4],
                self.p[5],
            ),
        }
    }
    pub fn angle(&self) -> f64 {
        self.p[3].atan2(self.p[2])
    }
    pub fn scale(&self) -> f64 {
        self.p[2].hypot(self.p[3]) / self.radius
    }
}
#[derive(Clone, Copy)]
pub(super) struct Fit {
    pub pose: Pose,
    pub support: usize,
    pub rms: f64,
    pub max: f64,
    pub covariance: [[f32; 4]; 4],
}
/// Reusable refinement scratch: residual magnitudes and the frozen observation set.
#[derive(Default)]
pub(super) struct Workspace {
    pub residuals: Vec<f64>,
    pub observations: Vec<u32>,
}
/// Valid previous pixels whose current-image taps stay valid while the map moves by less
/// than 1 px from `pose` (see `Plane::window_valid`). The set is then fixed.
fn freeze(a: &Plane, b: &Plane, pose: &Pose, region: [usize; 4], out: &mut Vec<u32>) {
    out.clear();
    for y in region[2]..region[3] {
        for x in region[0]..region[1] {
            let i = y * a.w + x;
            let q = pose.map(x as f64, y as f64);
            if a.valid[i] && b.window_valid(q[0], q[1]) {
                out.push(i as u32);
            }
        }
    }
}
/// Largest per-axis displacement between two maps over a region; for affine maps the
/// extremes are at the corners.
fn departure(p: &Pose, q: &Pose, region: [usize; 4]) -> f64 {
    let (x0, x1) = (region[0] as f64, region[1].saturating_sub(1) as f64);
    let (y0, y1) = (region[2] as f64, region[3].saturating_sub(1) as f64);
    [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        .into_iter()
        .map(|(x, y)| {
            let (a, b) = (p.map(x, y), q.map(x, y));
            (a[0] - b[0]).abs().max((a[1] - b[1]).abs())
        })
        .fold(0.0, f64::max)
}
fn observations(
    a: &Plane,
    b: &Plane,
    pose: Pose,
    frozen: &[u32],
    mut f: impl FnMut(f64, f64, f64, f64, f64, f64),
) {
    for &i in frozen {
        let i = i as usize;
        let (x, y) = ((i % a.w) as f64, (i / a.w) as f64);
        let q = pose.map(x, y);
        if let Some((v, gx, gy)) = b.sample(q[0], q[1]) {
            f(
                a.data[i] as f64,
                v,
                gx,
                gy,
                (x - pose.center[0]) / pose.radius,
                (y - pose.center[1]) / pose.radius,
            );
        }
    }
}
/// 1.4826 × the median absolute photometric residual.
fn robust_scale(a: &Plane, b: &Plane, pose: Pose, scratch: &mut Workspace) -> f64 {
    let Workspace {
        residuals,
        observations: frozen,
    } = scratch;
    residuals.clear();
    observations(a, b, pose, frozen, |u, v, _, _, _, _| {
        residuals.push((pose.p[4] * v + pose.p[5] - u).abs())
    });
    let n = residuals.len();
    if n == 0 {
        return 1.0;
    }
    let (_, median, _) = residuals.select_nth_unstable_by(n / 2, f64::total_cmp);
    (*median * 1.4826).max(1e-4)
}
fn system(
    a: &Plane,
    b: &Plane,
    pose: Pose,
    cutoff: f64,
    frozen: &[u32],
) -> (Matrix, Vector, f64, f64, usize) {
    let (mut h, mut rhs, mut sum, mut max, mut n) =
        (Matrix::zeros(), Vector::zeros(), 0.0, 0.0f64, 0usize);
    observations(a, b, pose, frozen, |u, v, gx, gy, x, y| {
        let r = pose.p[4] * v + pose.p[5] - u;
        let w = (cutoff / r.abs().max(cutoff)).min(1.0);
        let g = pose.p[4];
        let j = Vector::new(
            g * gx,
            g * gy,
            g * (gx * x + gy * y),
            g * (-gx * y + gy * x),
            v,
            1.0,
        );
        h += j * j.transpose() * w;
        rhs += j * (w * r);
        sum += r * r;
        max = max.max(r.abs());
        n += 1;
    });
    (h, rhs, sum, max, n)
}
/// Robust six-parameter Gauss–Newton over a frozen observation set, re-frozen only when
/// the map leaves the ±1 px window of the pose it was frozen at. The Huber scale is
/// estimated at the seed and after the first step, then fixed; convergence tests the
/// motion step alone.
pub(super) fn refine(
    a: &Plane,
    b: &Plane,
    mut pose: Pose,
    region: [usize; 4],
    min_support: usize,
    tuning: &TranslationTuning,
    scratch: &mut Workspace,
) -> Result<Fit, TranslationStatus> {
    let mut frozen_at = pose;
    freeze(a, b, &frozen_at, region, &mut scratch.observations);
    let mut cutoff = 0.0;
    let mut converged = false;
    for iteration in 0..tuning.iterations.get() {
        if departure(&pose, &frozen_at, region) >= 1.0 {
            frozen_at = pose;
            freeze(a, b, &frozen_at, region, &mut scratch.observations);
        }
        if scratch.observations.len() < min_support {
            return Err(TranslationStatus::InsufficientSupport);
        }
        if iteration < 2 {
            cutoff = tuning.huber as f64 * robust_scale(a, b, pose, scratch);
        }
        let (h, rhs, _, _, n) = system(a, b, pose, cutoff, &scratch.observations);
        if n < min_support {
            return Err(TranslationStatus::InsufficientSupport);
        }
        let chol = h.cholesky().ok_or(TranslationStatus::LowTexture)?;
        let inv = chol.inverse();
        // Inverse Schur complement of the motion block: nuisance photometry
        // cannot create false observability. Fixed-size eigen solve allocates nothing.
        let worst = inv
            .fixed_view::<4, 4>(0, 0)
            .into_owned()
            .symmetric_eigen()
            .eigenvalues
            .max();
        if !worst.is_finite()
            || worst <= 0.0
            || 1.0 / (worst * n as f64) < tuning.min_texture as f64
        {
            return Err(TranslationStatus::LowTexture);
        }
        let mut step = chol.solve(&rhs);
        let motion = step.fixed_rows::<4>(0).norm();
        if motion > 1.0 {
            step /= motion;
        }
        pose.p -= step;
        if pose.p.iter().any(|v| !v.is_finite()) || pose.p[4] <= 0.0 || pose.scale() <= 0.0 {
            return Err(TranslationStatus::RefinementFailed);
        }
        if step.fixed_rows::<4>(0).norm() < tuning.tolerance as f64 {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(TranslationStatus::RefinementFailed);
    }
    if departure(&pose, &frozen_at, region) >= 1.0 {
        freeze(a, b, &pose, region, &mut scratch.observations);
    }
    let (h, _, sum, max, n) = system(a, b, pose, cutoff, &scratch.observations);
    if n < min_support {
        return Err(TranslationStatus::InsufficientSupport);
    }
    let inv = h.cholesky().ok_or(TranslationStatus::LowTexture)?.inverse();
    let (a, b) = (pose.p[2], pose.p[3]);
    let den = a * a + b * b;
    let j = SMatrix::<f64, 4, 4>::new(
        1.,
        0.,
        0.,
        0.,
        0.,
        1.,
        0.,
        0.,
        0.,
        0.,
        -b / den,
        a / den,
        0.,
        0.,
        a / den,
        b / den,
    );
    let covariance = j
        * inv.fixed_view::<4, 4>(0, 0)
        * j.transpose()
        * (sum / n.saturating_sub(6).max(1) as f64);
    let covariance = std::array::from_fn(|i| std::array::from_fn(|j| covariance[(i, j)] as f32));
    if covariance.iter().flatten().any(|v| !v.is_finite()) {
        return Err(TranslationStatus::RefinementFailed);
    }
    Ok(Fit {
        pose,
        support: n,
        rms: (sum / n as f64).sqrt(),
        max,
        covariance,
    })
}
