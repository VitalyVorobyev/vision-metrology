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
fn observations(
    a: &Plane,
    b: &Plane,
    pose: Pose,
    region: [usize; 4],
    mut f: impl FnMut(f64, f64, f64, f64, f64, f64),
) {
    for y in region[2]..region[3] {
        for x in region[0]..region[1] {
            let i = y * a.w + x;
            if !a.valid[i] {
                continue;
            }
            let q = pose.map(x as f64, y as f64);
            if let Some((v, gx, gy)) = b.sample(q[0], q[1]) {
                f(
                    a.data[i] as f64,
                    v,
                    gx,
                    gy,
                    (x as f64 - pose.center[0]) / pose.radius,
                    (y as f64 - pose.center[1]) / pose.radius,
                );
            }
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn system(
    a: &Plane,
    b: &Plane,
    pose: Pose,
    region: [usize; 4],
    tuning: &TranslationTuning,
    residuals: &mut Vec<f64>,
) -> (Matrix, Vector, f64, f64, usize) {
    residuals.clear();
    observations(a, b, pose, region, |u, v, _, _, _, _| {
        residuals.push((pose.p[4] * v + pose.p[5] - u).abs())
    });
    let n = residuals.len();
    let scale = if n == 0 {
        1.0
    } else {
        let (_, median, _) = residuals.select_nth_unstable_by(n / 2, f64::total_cmp);
        (*median * 1.4826).max(1e-4)
    };
    let cutoff = tuning.huber as f64 * scale;
    let (mut h, mut rhs, mut sum, mut max) = (Matrix::zeros(), Vector::zeros(), 0.0, 0.0f64);
    observations(a, b, pose, region, |u, v, gx, gy, x, y| {
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
    });
    (h, rhs, sum, max, n)
}
pub(super) fn refine(
    a: &Plane,
    b: &Plane,
    mut pose: Pose,
    region: [usize; 4],
    min_support: usize,
    tuning: &TranslationTuning,
    residuals: &mut Vec<f64>,
) -> Result<Fit, TranslationStatus> {
    let mut converged = false;
    for _ in 0..tuning.iterations.get() {
        let (h, rhs, _, _, n) = system(a, b, pose, region, tuning, residuals);
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
        if step.fixed_rows::<4>(0).norm() < tuning.tolerance as f64
            && step[4].abs() < tuning.tolerance as f64
            && step[5].abs() < tuning.tolerance as f64
        {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(TranslationStatus::RefinementFailed);
    }
    let (h, _, sum, max, n) = system(a, b, pose, region, tuning, residuals);
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
