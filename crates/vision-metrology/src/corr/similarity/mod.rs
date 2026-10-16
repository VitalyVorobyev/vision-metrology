//! Local native masked similarity registration. No angle bank or global scale search.
mod config;
mod refine;
mod result;
pub use config::SimilarityConfig;
pub use result::Similarity;

use super::{RegionCheck, TranslationEstimator, TranslationStatus, plane::Plane};
use nalgebra::{Matrix4, SVector, Vector4};
use refine::Pose;
use vm_primitives::{Error, ImageView, Pixel, Similarity2f, Vec2f, similarity_parts, wrap_angle};

/// Reusable local estimator of rotation, uniform scale and translation.
///
/// **Experimental** (feature `corr-experimental`): outside the semver promise until its
/// regional consistency check is recalibrated for real data. Its API may change or be
/// removed in any release.
///
/// Maps previous pixel centres into current coordinates. Mask-safe prediction warping
/// initializes bounded translation correlation, then native-image robust refinement
/// estimates four motion and two photometric parameters. This is not global pose search.
/// Prepared native calls at unchanged dimensions/configuration allocate nothing.
#[derive(Default)]
pub struct SimilarityEstimator {
    previous: Plane,
    current: Plane,
    warped: Vec<f32>,
    previous_valid: Vec<u8>,
    warped_valid: Vec<u8>,
    scratch: refine::Workspace,
    initializer: TranslationEstimator,
}
impl SimilarityEstimator {
    /// Construct empty scratch; the first valid call prepares capacity.
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate a map `q_current = s R p_previous + t` near a prediction (identity by default).
    /// All interpolation and derivative taps must be valid; masks belong to their image frames.
    /// Rotation/log-scale bounds qualify local refinement, not an exhaustive search.
    ///
    /// # Errors
    /// Invalid configuration, nonpositive/nonfinite prediction scale, invalid mask/image
    /// dimensions, nonfinite prediction, or nonfinite unmasked pixels.
    pub fn estimate<P: Pixel>(
        &mut self,
        previous: &ImageView<'_, P>,
        current: &ImageView<'_, P>,
        previous_mask: Option<&ImageView<'_, u8>>,
        current_mask: Option<&ImageView<'_, u8>>,
        prediction: Option<Similarity2f>,
        cfg: &SimilarityConfig,
    ) -> Result<Similarity, Error> {
        cfg.validate()?;
        for (w, h, m) in [
            (previous.width(), previous.height(), previous_mask),
            (current.width(), current.height(), current_mask),
        ] {
            if w < 4
                || h < 4
                || w > i32::MAX as usize / 2
                || h > i32::MAX as usize / 2
                || m.is_some_and(|m| m.width() != w || m.height() != h)
            {
                return Err(Error::InvalidConfig(
                    "invalid similarity image or mask dimensions",
                ));
            }
        }
        let prediction = prediction.unwrap_or_else(Similarity2f::identity);
        let (t, angle, scale) = similarity_parts(&prediction);
        if !t.iter().all(|v| v.is_finite() && v.abs() <= 1e6)
            || !angle.is_finite()
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(Error::InvalidConfig("invalid similarity prediction"));
        }
        let (mean_a, scale_a, na) = self.previous.load(previous, previous_mask)?;
        let (mean_b, scale_b, nb) = self.current.load(current, current_mask)?;
        let (a, b) = (&self.previous, &self.current);
        let n = a.w * a.h;
        self.warped.resize(n, 0.);
        self.previous_valid.resize(n, 0);
        self.warped_valid.resize(n, 0);
        let scratch = &mut self.scratch;
        scratch
            .residuals
            .reserve(n - scratch.residuals.len().min(n));
        scratch
            .observations
            .reserve(n - scratch.observations.len().min(n));
        let center = [(a.w - 1) as f64 * 0.5, (a.h - 1) as f64 * 0.5];
        let radius = center[0].hypot(center[1]).max(1.0);
        let (sin, cos) = (angle as f64).sin_cos();
        let (u, v) = (scale as f64 * cos, scale as f64 * sin);
        let base = Pose {
            center,
            radius,
            p: SVector::<f64, 6>::new(
                u * center[0] - v * center[1] + t.x as f64 - center[0],
                v * center[0] + u * center[1] + t.y as f64 - center[1],
                u * radius,
                v * radius,
                1.,
                0.,
            ),
        };
        for y in 0..a.h {
            for x in 0..a.w {
                let i = y * a.w + x;
                self.previous_valid[i] = u8::from(a.valid[i]);
                let q = base.map(x as f64, y as f64);
                let sample = b.sample(q[0], q[1]);
                self.warped[i] = sample.map_or(0., |s| s.0 as f32);
                self.warped_valid[i] = u8::from(sample.is_some());
            }
        }
        let av = ImageView::from_slice(a.w, a.h, a.w, &a.data)?;
        let bv = ImageView::from_slice(a.w, a.h, a.w, &self.warped)?;
        let am = ImageView::from_slice(a.w, a.h, a.w, &self.previous_valid)?;
        let bm = ImageView::from_slice(a.w, a.h, a.w, &self.warped_valid)?;
        // Search only: the integer peak seeds the similarity solve, so no translation
        // refinement runs (the similarity refinement starts from this peak).
        let (initial, _) =
            self.initializer
                .search(&av, &bv, Some(&am), Some(&bm), None, &cfg.translation)?;
        let mut out = Similarity::empty(initial.status);
        out.score = initial.score;
        out.margin = initial.margin;
        out.support = initial.support;
        if na.min(nb) < cfg.translation.min_support.get() {
            out.status = TranslationStatus::InsufficientSupport;
            return Ok(out);
        }
        if scale_a <= 0.0 || scale_b <= 0.0 {
            out.status = TranslationStatus::LowTexture;
            return Ok(out);
        }
        // A peak that passed the search gates initializes the similarity solve; a
        // translation-only refinement is never asked to converge under rotation/scale.
        if initial.status != TranslationStatus::Accepted {
            return Ok(out);
        }
        let Some(shift) = initial.shift else {
            return Ok(out);
        };
        let mut seed = base;
        seed.p[0] += u * shift.x as f64 - v * shift.y as f64;
        seed.p[1] += v * shift.x as f64 + u * shift.y as f64;
        let fit = match refine::refine(
            a,
            b,
            seed,
            [0, a.w, 0, a.h],
            cfg.translation.min_support.get(),
            &cfg.translation.tuning,
            &mut self.scratch,
        ) {
            Ok(f) => f,
            Err(s) => {
                out.status = s;
                return Ok(out);
            }
        };
        let pose = fit.pose;
        let acoef = pose.p[2] / radius;
        let bcoef = pose.p[3] / radius;
        out.transform = Some(Similarity2f::new(
            Vec2f::new(
                (center[0] + pose.p[0] - acoef * center[0] + bcoef * center[1]) as f32,
                (center[1] + pose.p[1] - bcoef * center[0] - acoef * center[1]) as f32,
            ),
            pose.angle() as f32,
            pose.scale() as f32,
        ));
        out.support = fit.support;
        out.gain = Some((pose.p[4] * scale_a / scale_b) as f32);
        out.offset =
            Some((mean_a + scale_a * pose.p[5] - pose.p[4] * scale_a / scale_b * mean_b) as f32);
        out.residual_rms = Some((fit.rms * scale_a) as f32);
        out.residual_max = Some((fit.max * scale_a) as f32);
        out.covariance = Some(fit.covariance);
        if [out.gain, out.offset, out.residual_rms, out.residual_max]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite())
        {
            out.status = TranslationStatus::RefinementFailed;
            out.gain = None;
            out.offset = None;
            out.residual_rms = None;
            out.residual_max = None;
            out.covariance = None;
            return Ok(out);
        }
        out.status = TranslationStatus::Accepted;
        if (pose.p[0] - base.p[0]).abs() >= cfg.translation.search_radius.get() as f64
            || (pose.p[1] - base.p[1]).abs() >= cfg.translation.search_radius.get() as f64
            || (wrap_angle((pose.angle() - angle as f64) as f32).abs() >= cfg.max_angle)
            || (pose.scale() / scale as f64).ln().abs() >= cfg.max_log_scale as f64
        {
            out.status = TranslationStatus::SearchBoundary;
            return Ok(out);
        }
        if let Some(check) = cfg.translation.region_check {
            let mut regions = [(pose, [[0.0f32; 4]; 4]); 4];
            for r in [
                [0, a.w / 2, 0, a.h / 2],
                [a.w / 2, a.w, 0, a.h / 2],
                [0, a.w / 2, a.h / 2, a.h],
                [a.w / 2, a.w, a.h / 2, a.h],
            ] {
                // Each region is parameterized about its own centroid and RMS radius, so
                // its covariance describes the region itself, without extrapolation.
                let Some((centroid, rms)) = extent(a, r) else {
                    continue;
                };
                if let Ok(f) = refine::refine(
                    a,
                    b,
                    pose.recentred(centroid, rms),
                    r,
                    (cfg.translation.min_support.get() / 4).max(16),
                    &cfg.translation.tuning,
                    &mut self.scratch,
                ) {
                    regions[out.regions] = (f.pose, f.covariance);
                    out.regions += 1;
                }
            }
            if out.regions >= 2 {
                let (error, chi2) = regional_disagreement(&pose, &regions[..out.regions], check);
                out.region_error = Some(error as f32);
                out.region_chi2 = Some((chi2 as f32).min(f32::MAX));
                if chi2.is_nan() || chi2 > check.max_chi2 as f64 {
                    out.status = TranslationStatus::ModelMismatch;
                }
            }
        }
        Ok(out)
    }
}
/// Centroid and RMS radius of a region's valid previous pixels.
fn extent(a: &Plane, region: [usize; 4]) -> Option<([f64; 2], f64)> {
    let (mut n, mut sx, mut sy, mut sxx) = (0usize, 0.0f64, 0.0f64, 0.0f64);
    for y in region[2]..region[3] {
        for x in region[0]..region[1] {
            if a.valid[y * a.w + x] {
                let (x, y) = (x as f64, y as f64);
                n += 1;
                sx += x;
                sy += y;
                sxx += x * x + y * y;
            }
        }
    }
    if n == 0 {
        return None;
    }
    let c = [sx / n as f64, sy / n as f64];
    let rms = (sxx / n as f64 - c[0] * c[0] - c[1] * c[1]).max(0.0).sqrt();
    Some((c, rms.max(1.0)))
}

/// Each regional fit against the global one at the region's centroid `c` and RMS radius
/// `ρ`: `d = [Δq(c), ρ Δangle, ρ Δlog-scale]` in px, normalized by the regional 4×4
/// covariance (scaled to px) plus the bias floor. For nested fits
/// `Var(θ_r − θ_g) = C_r − C_g ≤ C_r`, so the regional covariance alone bounds the
/// discrepancy's. Returns the largest `|Δq(c)| + ρ |(Δangle, Δlog-scale)|`, a bound on the
/// two maps' disagreement within radius `ρ`, and the largest χ² (4 dof).
fn regional_disagreement(
    global: &Pose,
    regions: &[(Pose, [[f32; 4]; 4])],
    check: RegionCheck,
) -> (f64, f64) {
    let floor = (check.bias_floor_px as f64).powi(2);
    let (mut error, mut chi2) = (0.0f64, 0.0f64);
    for (region, covariance) in regions {
        let (c, rho) = (region.center, region.radius);
        let (q, g) = (region.map(c[0], c[1]), global.map(c[0], c[1]));
        let turn = region.angle() - global.angle();
        let turn = turn.sin().atan2(turn.cos());
        let stretch = (region.scale() / global.scale()).ln();
        let d = Vector4::new(q[0] - g[0], q[1] - g[1], rho * turn, rho * stretch);
        let units = Vector4::new(1.0, 1.0, rho, rho);
        let s = Matrix4::from_fn(|i, j| {
            covariance[i][j] as f64 * units[i] * units[j] + if i == j { floor } else { 0.0 }
        });
        let q = s
            .cholesky()
            .map_or(f64::INFINITY, |chol| d.dot(&chol.solve(&d)));
        error = error.max(d[0].hypot(d[1]) + d[2].hypot(d[3]));
        chi2 = chi2.max(if q.is_finite() { q } else { f64::INFINITY });
    }
    (error, chi2)
}

#[cfg(test)]
mod tests;
