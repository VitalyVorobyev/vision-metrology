//! Local native masked similarity registration. No angle bank or global scale search.
mod config;
mod refine;
mod result;
pub use config::SimilarityConfig;
pub use result::Similarity;

use super::{TranslationEstimator, TranslationStatus, plane::Plane};
use nalgebra::SVector;
use refine::Pose;
use vm_primitives::{Error, ImageView, Pixel, Similarity2f, Vec2f, similarity_parts, wrap_angle};

/// Reusable local estimator of rotation, uniform scale and translation.
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
    residuals: Vec<f64>,
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
        self.residuals.reserve(n - self.residuals.len().min(n));
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
        let mut init_cfg = cfg.translation.clone();
        init_cfg.max_region_error = None;
        let initial = self
            .initializer
            .estimate(&av, &bv, Some(&am), Some(&bm), None, &init_cfg)?;
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
        // The coarse peak remains a supported initialization even if the translation-only
        // numerical model cannot converge under observable rotation/scale.
        if !matches!(
            initial.status,
            TranslationStatus::Accepted | TranslationStatus::RefinementFailed
        ) {
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
            &mut self.residuals,
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
        if let Some(limit) = cfg.translation.max_region_error {
            let mut regions = [pose; 4];
            for r in [
                [0, a.w / 2, 0, a.h / 2],
                [a.w / 2, a.w, 0, a.h / 2],
                [0, a.w / 2, a.h / 2, a.h],
                [a.w / 2, a.w, a.h / 2, a.h],
            ] {
                if let Ok(f) = refine::refine(
                    a,
                    b,
                    pose,
                    r,
                    (cfg.translation.min_support.get() / 4).max(16),
                    &cfg.translation.tuning,
                    &mut self.residuals,
                ) {
                    regions[out.regions] = f.pose;
                    out.regions += 1;
                }
            }
            if out.regions >= 2 {
                let mut error = 0.0f64;
                for i in 0..out.regions {
                    for j in 0..i {
                        for (x, y) in [
                            (0., 0.),
                            ((a.w - 1) as f64, 0.),
                            (0., (a.h - 1) as f64),
                            ((a.w - 1) as f64, (a.h - 1) as f64),
                        ] {
                            let (p, q) = (regions[i].map(x, y), regions[j].map(x, y));
                            error = error.max((p[0] - q[0]).hypot(p[1] - q[1]));
                        }
                    }
                }
                out.region_error = Some(error as f32);
                if error > limit as f64 {
                    out.status = TranslationStatus::ModelMismatch;
                }
            }
        }
        Ok(out)
    }
}
#[cfg(test)]
mod tests;
