//! Specialized two-mask registration, separate from ordinary corrmatch matching.

mod config;
mod plane;
mod refine;
mod result;
mod search;

pub use config::{TranslationConfig, TranslationTuning};
pub use result::{Translation, TranslationStatus};

use plane::Plane;
use vm_primitives::{Error, ImageView, Pixel, Vec2f};

/// Reusable masked subpixel translation estimator.
///
/// Nonzero mask pixels are valid. Masks have their own image's coordinate frame;
/// sampling and pyramid reduction require every contributing pixel to be valid.
/// Correlation initializes robust `(dx, dy, gain, offset)` Gauss–Newton refinement.
/// Region checks can reject observable rotation or deformation, but do not prove
/// the translation model when there is little spatially distributed texture.
///
/// Scratch grows on the first call or when dimensions/configured capacity grow.
/// Subsequent estimates with the same sizes and configuration allocate nothing.
/// Failed results must not be used as fixtures. There is no temporal adaptation.
#[derive(Default)]
pub struct TranslationEstimator {
    previous: Vec<Plane>,
    current: Vec<Plane>,
    scores: Vec<([i32; 2], f64, usize)>,
    scratch: refine::Workspace,
}

impl TranslationEstimator {
    /// Construct empty scratch; the first estimate prepares its capacity.
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate content motion from `previous` into `current`.
    ///
    /// The prediction centers the bounded search; `None` means zero motion.
    /// Intensities may be u8, u16 or f32, without quantization. Dimensions may
    /// differ. Photometry is `previous ≈ gain * current(x + shift) + offset`.
    ///
    /// # Errors
    /// Invalid configuration, mask dimensions, images smaller than 4×4,
    /// non-finite prediction, or non-finite valid f32 pixels.
    pub fn estimate<P: Pixel>(
        &mut self,
        previous: &ImageView<'_, P>,
        current: &ImageView<'_, P>,
        previous_mask: Option<&ImageView<'_, u8>>,
        current_mask: Option<&ImageView<'_, u8>>,
        prediction: Option<Vec2f>,
        cfg: &TranslationConfig,
    ) -> Result<Translation, Error> {
        cfg.validate()?;
        for (img_w, img_h, mask) in [
            (previous.width(), previous.height(), previous_mask),
            (current.width(), current.height(), current_mask),
        ] {
            if img_w < 4
                || img_h < 4
                || img_w > i32::MAX as usize / 2
                || img_h > i32::MAX as usize / 2
                || mask.is_some_and(|m| m.width() != img_w || m.height() != img_h)
            {
                return Err(Error::InvalidConfig(
                    "invalid translation image or mask dimensions",
                ));
            }
        }
        let prediction = prediction.unwrap_or_else(|| Vec2f::new(0.0, 0.0));
        if !prediction.x.is_finite()
            || !prediction.y.is_finite()
            || prediction.x.abs() > 1e6
            || prediction.y.abs() > 1e6
        {
            return Err(Error::InvalidConfig("invalid translation prediction"));
        }
        let mut levels = 1;
        let mut size = previous
            .width()
            .min(previous.height())
            .min(current.width())
            .min(current.height());
        while levels < cfg.tuning.max_levels.get() && size >= 32 {
            levels += 1;
            size /= 2;
        }
        self.previous.resize_with(levels, Plane::default);
        self.current.resize_with(levels, Plane::default);
        let (mean_a, scale_a, na) = self.previous[0].load(previous, previous_mask)?;
        let (mean_b, scale_b, nb) = self.current[0].load(current, current_mask)?;
        for l in 1..levels {
            let (a, b) = self.previous.split_at_mut(l);
            b[0].downsample(&a[l - 1]);
            let (a, b) = self.current.split_at_mut(l);
            b[0].downsample(&a[l - 1]);
        }
        self.scratch.residuals.reserve(
            previous.width() * previous.height()
                - self
                    .scratch
                    .residuals
                    .len()
                    .min(previous.width() * previous.height()),
        );
        let mut shift = [prediction.x as f64, prediction.y as f64];
        let mut peak = None;
        let mut margin = f64::INFINITY;
        let radius = cfg.search_radius.get() as f64;
        let side = 2 * cfg.search_radius.get() + 1;
        self.scores
            .reserve(side * side - self.scores.len().min(side * side));
        // Prepare all buffers even when this valid call has no usable evidence.
        if na.min(nb) < cfg.min_support.get() {
            return Ok(Translation::empty(TranslationStatus::InsufficientSupport));
        }
        if scale_a <= 0.0 || scale_b <= 0.0 {
            return Ok(Translation::empty(TranslationStatus::LowTexture));
        }
        for l in (0..levels).rev() {
            let scale = (1usize << l) as f64;
            let center = [prediction.x as f64 / scale, prediction.y as f64 / scale];
            let bound = [
                (center[0] - radius / scale).ceil() as i32,
                (center[0] + radius / scale).floor() as i32,
                (center[1] - radius / scale).ceil() as i32,
                (center[1] + radius / scale).floor() as i32,
            ];
            let bounds = if peak.is_none() {
                bound
            } else {
                [
                    (shift[0] / scale).round() as i32 - 2,
                    (shift[0] / scale).round() as i32 + 2,
                    (shift[1] / scale).round() as i32 - 2,
                    (shift[1] / scale).round() as i32 + 2,
                ]
            };
            let bounds = [
                bounds[0].max(bound[0]),
                bounds[1].min(bound[1]),
                bounds[2].max(bound[2]),
                bounds[3].min(bound[3]),
            ];
            let min = (cfg.min_support.get() / (1usize << (2 * l))).max(16);
            let p = match search::search(
                &self.previous[l],
                &self.current[l],
                bounds,
                min,
                &mut self.scores,
            ) {
                Ok(p) => p,
                Err(_) if l > 0 => {
                    peak = None;
                    continue;
                }
                Err(status) => return Ok(Translation::empty(status)),
            };
            shift = p.shift.map(|v| v as f64 * scale);
            margin = margin.min(p.margin);
            peak = Some(p);
        }
        let peak = peak.unwrap();
        let mut out = Translation::empty(TranslationStatus::Accepted);
        out.shift = Some(Vec2f::new(shift[0] as f32, shift[1] as f32));
        out.score = Some(peak.score as f32);
        out.margin = Some(margin as f32);
        out.support = peak.support;
        if peak.score < cfg.min_score as f64 || margin < cfg.min_margin as f64 {
            out.status = TranslationStatus::Ambiguous;
            return Ok(out);
        }
        if (shift[0] - prediction.x as f64).abs() >= radius - 0.5
            || (shift[1] - prediction.y as f64).abs() >= radius - 0.5
        {
            out.status = TranslationStatus::SearchBoundary;
            return Ok(out);
        }
        let (a, b) = (&self.previous[0], &self.current[0]);
        let fit = match refine::refine(
            a,
            b,
            shift,
            [0, a.w, 0, a.h],
            cfg.min_support.get(),
            &cfg.tuning,
            &mut self.scratch,
        ) {
            Ok(f) => f,
            Err(s) => {
                out.status = s;
                return Ok(out);
            }
        };
        out.shift = Some(Vec2f::new(fit.p[0] as f32, fit.p[1] as f32));
        out.support = fit.n;
        out.residual_rms = Some((fit.rms * scale_a) as f32);
        out.residual_max = Some((fit.max * scale_a) as f32);
        let gain = fit.gain * scale_a / scale_b;
        out.gain = Some(gain as f32);
        out.offset = Some((mean_a + scale_a * fit.offset - gain * mean_b) as f32);
        out.covariance = Some(fit.covariance);
        if [out.residual_rms, out.residual_max, out.gain, out.offset]
            .into_iter()
            .flatten()
            .any(|v| !v.is_finite())
        {
            out.status = TranslationStatus::RefinementFailed;
            out.residual_rms = None;
            out.residual_max = None;
            out.gain = None;
            out.offset = None;
            out.covariance = None;
            return Ok(out);
        }
        if (fit.p[0] - prediction.x as f64).abs() >= radius
            || (fit.p[1] - prediction.y as f64).abs() >= radius
        {
            out.status = TranslationStatus::SearchBoundary;
            return Ok(out);
        }
        if let Some(limit) = cfg.max_region_error {
            let mut regions = [[0.0; 2]; 4];
            for r in [
                [0, a.w / 2, 0, a.h / 2],
                [a.w / 2, a.w, 0, a.h / 2],
                [0, a.w / 2, a.h / 2, a.h],
                [a.w / 2, a.w, a.h / 2, a.h],
            ] {
                if let Ok(f) = refine::refine(
                    a,
                    b,
                    fit.p,
                    r,
                    (cfg.min_support.get() / 4).max(16),
                    &cfg.tuning,
                    &mut self.scratch,
                ) {
                    regions[out.regions] = f.p;
                    out.regions += 1;
                }
            }
            if out.regions >= 2 {
                let mut error = 0.0f64;
                for i in 0..out.regions {
                    for j in 0..i {
                        error = error.max(
                            (regions[i][0] - regions[j][0]).hypot(regions[i][1] - regions[j][1]),
                        );
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
