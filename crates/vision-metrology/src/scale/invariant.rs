//! [`find_scale_invariant`]: estimate once, [`ShapeModel::resample_at`],
//! verify narrow — the whole "estimate-then-verify" strategy in one call.

use vm_primitives::{Error, ImageView, Point2f, Rect2f};

use crate::matching::{ShapeMatch, ShapeMatcher, ShapeModel, ShapeSearchConfig, pose_from};

use super::logpolar::{LogPolarScaleConfig, estimate_scale_logpolar};
use super::moments::{MomentScaleConfig, estimate_scale_moments};

/// Which estimator to use, and the hint it needs.
///
/// [`ScaleHint::Roi`] uses [`estimate_scale_moments`] (needs a part that
/// segments cleanly against its background inside `roi`);
/// [`ScaleHint::Center`] uses [`estimate_scale_logpolar`] (needs only an
/// approximate center, but needs format-4 teach data — see that function's
/// docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScaleHint {
    /// Segment and measure a blob inside this ROI.
    Roi(Rect2f),
    /// Log-polar correlate around this approximate center.
    Center(Point2f),
}

/// Parameters for [`find_scale_invariant`].
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ScaleInvariantConfig {
    /// Used when the hint is [`ScaleHint::Roi`].
    pub moments: MomentScaleConfig,
    /// Used when the hint is [`ScaleHint::Center`].
    pub logpolar: LogPolarScaleConfig,
    /// Verify-stage search over the resampled model. Its own `scale_range`
    /// is `None` by default, which lets [`ShapeModel::resample_at`]'s own
    /// narrow band (see that function) govern the search — widening it
    /// here defeats the constant-cost point of estimate-then-verify.
    pub search: ShapeSearchConfig,
}

/// Estimate the scene's scale relative to `model`, resample `model` at that
/// estimate, and search a narrow band around it — the whole
/// estimate-then-verify strategy (module docs) as one call.
///
/// Returned [`ShapeMatch`]es have their `pose` rebuilt into `model`'s own
/// (pre-resample) reference-image frame, so a caller — `MetrologyModel`,
/// `model_frame_map`, or anything else that consumes `ShapeMatch::pose` —
/// sees no difference from a `model.find(...)` call, except that
/// `.scale()` now reads close to the scene's true scale rather than close
/// to `1.0`: **`found_scale = ŝ · verify_scale`**, applied here, not left
/// for the caller to multiply back in. `.position`, `.angle()`, `.score`
/// and `.support` are unaffected by the resample (see the module's
/// implementation note in `resample_at` — a uniform rescale does not move
/// the reference point's *reported* pixel location, and does not rotate
/// anything).
///
/// # Errors
/// Propagates the chosen estimator's errors verbatim ([`estimate_scale_moments`]
/// for [`ScaleHint::Roi`], [`estimate_scale_logpolar`] for
/// [`ScaleHint::Center`]) and [`ShapeModel::resample_at`]'s. Returns
/// `Ok(Vec::new())`, not an error, when the estimate is fine but nothing
/// scores above `cfg.search.min_score` in the verify stage — same
/// "not found is a result" convention as [`ShapeMatcher::find`] itself.
pub fn find_scale_invariant(
    model: &ShapeModel,
    scene: &ImageView<'_, u8>,
    hint: ScaleHint,
    cfg: &ScaleInvariantConfig,
) -> Result<Vec<ShapeMatch>, Error> {
    let est = match hint {
        ScaleHint::Roi(roi) => estimate_scale_moments(model, scene, roi, &cfg.moments)?,
        ScaleHint::Center(c) => estimate_scale_logpolar(model, scene, c, &cfg.logpolar)?,
    };

    let resampled = model.resample_at(est.scale)?;
    let mut matcher = ShapeMatcher::new();
    let mut matches = matcher.find(scene, &resampled, &cfg.search);

    let origin = model.origin();
    for m in &mut matches {
        let true_scale = m.scale() * est.scale;
        m.pose = pose_from(m.position, m.angle(), true_scale, origin);
    }
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::{ScaleHint, ScaleInvariantConfig, find_scale_invariant};
    use crate::matching::{ShapeModelBuilder, ShapeModelConfig};
    use vm_primitives::{Image, Rect2f};

    fn disc(size: usize, cx: f32, cy: f32, r: f32) -> Image<u8> {
        let mut data = vec![25u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() < r {
                    data[y * size + x] = 230;
                }
            }
        }
        Image::from_vec(size, size, data).expect("valid image")
    }

    #[test]
    fn roi_hint_finds_and_reports_the_true_scale() {
        let teach_img = disc(260, 130.0, 130.0, 30.0);
        let roi = Rect2f {
            x: 60.0,
            y: 60.0,
            width: 140.0,
            height: 140.0,
        };
        let model = ShapeModelBuilder::new()
            .build(&teach_img.as_view(), roi, &ShapeModelConfig::default())
            .expect("model builds");

        let scene = disc(260, 130.0, 130.0, 54.0); // ~1.8x
        let cfg = ScaleInvariantConfig {
            moments: crate::scale::MomentScaleConfig {
                polarity: crate::scale::BlobPolarity::BrightOnDark,
                ..Default::default()
            },
            ..Default::default()
        };
        let matches = find_scale_invariant(&model, &scene.as_view(), ScaleHint::Roi(roi), &cfg)
            .expect("estimate-then-verify succeeds");
        let m = matches.first().expect("found the disc");
        assert!(
            (m.scale() - 1.8).abs() < 0.15,
            "expected scale near 1.8, got {}",
            m.scale()
        );
    }
}
