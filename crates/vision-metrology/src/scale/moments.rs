//! [`estimate_scale_moments`]: scale from a segmented blob's spatial spread.

use vm_primitives::{Error, Image, ImageView, Point2f, Rect2f};

use crate::contour::Connectivity;
use crate::matching::ShapeModel;
use crate::segment::{component_stats, label_connected_components_u8, otsu_threshold_u8};

use super::ScaleEstimate;

/// Which side of the Otsu threshold the object's own pixels fall on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlobPolarity {
    /// The object is darker than its background.
    #[default]
    DarkOnBright,
    /// The object is brighter than its background.
    BrightOnDark,
}

/// Parameters for [`estimate_scale_moments`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MomentScaleConfig {
    /// Which side of the ROI's own Otsu threshold is foreground.
    pub polarity: BlobPolarity,
    /// Connected components smaller than this many pixels are not
    /// candidates for "the part".
    pub min_area: u32,
}

impl Default for MomentScaleConfig {
    fn default() -> Self {
        Self {
            polarity: BlobPolarity::default(),
            min_area: 25,
        }
    }
}

/// Estimate a scale factor by comparing a segmented scene blob's own outer
/// radius to the taught model's.
///
/// Thresholds `roi` with its own Otsu threshold ([`crate::segment::otsu_threshold_u8`]),
/// labels connected components ([`crate::segment::label_connected_components_u8`],
/// 8-connected), and keeps the component nearest the ROI's own center — the
/// working assumption is an isolated part roughly centered in `roi`, the
/// same assumption a caller supplying a tracking/detection ROI already
/// makes. Compares that component's own **maximum distance from its
/// centroid to any of its own foreground pixels** against
/// `model.level(0).radius()` — the model's own maximum `|d|` — rather than
/// e.g. a filled-area or radius-of-gyration ratio: those mix a *filled*
/// scene silhouette against the model's *boundary-only* edge points, which
/// are systematically different quantities at the same true radius (a
/// filled disc's radius of gyration is `R/√2`; its boundary ring's is `R`),
/// and area/`R²` compounds that bias further. An outer-radius comparison
/// is one consistent notion of "how big is the object" on both sides
/// instead. This estimator works on **any** `ShapeModel` — format 3 or
/// 4 — since `level(0).radius()` needs no stored teach data, unlike
/// [`resample_at`](crate::matching::ShapeModel::resample_at) or
/// [`estimate_scale_logpolar`](super::estimate_scale_logpolar).
///
/// `score` is the chosen component's *fill fraction* — `pixel_count /
/// bbox_area` — a compactness diagnostic, not a probability: `1.0` for a
/// solid disc or rectangle, lower for a ragged or hollow silhouette. It is
/// not comparable to [`estimate_scale_logpolar`](super::estimate_scale_logpolar)'s ZNCC score.
///
/// `angle` is always `None` — an axis-aligned area comparison carries no
/// rotation information.
///
/// # Errors
/// - [`Error::InvalidConfig`] when `roi` does not overlap `scene`.
/// - [`Error::InsufficientData`] when no component of at least `min_area`
///   pixels survives thresholding.
/// - [`Error::Degenerate`] when the model's own points collapse to a single
///   position (zero spatial extent — nothing to compare against).
pub fn estimate_scale_moments(
    model: &ShapeModel,
    scene: &ImageView<'_, u8>,
    roi: Rect2f,
    cfg: &MomentScaleConfig,
) -> Result<ScaleEstimate, Error> {
    let (sw, sh) = (scene.width(), scene.height());
    let x0 = roi.x.round().clamp(0.0, sw as f32) as usize;
    let y0 = roi.y.round().clamp(0.0, sh as f32) as usize;
    let x1 = (roi.x + roi.width).round().clamp(0.0, sw as f32) as usize;
    let y1 = (roi.y + roi.height).round().clamp(0.0, sh as f32) as usize;
    if x1 <= x0 || y1 <= y0 {
        return Err(Error::InvalidConfig(
            "estimate_scale_moments: roi does not overlap the scene",
        ));
    }
    let sub = scene.subview(x0, y0, x1 - x0, y1 - y0)?;

    let t = otsu_threshold_u8(&sub);
    let (sub_w, sub_h) = (sub.width(), sub.height());
    let mut mask = vec![0u8; sub_w * sub_h];
    for y in 0..sub_h {
        let row = sub.row(y);
        for x in 0..sub_w {
            let fg = match cfg.polarity {
                BlobPolarity::DarkOnBright => row[x] <= t,
                BlobPolarity::BrightOnDark => row[x] > t,
            };
            mask[y * sub_w + x] = u8::from(fg) * 255;
        }
    }
    let mask_img = Image::from_vec(sub_w, sub_h, mask).expect("dimensions match the buffer");
    let cl = label_connected_components_u8(&mask_img.as_view(), Connectivity::C8);
    let stats = component_stats(&cl, cfg.min_area);
    let center = Point2f::new(sub_w as f32 * 0.5, sub_h as f32 * 0.5);
    let best = stats
        .iter()
        .min_by(|a, b| {
            let da = (a.centroid - center).norm_squared();
            let db = (b.centroid - center).norm_squared();
            da.partial_cmp(&db).expect("finite centroids")
        })
        .ok_or(Error::InsufficientData {
            need: cfg.min_area.max(1) as usize,
            got: 0,
        })?;

    let (w, h) = (cl.label_map.width(), cl.label_map.height());
    let data = cl.label_map.data();
    let mut max_sq = 0.0f64;
    let mut n = 0u64;
    for y in 0..h {
        for x in 0..w {
            if data[y * w + x] != best.label {
                continue;
            }
            let dx = x as f64 - best.centroid.x as f64;
            let dy = y as f64 - best.centroid.y as f64;
            max_sq = max_sq.max(dx * dx + dy * dy);
            n += 1;
        }
    }
    debug_assert!(n > 0, "component_stats already filtered empty labels");
    let r_scene = max_sq.sqrt();

    let r_model = model.level(0).map_or(0.0, |l| l.radius() as f64);
    if r_model <= 0.0 {
        return Err(Error::Degenerate(
            "estimate_scale_moments: model has zero spatial extent",
        ));
    }

    let scale = (r_scene / r_model) as f32;
    let bbox_area = (best.bbox.width.max(1.0) * best.bbox.height.max(1.0)) as f64;
    let fill = (best.pixel_count as f64 / bbox_area) as f32;
    Ok(ScaleEstimate {
        scale,
        angle: None,
        score: fill.clamp(0.0, 1.0),
    })
}

#[cfg(test)]
mod tests {
    use super::{BlobPolarity, MomentScaleConfig, estimate_scale_moments};
    use crate::matching::{ShapeModelBuilder, ShapeModelConfig};
    use vm_primitives::{Image, Rect2f};

    fn disc(size: usize, cx: f32, cy: f32, r: f32) -> Image<u8> {
        let mut data = vec![30u8; size * size];
        for y in 0..size {
            for x in 0..size {
                let dx = x as f32 - cx;
                let dy = y as f32 - cy;
                if (dx * dx + dy * dy).sqrt() < r {
                    data[y * size + x] = 220;
                }
            }
        }
        Image::from_vec(size, size, data).expect("valid image")
    }

    #[test]
    fn recovers_the_scale_of_a_larger_disc() {
        let teach_img = disc(200, 100.0, 100.0, 30.0);
        let roi = Rect2f {
            x: 40.0,
            y: 40.0,
            width: 120.0,
            height: 120.0,
        };
        let model = ShapeModelBuilder::new()
            .build(&teach_img.as_view(), roi, &ShapeModelConfig::default())
            .expect("model builds");

        // Scene disc at 1.6x the taught radius, same relative ROI (so the
        // scene component's own centroid sits near the ROI center by
        // construction, matching this estimator's "isolated, roughly
        // centered part" assumption).
        let scene_img = disc(200, 100.0, 100.0, 48.0);
        let est = estimate_scale_moments(
            &model,
            &scene_img.as_view(),
            roi,
            &MomentScaleConfig {
                polarity: BlobPolarity::BrightOnDark,
                ..Default::default()
            },
        )
        .expect("estimate succeeds");
        assert!(
            (est.scale - 1.6).abs() < 0.1,
            "expected scale near 1.6, got {}",
            est.scale
        );
        assert!(est.angle.is_none());
    }

    #[test]
    fn an_roi_with_nothing_foreground_errors() {
        let teach_img = disc(200, 100.0, 100.0, 30.0);
        let roi = Rect2f {
            x: 40.0,
            y: 40.0,
            width: 120.0,
            height: 120.0,
        };
        let model = ShapeModelBuilder::new()
            .build(&teach_img.as_view(), roi, &ShapeModelConfig::default())
            .expect("model builds");

        let flat = Image::new_fill(200, 200, 128u8);
        let cfg = MomentScaleConfig::default();
        assert!(estimate_scale_moments(&model, &flat.as_view(), roi, &cfg).is_err());
    }
}
