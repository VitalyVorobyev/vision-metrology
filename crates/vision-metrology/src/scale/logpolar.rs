//! [`estimate_scale_logpolar`]: scale (and, optionally, rotation) from a
//! log-polar ZNCC correlation — Fourier-Mellin's classic trick, without an
//! FFT.
//!
//! Both sides of the correlation are **synthesized edge-density rasters**,
//! not raw photometric patches: a [`ShapeModel`] stores teach-time edge
//! points (format 4), not pixels, so there is no reference *image* patch to
//! reach for. The model's own teach points are splatted onto a small
//! canvas (a Gaussian dab per point, weighted by its own gradient
//! magnitude); the scene side runs [`Edge2DDetector`] over a crop around
//! `approx_center` and splats *its* edgels the same way. Comparing two
//! edge-density fields rather than two raw-intensity patches is also what
//! `matching::ShapeMatcher` itself does (gradient direction, not pixel
//! value) — a deliberate consistency, not an accident of implementation.

use vm_primitives::{
    BorderMode, Edge2DConfig, Edge2DDetector, Error, Image, ImageView, Point2f, Rect2f, Vec2f,
};

use crate::corr::{CorrConfig, CorrMetric, CorrTemplate, CorrTemplateConfig, find};
use crate::matching::ShapeModel;
use crate::warp::{Interp, Map};

use super::ScaleEstimate;

/// Log-polar template rows (radius) / columns (angle, 2° per column).
const LP_ROWS: usize = 96;
const LP_COLS: usize = 180;
/// `r_min` as a fraction of the model's own level-0 radius; floored at 2 px.
const MIN_RADIUS_FRACTION: f32 = 0.08;
/// `r_max` margin over the model's own level-0 radius.
const RADIUS_MARGIN: f32 = 1.15;
/// Splat dab: a 3x3 kernel, `exp(-d2 / (2 * sigma^2))` with `sigma ~= 0.85`.
const DAB_FALLOFF: f32 = 1.4;

/// Parameters for [`estimate_scale_logpolar`].
#[derive(Debug, Clone, PartialEq)]
pub struct LogPolarScaleConfig {
    /// Scale search range, relative to the model's own taught scale
    /// (`1.0`). The scene's log-polar unwrap covers this whole range at the
    /// template's own log-radius resolution, so widening it costs more rows
    /// (and therefore more correlation work) — still one correlation, not
    /// a scan.
    pub scale_search: (f32, f32),
    /// Rotation search margin either side of zero, radians. `None` (the
    /// default) assumes negligible rotation between teach and scene and
    /// searches scale only — cheaper, and the common case for a fixture
    /// that only drifts along its optical axis. `Some(margin)` widens the
    /// scene's angular unwrap by `margin` on each side and reports the
    /// shift as `angle` — a rotation *larger* than `margin` is outside the
    /// search window and is not detected (no error; the correlation peak
    /// simply lands at the window edge).
    pub angle_margin: Option<f32>,
    /// Edge detector used to build the scene's own edge-density raster.
    pub edge: Edge2DConfig,
}

impl Default for LogPolarScaleConfig {
    fn default() -> Self {
        Self {
            scale_search: (0.4, 2.5),
            angle_margin: None,
            edge: Edge2DConfig::default(),
        }
    }
}

/// Estimate scale (and, with [`LogPolarScaleConfig::angle_margin`] set,
/// rotation) via log-polar ZNCC correlation.
///
/// `score` is the underlying [`crate::corr::CorrMatch::score`] — ZNCC in
/// `[-1, 1]` — of the log-polar correlation, **not** comparable to
/// [`estimate_scale_moments`](super::estimate_scale_moments)'s fill-fraction
/// score.
///
/// # Errors
/// - [`Error::InvalidConfig`] when `scale_search` is not a positive,
///   ordered range, or `approx_center` is far enough off-image that the
///   scene crop does not overlap it.
/// - [`Error::InvalidConfig`] when `model` has no stored teach data
///   (format-3 document, or too few surviving points) — see
///   [`ShapeModel::resample_at`] for the same requirement and why.
/// - [`Error::Degenerate`] when the model's own radius is too small for a
///   meaningful log-radius range, or corrmatch finds the synthesized
///   template or scene raster degenerate (near-zero variance — an
///   under-textured crop).
pub fn estimate_scale_logpolar(
    model: &ShapeModel,
    scene: &ImageView<'_, u8>,
    approx_center: Point2f,
    cfg: &LogPolarScaleConfig,
) -> Result<ScaleEstimate, Error> {
    let (lo, hi) = cfg.scale_search;
    if !(lo > 0.0 && hi >= lo) {
        return Err(Error::InvalidConfig(
            "estimate_scale_logpolar: scale_search must be positive and ordered",
        ));
    }
    let teach = model.teach_points().ok_or(Error::InvalidConfig(
        "estimate_scale_logpolar: this model has no stored teach-time edge data — see \
         ShapeModel::resample_at's docs for the same requirement",
    ))?;

    let r_ref = model
        .level(0)
        .map(|l| l.radius())
        .filter(|r| *r > 0.0)
        .ok_or(Error::Degenerate(
            "estimate_scale_logpolar: model has zero level-0 radius",
        ))?;
    let r_min = (r_ref * MIN_RADIUS_FRACTION).max(2.0);
    let r_max_ref = r_ref * RADIUS_MARGIN;
    if r_max_ref <= r_min * 1.2 {
        return Err(Error::Degenerate(
            "estimate_scale_logpolar: model radius too small for a log-polar range",
        ));
    }
    let r_min_scene = (r_min * lo).max(1.0);
    let r_max_scene = r_max_ref * hi;

    // ---- teach-side synthetic raster --------------------------------
    let canvas_r = (r_max_ref.ceil() as i64 + 2).max(4);
    let canvas_size = (2 * canvas_r + 1) as usize;
    let canvas_center = Point2f::new(canvas_r as f32, canvas_r as f32);
    let teach_pts: Vec<(Vec2f, f32)> = teach.iter().map(|p| (p.d, p.strength)).collect();
    let teach_canvas = splat_canvas(canvas_size, canvas_size, canvas_center, &teach_pts);

    // ---- scene-side synthetic raster ---------------------------------
    let half = (r_max_scene.ceil() as i64 + 2).max(4);
    let (sw, sh) = (scene.width() as i64, scene.height() as i64);
    let cx = approx_center.x.round() as i64;
    let cy = approx_center.y.round() as i64;
    let x0 = (cx - half).clamp(0, sw);
    let y0 = (cy - half).clamp(0, sh);
    let x1 = (cx + half + 1).clamp(0, sw);
    let y1 = (cy + half + 1).clamp(0, sh);
    if x1 <= x0 || y1 <= y0 {
        return Err(Error::InvalidConfig(
            "estimate_scale_logpolar: approx_center's crop does not overlap the scene",
        ));
    }
    let (crop_w, crop_h) = ((x1 - x0) as usize, (y1 - y0) as usize);
    let crop = scene.subview(x0 as usize, y0 as usize, crop_w, crop_h)?;
    let crop_center = Point2f::new(approx_center.x - x0 as f32, approx_center.y - y0 as f32);

    let mut det = Edge2DDetector::new();
    let edgels = det.detect(&crop, &cfg.edge);
    let scene_pts: Vec<(Vec2f, f32)> = edgels
        .iter()
        .map(|e| (e.p - crop_center, e.strength))
        .collect();
    let scene_canvas = splat_canvas(crop_w, crop_h, crop_center, &scene_pts);

    // ---- log-polar unwrap: same log-radius pitch on both sides -------
    let pitch_r = (r_max_ref / r_min).ln() / LP_ROWS as f32;
    let rows_scene = ((r_max_scene / r_min_scene).ln() / pitch_r)
        .round()
        .max(LP_ROWS as f32) as usize;

    let angle_margin = cfg.angle_margin.unwrap_or(0.0).max(0.0);
    let pitch_phi = core::f32::consts::TAU / LP_COLS as f32;
    let extra_cols = if angle_margin > 0.0 {
        (2.0 * angle_margin / pitch_phi).round() as usize
    } else {
        0
    };
    let cols_scene = LP_COLS + extra_cols;

    let teach_map = Map::log_polar(
        canvas_center,
        r_min..r_max_ref,
        0.0..core::f32::consts::TAU,
        LP_COLS,
        LP_ROWS,
    );
    let mut teach_lp = vec![0u8; LP_COLS * LP_ROWS];
    teach_map.apply(
        &teach_canvas.as_view(),
        &mut teach_lp,
        Interp::Bilinear,
        BorderMode::Constant(0),
    )?;
    let teach_lp_img = Image::from_vec(LP_COLS, LP_ROWS, teach_lp).expect("size matches buffer");

    let scene_map = Map::log_polar(
        crop_center,
        r_min_scene..r_max_scene,
        -angle_margin..(core::f32::consts::TAU + angle_margin),
        cols_scene,
        rows_scene,
    );
    let mut scene_lp = vec![0u8; cols_scene * rows_scene];
    scene_map.apply(
        &scene_canvas.as_view(),
        &mut scene_lp,
        Interp::Bilinear,
        BorderMode::Constant(0),
    )?;
    let scene_lp_img =
        Image::from_vec(cols_scene, rows_scene, scene_lp).expect("size matches buffer");

    // ---- correlate: translation in (phi, log r) is (angle, log scale) --
    let template = CorrTemplate::from_image(
        &teach_lp_img.as_view(),
        Rect2f {
            x: 0.0,
            y: 0.0,
            width: LP_COLS as f32,
            height: LP_ROWS as f32,
        },
        &CorrTemplateConfig {
            rotation: false,
            ..CorrTemplateConfig::default()
        },
    )?;
    let m = find(
        &template,
        &scene_lp_img.as_view(),
        &CorrConfig {
            rotation: false,
            metric: CorrMetric::Zncc,
            ..CorrConfig::default()
        },
    )?;

    let x0f = m.position.x - 0.5 * (LP_COLS as f32 - 1.0);
    let y0f = m.position.y - 0.5 * (LP_ROWS as f32 - 1.0);
    let scale = lo * (y0f * pitch_r).exp();
    let angle = cfg.angle_margin.map(|_| x0f * pitch_phi - angle_margin);

    Ok(ScaleEstimate {
        scale,
        angle,
        score: m.score,
    })
}

/// Splat weighted 2-D offsets onto a `w x h` `u8` canvas as small Gaussian
/// dabs, `center + offset` positioned, saturating at 255. `data[i].max(v)`
/// (not `+=`) so overlapping dabs do not blow past 255 and bias the
/// correlation toward busy regions beyond what their edge density already
/// implies.
fn splat_canvas(w: usize, h: usize, center: Point2f, points: &[(Vec2f, f32)]) -> Image<u8> {
    let mut data = vec![0u8; w * h];
    for &(d, strength) in points {
        let cx = center.x + d.x;
        let cy = center.y + d.y;
        let ix = cx.round() as i64;
        let iy = cy.round() as i64;
        let amp = strength.clamp(0.0, 255.0);
        for dy in -1i64..=1 {
            for dx in -1i64..=1 {
                let x = ix + dx;
                let y = iy + dy;
                if x < 0 || y < 0 || x as usize >= w || y as usize >= h {
                    continue;
                }
                let d2 = (dx * dx + dy * dy) as f32;
                let val = (amp * (-d2 / DAB_FALLOFF).exp()) as u8;
                let idx = y as usize * w + x as usize;
                data[idx] = data[idx].max(val);
            }
        }
    }
    Image::from_vec(w, h, data).expect("dimensions match the buffer")
}

#[cfg(test)]
mod tests {
    use super::{LogPolarScaleConfig, estimate_scale_logpolar};
    use crate::matching::{ShapeModelBuilder, ShapeModelConfig};
    use vm_primitives::{Image, Point2f, Rect2f};

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
    fn recovers_the_scale_of_a_larger_disc() {
        let teach_img = disc(220, 110.0, 110.0, 30.0);
        let roi = Rect2f {
            x: 40.0,
            y: 40.0,
            width: 140.0,
            height: 140.0,
        };
        let model = ShapeModelBuilder::new()
            .build(&teach_img.as_view(), roi, &ShapeModelConfig::default())
            .expect("model builds");

        let scene_img = disc(220, 110.0, 110.0, 45.0);
        let est = estimate_scale_logpolar(
            &model,
            &scene_img.as_view(),
            Point2f::new(110.0, 110.0),
            &LogPolarScaleConfig::default(),
        )
        .expect("estimate succeeds");
        assert!(
            (est.scale - 1.5).abs() < 0.2,
            "expected scale near 1.5, got {}",
            est.scale
        );
        assert!(est.angle.is_none());
    }

    #[test]
    fn rejects_an_unordered_scale_search_range() {
        let teach_img = disc(220, 110.0, 110.0, 30.0);
        let roi = Rect2f {
            x: 40.0,
            y: 40.0,
            width: 140.0,
            height: 140.0,
        };
        let model = ShapeModelBuilder::new()
            .build(&teach_img.as_view(), roi, &ShapeModelConfig::default())
            .expect("model builds");
        let scene_img = disc(220, 110.0, 110.0, 30.0);
        let cfg = LogPolarScaleConfig {
            scale_search: (2.0, 0.5),
            ..Default::default()
        };
        assert!(
            estimate_scale_logpolar(
                &model,
                &scene_img.as_view(),
                Point2f::new(110.0, 110.0),
                &cfg
            )
            .is_err()
        );
    }
}
