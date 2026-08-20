//! [`CorrTemplate`] compiles a reference patch once; [`find`] / [`find_topk`]
//! search it in a scene via corrmatch's coarse-to-fine ZNCC/SSD matcher.

use core::num::NonZeroUsize;
use std::sync::Arc;

use vm_primitives::{Error, ImageView, Point2f, Rect2f};

use super::adapter::{corr_view, extract_rect_u8, map_corr_err, scene_buf};
use super::config::{CorrConfig, CorrMetric, CorrTemplateConfig};

/// Compiled matching assets for one reference patch: corrmatch's own
/// pyramid, and — when [`CorrTemplateConfig::rotation`] is set — a per-level
/// angle bank of masked ZNCC/SSD plans.
///
/// Build once with [`CorrTemplate::from_image`], reuse across many
/// [`find`]/[`find_topk`] calls; each call clones the internal `Arc`, which
/// is cheap (the compiled assets themselves are not duplicated).
pub struct CorrTemplate {
    compiled: Arc<corrmatch::CompiledTemplate>,
    width: usize,
    height: usize,
    rotation: bool,
}

impl CorrTemplate {
    /// Compiles a reference patch: `rect` (rounded to whole pixels, clamped
    /// to `img`) taken out of `img` and handed to corrmatch's own pyramid +
    /// (optionally) angle-bank compiler.
    ///
    /// # Errors
    /// [`Error::InvalidConfig`] if `rect` does not overlap `img`, or if
    /// corrmatch rejects the patch (e.g. too small, or degenerate — zero
    /// variance).
    pub fn from_image(
        img: &ImageView<'_, u8>,
        rect: Rect2f,
        cfg: &CorrTemplateConfig,
    ) -> Result<Self, Error> {
        let (data, width, height) = extract_rect_u8(img, rect)?;
        let tpl = corrmatch::Template::new(data, width, height).map_err(map_corr_err)?;
        let max_levels = cfg.max_levels.map_or(6, NonZeroUsize::get);

        let compiled = if cfg.rotation {
            let mut compile_cfg = corrmatch::CompileConfig::default();
            compile_cfg.max_levels = max_levels;
            compile_cfg.coarse_step_deg = cfg.tuning.coarse_angle_step_deg;
            compile_cfg.min_step_deg = cfg.tuning.min_angle_step_deg;
            compile_cfg.fill_value = cfg.tuning.fill_value;
            compile_cfg.precompute_coarsest = cfg.tuning.precompute_coarsest;
            corrmatch::CompiledTemplate::compile_rotated(&tpl, compile_cfg).map_err(map_corr_err)?
        } else {
            let mut compile_cfg = corrmatch::CompileConfigNoRot::default();
            compile_cfg.max_levels = max_levels;
            corrmatch::CompiledTemplate::compile_unrotated(&tpl, compile_cfg)
                .map_err(map_corr_err)?
        };

        Ok(Self {
            compiled: Arc::new(compiled),
            width,
            height,
            rotation: cfg.rotation,
        })
    }

    /// Template width in pixels, as compiled (the rounded `rect`).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Template height in pixels, as compiled (the rounded `rect`).
    pub fn height(&self) -> usize {
        self.height
    }

    /// Whether this template was compiled with rotation support.
    pub fn is_rotated(&self) -> bool {
        self.rotation
    }
}

/// One match: the template's **centre** in scene coordinates, its rotation,
/// and a correlation score.
///
/// Not [`ShapeMatch`](crate::matching::ShapeMatch): `score` is a raw
/// correlation coefficient (or negative SSE), not an occlusion-robust
/// `1 - occluded_fraction`, and there is no scale — corrmatch has no scale
/// search (see the `corr` module docs for the estimate-then-verify path
/// that closes that gap when it is needed).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrMatch {
    /// Template centre in scene, level-0 pixel-centre coordinates.
    pub position: Point2f,
    /// Rotation, radians. Zero unless the search enabled rotation. Same
    /// sign convention as
    /// [`ShapeMatch::angle`](crate::matching::ShapeMatch::angle) — pinned by
    /// `tests/corrmatch_bridge.rs`.
    pub angle: f32,
    /// Correlation score: ZNCC in `[-1, 1]` (higher is better), or SSD as a
    /// negative sum-of-squared-differences (higher is still better).
    pub score: f32,
}

fn build_match_config(
    cfg: &CorrConfig,
    template_rotation: bool,
) -> Result<corrmatch::MatchConfig, Error> {
    if cfg.rotation && !template_rotation {
        return Err(Error::InvalidConfig(
            "corr: rotation search requires a CorrTemplate compiled with CorrTemplateConfig { rotation: true, .. }",
        ));
    }
    let mut mc = corrmatch::MatchConfig::default();
    mc.metric = match cfg.metric {
        CorrMetric::Zncc => corrmatch::Metric::Zncc,
        CorrMetric::Ssd => corrmatch::Metric::Ssd,
    };
    mc.rotation = if cfg.rotation {
        corrmatch::RotationMode::Enabled
    } else {
        corrmatch::RotationMode::Disabled
    };
    mc.parallel = cfg.tuning.parallel;
    mc.max_image_levels = cfg.tuning.max_image_levels.map_or(6, NonZeroUsize::get);
    mc.beam_width = cfg.tuning.beam_width;
    mc.per_angle_topk = cfg.tuning.per_angle_topk;
    mc.nms_radius = cfg.tuning.nms_radius;
    mc.roi_radius = cfg.tuning.roi_radius;
    mc.angle_half_range_steps = cfg.tuning.angle_half_range_steps;
    mc.min_var_i = cfg.tuning.min_var_i;
    mc.min_score = cfg.min_score.unwrap_or(f32::NEG_INFINITY);
    Ok(mc)
}

fn to_corr_match(m: corrmatch::Match, tw: usize, th: usize) -> CorrMatch {
    CorrMatch {
        position: Point2f::new(m.x + 0.5 * (tw as f32 - 1.0), m.y + 0.5 * (th as f32 - 1.0)),
        angle: m.angle_deg.to_radians(),
        score: m.score,
    }
}

/// Finds `template`'s best match in `scene`.
///
/// # Errors
/// [`Error::InvalidConfig`] if `cfg.rotation` is set on a template compiled
/// without rotation support, or `scene`'s dimensions are invalid.
/// [`Error::Degenerate`] if corrmatch finds no candidate at all (e.g.
/// `scene` smaller than the template).
pub fn find(
    template: &CorrTemplate,
    scene: &ImageView<'_, u8>,
    cfg: &CorrConfig,
) -> Result<CorrMatch, Error> {
    let match_cfg = build_match_config(cfg, template.rotation)?;
    let buf = scene_buf(scene);
    let view = corr_view(buf.as_slice(), scene.width(), scene.height())?;
    let matcher = corrmatch::Matcher::from_arc(template.compiled.clone()).with_config(match_cfg);
    let m = matcher.match_image(view).map_err(map_corr_err)?;
    Ok(to_corr_match(m, template.width, template.height))
}

/// Finds up to `k` matches in `scene`, best score first.
///
/// # Errors
/// Same as [`find`].
pub fn find_topk(
    template: &CorrTemplate,
    scene: &ImageView<'_, u8>,
    k: usize,
    cfg: &CorrConfig,
) -> Result<Vec<CorrMatch>, Error> {
    let match_cfg = build_match_config(cfg, template.rotation)?;
    let buf = scene_buf(scene);
    let view = corr_view(buf.as_slice(), scene.width(), scene.height())?;
    let matcher = corrmatch::Matcher::from_arc(template.compiled.clone()).with_config(match_cfg);
    let ms = matcher.match_image_topk(view, k).map_err(map_corr_err)?;
    Ok(ms
        .into_iter()
        .map(|m| to_corr_match(m, template.width, template.height))
        .collect())
}
