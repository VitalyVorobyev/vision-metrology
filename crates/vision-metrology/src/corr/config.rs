//! Configuration for `corr`: template compilation, search, and displacement.

use core::num::NonZeroUsize;

use vm_primitives::Rect2f;

/// Compile-time assets for one reference patch.
///
/// Mirrors corrmatch's `CompileConfig` / `CompileConfigNoRot` split: whether
/// to build a per-level angle bank of masked ZNCC/SSD plans is decided once,
/// here, because it changes what the compiled assets *contain* — a search
/// later cannot ask for rotation a template was not compiled with (see
/// [`CorrConfig::rotation`]).
#[derive(Debug, Clone, PartialEq)]
pub struct CorrTemplateConfig {
    /// Compile a per-level angle bank so a later search can enable rotation.
    ///
    /// Costs more at compile time (masked plans for every angle at the
    /// coarsest level, by default — see
    /// [`CorrTemplateTuning::precompute_coarsest`]). Without it,
    /// [`CorrConfig::rotation = true`](CorrConfig::rotation) fails with
    /// [`Error::InvalidConfig`](vm_primitives::Error::InvalidConfig).
    pub rotation: bool,
    /// Pyramid levels to build. `None` uses corrmatch's own default (6),
    /// trimmed automatically once a level gets too small to be useful.
    pub max_levels: Option<NonZeroUsize>,
    /// Advanced compile-time knobs, only used when `rotation` is `true`.
    pub tuning: CorrTemplateTuning,
}

impl Default for CorrTemplateConfig {
    fn default() -> Self {
        Self {
            rotation: true,
            max_levels: None,
            tuning: CorrTemplateTuning::default(),
        }
    }
}

/// Advanced [`CorrTemplateConfig`] knobs — corrmatch's rotated-compile
/// parameters, 1:1. A working setup rarely touches these.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CorrTemplateTuning {
    /// Coarse rotation step in degrees at the coarsest pyramid level.
    pub coarse_angle_step_deg: f32,
    /// Minimum rotation step in degrees, a floor across every level.
    pub min_angle_step_deg: f32,
    /// Fill value for pixels a template rotation exposes outside the
    /// original patch.
    pub fill_value: u8,
    /// Precompute every angle at the coarsest level up front, rather than
    /// lazily on first use.
    pub precompute_coarsest: bool,
}

impl Default for CorrTemplateTuning {
    fn default() -> Self {
        Self {
            coarse_angle_step_deg: 10.0,
            min_angle_step_deg: 0.5,
            fill_value: 0,
            precompute_coarsest: true,
        }
    }
}

/// Matching metric.
///
/// `Zncc` (zero-mean normalized cross-correlation, roughly `[-1, 1]`) is
/// robust to illumination offset/gain and is the default. `Ssd` (reported
/// as a negative sum-of-squared-differences, so higher is still better) is
/// cheaper but assumes matched brightness/contrast between template and
/// scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CorrMetric {
    /// Zero-mean normalized cross-correlation.
    #[default]
    Zncc,
    /// Sum of squared differences, reported as its negative.
    Ssd,
}

/// Parameters for one [`find`](super::find) / [`find_topk`](super::find_topk) search.
///
/// The default is translation-only (`rotation: false`, matching corrmatch's
/// own `MatchConfig::default()`) — enable rotation deliberately once the
/// template is compiled to support it.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct CorrConfig {
    /// Search rotation. Requires a [`CorrTemplate`](super::CorrTemplate)
    /// compiled with [`CorrTemplateConfig::rotation`] `= true`.
    pub rotation: bool,
    /// Matching metric.
    pub metric: CorrMetric,
    /// Minimum score to accept. `None` = no floor (report whatever scores
    /// highest).
    pub min_score: Option<f32>,
    /// Search effort: the knobs a working setup rarely touches.
    pub tuning: CorrSearchTuning,
}

/// Advanced [`CorrConfig`] knobs — corrmatch's `MatchConfig` search-effort
/// fields, 1:1. Defaults are corrmatch's own.
#[derive(Debug, Clone, PartialEq)]
pub struct CorrSearchTuning {
    /// Run the coarse-to-fine search in parallel (needs corrmatch's `rayon`
    /// feature; ignored — sequential — when it is not compiled in).
    pub parallel: bool,
    /// Maximum pyramid levels to build for the scene. `None` uses
    /// corrmatch's own default (6).
    pub max_image_levels: Option<NonZeroUsize>,
    /// Beam width kept per level after merge and non-max suppression.
    pub beam_width: usize,
    /// Top-M peaks per angle at the coarsest level. Ignored when `rotation`
    /// is `false`.
    pub per_angle_topk: usize,
    /// Spatial NMS radius in pixels for the current level.
    pub nms_radius: usize,
    /// Refinement ROI radius in pixels for the current level.
    pub roi_radius: usize,
    /// Angle neighborhood half-range, in multiples of the compiled grid
    /// step. Ignored when `rotation` is `false`.
    pub angle_half_range_steps: usize,
    /// Minimum variance for image patches; ignored for the `Ssd` metric.
    pub min_var_i: f32,
}

impl Default for CorrSearchTuning {
    fn default() -> Self {
        Self {
            parallel: false,
            max_image_levels: None,
            beam_width: 8,
            per_angle_topk: 3,
            nms_radius: 6,
            roi_radius: 8,
            angle_half_range_steps: 1,
            min_var_i: 1e-8,
        }
    }
}

/// Subpixel refinement stage for [`displacement`](super::displacement).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refine {
    /// Report corrmatch's own quadratic-peak subpixel estimate directly.
    ///
    /// Fast, but a quadratic fit to a discrete correlation surface is
    /// biased toward integer pixel positions ("pixel-locking") —
    /// quantified by `tests/accuracy.rs`'s `displacement_quadratic` row
    /// against `displacement_lk`.
    None,
    /// Follow stage 1 with translation-only inverse-compositional
    /// Lucas-Kanade, seeded at the stage-1 shift.
    LucasKanade {
        /// Iteration count. The normal-equations Hessian is built once from
        /// the template's own gradient (inverse-compositional), so each
        /// extra iteration costs one bilinear resample of the window, no
        /// more.
        iters: u32,
    },
}

impl Default for Refine {
    /// `LucasKanade { iters: 3 }` — the plan's default iteration count.
    /// Enum derive(Default) requires a unit default variant, which
    /// `LucasKanade`'s payload rules out, hence the manual impl.
    fn default() -> Self {
        Refine::LucasKanade { iters: 3 }
    }
}

/// Parameters for [`displacement`](super::displacement).
#[derive(Debug, Clone, PartialEq)]
pub struct DisplacementConfig {
    /// Window in `prev` to track, level-0 pixel coordinates.
    pub window: Rect2f,
    /// Search radius `(x, y)` in `curr`, in pixels, around the window's
    /// position in `prev`.
    pub search: (i32, i32),
    /// Subpixel refinement stage.
    pub refine: Refine,
    /// Minimum stage-1 (corrmatch ZNCC) score to accept. Below this,
    /// [`displacement`](super::displacement) returns
    /// [`Error::Degenerate`](vm_primitives::Error::Degenerate).
    pub min_score: f32,
}

impl Default for DisplacementConfig {
    fn default() -> Self {
        Self {
            // No sensible default window exists — callers must set one; a
            // zero-sized rect fails validation with a clear error rather
            // than silently tracking an arbitrary 1x1 corner.
            window: Rect2f::default(),
            search: (12, 12),
            refine: Refine::default(),
            min_score: 0.5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_the_documented_ones() {
        let t = CorrTemplateConfig::default();
        assert!(t.rotation);
        assert_eq!(t.max_levels, None);
        assert_eq!(t.tuning.coarse_angle_step_deg, 10.0);
        assert_eq!(t.tuning.min_angle_step_deg, 0.5);

        let c = CorrConfig::default();
        assert!(!c.rotation);
        assert_eq!(c.metric, CorrMetric::Zncc);
        assert_eq!(c.min_score, None);
        assert_eq!(c.tuning.beam_width, 8);

        let d = DisplacementConfig::default();
        assert_eq!(d.search, (12, 12));
        assert_eq!(d.refine, Refine::LucasKanade { iters: 3 });
        assert_eq!(d.min_score, 0.5);
    }
}
