//! Configuration for rigid and similarity (rotation + uniform scale + translation)
//! edge matching.

use vm_core::Rect2f;

/// Unified configuration for directed-edge matching.
///
/// Supports two transform families depending on the `scale_range` and
/// `scale_step` settings:
/// - **Rigid (R|t):** set `scale_range = (1.0, 1.0)` and `scale_step = 0.0`
///   (the default). The outer scale loop runs a single iteration with scale 1.0
///   and adds no runtime overhead.
/// - **Similarity (sR|t):** set `scale_range = (s_min, s_max)` and
///   `scale_step > 0.0`. The grid search sweeps angle × scale × position.
///
/// ## Grid search dimensions
/// - Angles: `angle_range` in steps of `angle_step`.
/// - Scales: `scale_range` in steps of `scale_step` (single step when rigid).
/// - Positions: `position_search` in steps of `chamfer_threshold / 2` on each axis.
///
/// ## Resolution factor (OQ-3)
/// Full resolution only for now. The `resolution_factor` field is present
/// for forward compatibility but must be set to `1.0`; values other than
/// `1.0` are treated as `1.0` in the current implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct MatchConfig {
    /// Search range in radians, e.g. `(-PI, PI)` for unconstrained rotation.
    pub angle_range: (f32, f32),
    /// Angular step size in radians. Default: 0.01 rad ≈ 0.57°.
    pub angle_step: f32,
    /// Uniform-scale search range. Default `(1.0, 1.0)` disables scale search.
    ///
    /// Both components must be positive. The range is interpreted as
    /// `[scale_range.0, scale_range.1]` swept in steps of `scale_step`.
    pub scale_range: (f32, f32),
    /// Uniform-scale step. Default `0.0` disables scale search (single step
    /// at `scale_range.0`). Ignored when `scale_range == (1.0, 1.0)`.
    pub scale_step: f32,
    /// Image-space search region for the model centroid.
    pub position_search: Rect2f,
    /// Chamfer distance threshold in pixels: edgels closer than this count
    /// as inliers. Also controls the coarse translation step (`threshold / 2`).
    pub chamfer_threshold: f32,
    /// Minimum fraction of inlier edgels required to accept a match. Range [0, 1].
    pub min_score: f32,
    /// If `true`, run ICP refinement on each of the top-K coarse candidates
    /// before computing the final score.
    pub refine_icp: bool,
    /// Number of coarse candidates (by chamfer score) kept for ICP refinement.
    pub top_k: usize,
    /// Reserved for future sub-resolution map support (OQ-3). Keep at `1.0`.
    pub resolution_factor: f32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        Self {
            angle_range: (-core::f32::consts::PI, core::f32::consts::PI),
            angle_step: 0.01,
            scale_range: (1.0, 1.0),
            scale_step: 0.0,
            position_search: Rect2f {
                x: 0.0,
                y: 0.0,
                width: 640.0,
                height: 480.0,
            },
            chamfer_threshold: 3.0,
            min_score: 0.5,
            refine_icp: true,
            top_k: 5,
            resolution_factor: 1.0,
        }
    }
}

/// Backward-compatible alias for rigid (no scale) matching configuration.
///
/// Identical to [`MatchConfig`] with `scale_range = (1.0, 1.0)` and
/// `scale_step = 0.0` (the default). All existing code using
/// `RigidMatchConfig` continues to compile unchanged.
pub type RigidMatchConfig = MatchConfig;

impl MatchConfig {
    /// Returns `true` if this config performs a rigid (no scale) search.
    ///
    /// A config is considered rigid when `scale_range.0 == scale_range.1` or
    /// `scale_step == 0.0`.
    #[inline]
    pub fn is_rigid(&self) -> bool {
        (self.scale_range.0 - self.scale_range.1).abs() < 1e-7 || self.scale_step <= 0.0
    }

    /// Iterate over scale values in the search grid.
    ///
    /// Yields exactly one value (`scale_range.0`) when the config is rigid.
    pub fn scale_iter(&self) -> impl Iterator<Item = f32> + use<'_> {
        let s_min = self.scale_range.0.min(self.scale_range.1);
        let s_max = self.scale_range.0.max(self.scale_range.1);
        let n = if self.is_rigid() {
            1usize
        } else {
            let step = self.scale_step.max(1e-6);
            // Round to avoid floating-point overshoot: use saturating integer count.
            ((s_max - s_min) / step).round() as usize + 1
        };
        let step = if self.is_rigid() {
            0.0
        } else {
            self.scale_step.max(1e-6)
        };
        (0..n).map(move |i| {
            if self.is_rigid() {
                self.scale_range.0
            } else {
                (s_min + i as f32 * step).min(s_max)
            }
        })
    }
}
