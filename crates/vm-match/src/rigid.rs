//! Configuration for rigid (rotation + translation) edge matching.

use vm_core::Rect2f;

/// Configuration for rigid (R + t) directed-edge matching.
///
/// The coarse grid search sweeps `angle_range` in steps of `angle_step`,
/// and `position_search` in steps of `chamfer_threshold / 2` on each axis.
/// The top-`top_k` coarse candidates by chamfer score are refined with ICP
/// and normal-coherence scoring before selecting the best result.
///
/// ## Resolution factor (OQ-3)
/// Full resolution only for now. The `resolution_factor` field is present
/// for forward compatibility but must be set to `1.0`; values other than
/// `1.0` are treated as `1.0` in the current implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct RigidMatchConfig {
    /// Search range in radians, e.g. `(-PI, PI)` for unconstrained rotation.
    pub angle_range: (f32, f32),
    /// Angular step size in radians. Default: 0.01 rad ≈ 0.57°.
    pub angle_step: f32,
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

impl Default for RigidMatchConfig {
    fn default() -> Self {
        Self {
            angle_range: (-core::f32::consts::PI, core::f32::consts::PI),
            angle_step: 0.01,
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
