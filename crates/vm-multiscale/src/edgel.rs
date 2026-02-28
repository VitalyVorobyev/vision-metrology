//! Scale-annotated edgel type produced by [`MultiScaleEdgeDetector`].
//!
//! [`MultiScaleEdgeDetector`]: crate::MultiScaleEdgeDetector

use vm_core::{Point2f, Vec2f};

/// A 2-D edgel detected at a specific pyramid level, with its position mapped
/// back to level-0 (original-resolution) pixel coordinates.
///
/// All spatial fields (`p`, `idx`) are expressed in **level-0** coordinates so
/// that edgels from different scales can be compared directly.
///
/// The `scale` and `level` fields record at which pyramid level the edgel was
/// found, allowing downstream algorithms (e.g. LSD, ellipse fitting) to weight
/// or filter by scale.
#[derive(Debug, Clone, PartialEq)]
pub struct ScaleAnnotatedEdgel {
    /// Subpixel position in level-0 pixel coordinates.
    pub p: Point2f,
    /// Unit normal (dark-to-bright direction).
    pub n: Vec2f,
    /// Gradient magnitude at the detection level (before coordinate mapping).
    pub strength: f32,
    /// Effective Gaussian sigma at the detection level: `base_sigma * 2^level`.
    pub scale: f32,
    /// Pyramid level at which this edgel was detected (0 = original resolution).
    pub level: usize,
    /// Level-0 integer grid cell: `(idx.0 * 2^level, idx.1 * 2^level)`.
    pub idx: (usize, usize),
}
