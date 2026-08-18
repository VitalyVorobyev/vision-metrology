//! Native Python return types for the `vision_metrology` module.

use pyo3::prelude::*;

/// A single 2-D subpixel edge detection result.
///
/// Positions follow pixel-center convention: integer coordinate `i` is the center of pixel `i`.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct Edgel {
    /// Subpixel x coordinate (pixel-center convention).
    pub x: f32,
    /// Subpixel y coordinate (pixel-center convention).
    pub y: f32,
    /// X component of the unit outward normal (dark-to-bright direction).
    pub nx: f32,
    /// Y component of the unit outward normal (dark-to-bright direction).
    pub ny: f32,
    /// Edge response strength after non-maximum suppression.
    pub strength: f32,
}

#[pymethods]
impl Edgel {
    fn __repr__(&self) -> String {
        format!(
            "Edgel(x={:.3}, y={:.3}, nx={:.3}, ny={:.3}, strength={:.3})",
            self.x, self.y, self.nx, self.ny, self.strength
        )
    }
}

/// A detected line segment with subpixel endpoints.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct LineSegment {
    /// X coordinate of the first endpoint (pixel-center convention).
    pub x1: f32,
    /// Y coordinate of the first endpoint.
    pub y1: f32,
    /// X coordinate of the second endpoint.
    pub x2: f32,
    /// Y coordinate of the second endpoint.
    pub y2: f32,
    /// Estimated support-region width in pixels.
    pub width: f32,
    /// log10(NFA): negative values indicate significant detections.
    pub nfa: f32,
    /// Orientation angle in radians.
    pub angle: f32,
    /// Euclidean length of the segment in pixels.
    pub length: f32,
}

#[pymethods]
impl LineSegment {
    fn __repr__(&self) -> String {
        format!(
            "LineSegment(({:.1},{:.1})->({:.1},{:.1}), len={:.1}, angle={:.3})",
            self.x1, self.y1, self.x2, self.y2, self.length, self.angle
        )
    }
}

/// A fitted ellipse in geometric form.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct Ellipse {
    /// X coordinate of the ellipse center (pixel-center convention).
    pub cx: f32,
    /// Y coordinate of the ellipse center.
    pub cy: f32,
    /// Semi-major axis length in pixels.
    pub a: f32,
    /// Semi-minor axis length in pixels.
    pub b: f32,
    /// Rotation angle of the major axis in radians.
    pub angle: f32,
}

#[pymethods]
impl Ellipse {
    fn __repr__(&self) -> String {
        format!(
            "Ellipse(cx={:.2}, cy={:.2}, a={:.2}, b={:.2}, angle={:.4})",
            self.cx, self.cy, self.a, self.b, self.angle
        )
    }
}

/// Result of an edge-model match operation.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct MatchResult {
    /// X translation (pixels) that maps model-local to scene coordinates.
    pub tx: f32,
    /// Y translation (pixels).
    pub ty: f32,
    /// Rotation angle in radians.
    pub angle: f32,
    /// Uniform scale factor (1.0 for rigid matches).
    pub scale: f32,
    /// Inlier fraction in [0, 1]: proportion of model edgels within chamfer threshold.
    pub score: f32,
    /// Absolute number of inlier edgels.
    pub inlier_count: usize,
}

#[pymethods]
impl MatchResult {
    fn __repr__(&self) -> String {
        format!(
            "MatchResult(tx={:.2}, ty={:.2}, angle={:.4}, scale={:.4}, score={:.3}, inliers={})",
            self.tx, self.ty, self.angle, self.scale, self.score, self.inlier_count
        )
    }
}

/// Per-connected-component statistics from CCL segmentation.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct ComponentStats {
    /// Component label (1-based).
    pub label: u32,
    /// Number of foreground pixels in this component.
    pub pixel_count: u32,
    /// Centroid X coordinate (pixel-center convention).
    pub cx: f32,
    /// Centroid Y coordinate.
    pub cy: f32,
    /// Bounding box top-left X.
    pub bbox_x: f32,
    /// Bounding box top-left Y.
    pub bbox_y: f32,
    /// Bounding box width in pixels.
    pub bbox_w: f32,
    /// Bounding box height in pixels.
    pub bbox_h: f32,
}

#[pymethods]
impl ComponentStats {
    fn __repr__(&self) -> String {
        format!(
            "ComponentStats(label={}, pixels={}, cx={:.1}, cy={:.1})",
            self.label, self.pixel_count, self.cx, self.cy
        )
    }
}
