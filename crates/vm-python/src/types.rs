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
    /// RMS orthogonal deviation of the fitted points, in pixels.
    pub rms: f32,
    /// Largest orthogonal deviation, in pixels. Read this for form tolerance.
    pub max_dev: f32,
    /// How many input points the fit used.
    pub n_used: usize,
}

/// A fitted circle with the residual statistics that qualify it.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct Circle {
    /// X coordinate of the centre (pixel-centre convention).
    pub cx: f32,
    /// Y coordinate of the centre.
    pub cy: f32,
    /// Radius in pixels.
    pub r: f32,
    /// RMS radial deviation of the fitted points, in pixels.
    pub rms: f32,
    /// Largest radial deviation, in pixels. Read this for roundness.
    pub max_dev: f32,
    /// How many input points the fit used.
    pub n_used: usize,
}

#[pymethods]
impl Circle {
    fn __repr__(&self) -> String {
        format!(
            "Circle(cx={:.3}, cy={:.3}, r={:.3}, rms={:.4}, max_dev={:.4}, n_used={})",
            self.cx, self.cy, self.r, self.rms, self.max_dev, self.n_used
        )
    }
}

#[pymethods]
impl Ellipse {
    fn __repr__(&self) -> String {
        format!(
            "Ellipse(cx={:.3}, cy={:.3}, a={:.3}, b={:.3}, angle={:.4}, rms={:.4}, max_dev={:.4}, n_used={})",
            self.cx, self.cy, self.a, self.b, self.angle, self.rms, self.max_dev, self.n_used
        )
    }
}

/// One located instance of a :class:`ShapeModel`.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone)]
pub struct ShapeMatch {
    /// X coordinate, in scene pixels, where the model origin landed.
    pub x: f32,
    /// Y coordinate, in scene pixels, where the model origin landed.
    pub y: f32,
    /// Rotation in radians, wrapped to (-pi, pi].
    pub angle: f32,
    /// Uniform scale factor.
    pub scale: f32,
    /// Similarity score in [0, 1]; roughly `1 - occluded_fraction`.
    pub score: f32,
    /// Model points that found any gradient at all.
    pub support: usize,
    /// Pyramid level the reported score was evaluated at.
    pub level: usize,
}

impl From<vision_metrology::matching::ShapeMatch> for ShapeMatch {
    fn from(m: vision_metrology::matching::ShapeMatch) -> Self {
        Self {
            x: m.position.x,
            y: m.position.y,
            angle: m.angle(),
            scale: m.scale(),
            score: m.score,
            support: m.support,
            level: m.level,
        }
    }
}

#[pymethods]
impl ShapeMatch {
    /// The 2x3 similarity `[[a, b, tx], [c, d, ty]]` mapping reference-image
    /// coordinates to scene coordinates, as nested lists.
    ///
    /// Reconstructed from the reported pose parts, so it already includes the
    /// model origin: `scene = M @ [x, y, 1]`.
    pub fn matrix(&self, origin: (f32, f32)) -> [[f32; 3]; 2] {
        let (sn, cs) = self.angle.sin_cos();
        let (a, b) = (self.scale * cs, -self.scale * sn);
        let (c, d) = (self.scale * sn, self.scale * cs);
        [
            [a, b, self.x - (a * origin.0 + b * origin.1)],
            [c, d, self.y - (c * origin.0 + d * origin.1)],
        ]
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapeMatch(x={:.2}, y={:.2}, angle={:.4}, scale={:.4}, score={:.3}, support={})",
            self.x, self.y, self.angle, self.scale, self.score, self.support
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
