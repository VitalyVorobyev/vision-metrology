//! Python-visible [`ShapeModelConfig`]/[`ShapeSearchConfig`] mirrors.

use core::num::NonZeroUsize;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_metrology::matching::Contrast as NativeContrast;
use vision_metrology::matching::Polarity as NativePolarity;
use vision_metrology::matching::Refinement as NativeRefinement;
use vision_metrology::matching::ShapeModelConfig as NativeShapeModelConfig;
use vision_metrology::matching::ShapeSearchConfig as NativeShapeSearchConfig;
use vision_metrology::matching::ShapeSearchTuning as NativeShapeSearchTuning;
use vm_primitives::PreSmooth as NativePreSmooth;
use vm_primitives::{Point2f, Rect2f};

use super::edge::EdgeConfig;

/// Mirrors `vision_metrology::matching::Contrast`.
///
/// A bare `float` cannot stand in for this: which unit it means (Scharr
/// response on the raw pixel scale, or a fraction of the image's own dynamic
/// range) is exactly the ambiguity the Rust type exists to remove. Construct
/// with the two static methods; `min_contrast` parameters accept either.
#[pyclass(frozen, eq, from_py_object)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contrast {
    raw: Option<f32>,
    fraction: Option<f32>,
}

#[pymethods]
impl Contrast {
    /// Absolute Scharr response units on the input pixel scale.
    #[staticmethod]
    pub fn raw(value: f32) -> Self {
        Self {
            raw: Some(value),
            fraction: None,
        }
    }

    /// A fraction of the response an ideal step across the image's full
    /// dynamic range would produce; transfers unchanged between `uint8`,
    /// `uint16` and `float32` images of the same physical contrast.
    #[staticmethod]
    pub fn fraction_of_range(value: f32) -> Self {
        Self {
            raw: None,
            fraction: Some(value),
        }
    }

    fn __repr__(&self) -> String {
        match (self.raw, self.fraction) {
            (Some(v), _) => format!("Contrast.raw({v:.4})"),
            (_, Some(f)) => format!("Contrast.fraction_of_range({f:.4})"),
            _ => unreachable!("constructed only via the two static methods"),
        }
    }
}

impl Contrast {
    pub fn to_native(self) -> NativeContrast {
        match (self.raw, self.fraction) {
            (Some(v), _) => NativeContrast::Raw(v),
            (_, Some(f)) => NativeContrast::FractionOfRange(f),
            _ => unreachable!("constructed only via the two static methods"),
        }
    }

    fn from_native(c: NativeContrast) -> Self {
        match c {
            NativeContrast::Raw(v) => Self::raw(v),
            NativeContrast::FractionOfRange(f) => Self::fraction_of_range(f),
        }
    }
}

fn to_rect(roi: (f32, f32, f32, f32)) -> Rect2f {
    Rect2f {
        x: roi.0,
        y: roi.1,
        width: roi.2,
        height: roi.3,
    }
}

fn parse_polarity(s: &str) -> PyResult<NativePolarity> {
    match s {
        "match" => Ok(NativePolarity::Match),
        "ignore_global" => Ok(NativePolarity::IgnoreGlobal),
        "ignore_local" => Ok(NativePolarity::IgnoreLocal),
        other => Err(PyValueError::new_err(format!(
            "polarity must be 'match', 'ignore_global' or 'ignore_local', got '{other}'"
        ))),
    }
}

/// Mirrors `vision_metrology::matching::ShapeModelConfig`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct ShapeModelConfig {
    /// `None` selects the level count automatically.
    pub num_levels: Option<usize>,
    /// Edge detector configuration used at every pyramid level.
    pub edge: EdgeConfig,
    /// "none" or "binomial121".
    pub pre_smooth: String,
    pub min_contrast: Contrast,
    /// `None` keeps every point.
    pub max_points: Option<usize>,
    /// Reference point in reference-image coordinates. `None` = level-0 centroid.
    pub origin: Option<(f32, f32)>,
    /// The part's natural 0 degree direction in the reference image, radians.
    /// `0.0` keeps the model frame and the reference image's axes identical.
    pub reference_angle: f32,
    pub angle_min: f32,
    pub angle_max: f32,
    pub scale_min: f32,
    pub scale_max: f32,
    /// "match", "ignore_global" or "ignore_local".
    pub polarity: String,
    pub min_points_per_level: usize,
}

impl Default for ShapeModelConfig {
    fn default() -> Self {
        let n = NativeShapeModelConfig::default();
        Self {
            num_levels: n.num_levels.map(NonZeroUsize::get),
            edge: EdgeConfig::default(),
            pre_smooth: match n.pre_smooth {
                NativePreSmooth::None => "none".to_string(),
                NativePreSmooth::Binomial121 => "binomial121".to_string(),
            },
            min_contrast: Contrast::from_native(n.min_contrast),
            max_points: n.max_points.map(NonZeroUsize::get),
            origin: n.origin.map(|p| (p.x, p.y)),
            reference_angle: n.reference_angle,
            angle_min: n.angle_range.0,
            angle_max: n.angle_range.1,
            scale_min: n.scale_range.0,
            scale_max: n.scale_range.1,
            polarity: "match".to_string(),
            min_points_per_level: n.min_points_per_level,
        }
    }
}

#[pymethods]
impl ShapeModelConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        num_levels=None,
        edge=None,
        pre_smooth=None,
        min_contrast=None,
        max_points=None,
        origin=None,
        reference_angle=None,
        angle_min=None,
        angle_max=None,
        scale_min=None,
        scale_max=None,
        polarity=None,
        min_points_per_level=None
    ))]
    pub fn new(
        num_levels: Option<usize>,
        edge: Option<EdgeConfig>,
        pre_smooth: Option<String>,
        min_contrast: Option<Contrast>,
        max_points: Option<usize>,
        origin: Option<(f32, f32)>,
        reference_angle: Option<f32>,
        angle_min: Option<f32>,
        angle_max: Option<f32>,
        scale_min: Option<f32>,
        scale_max: Option<f32>,
        polarity: Option<String>,
        min_points_per_level: Option<usize>,
    ) -> Self {
        let d = Self::default();
        Self {
            num_levels: num_levels.or(d.num_levels),
            edge: edge.unwrap_or(d.edge),
            pre_smooth: pre_smooth.unwrap_or(d.pre_smooth),
            min_contrast: min_contrast.unwrap_or(d.min_contrast),
            max_points: max_points.or(d.max_points),
            origin: origin.or(d.origin),
            reference_angle: reference_angle.unwrap_or(d.reference_angle),
            angle_min: angle_min.unwrap_or(d.angle_min),
            angle_max: angle_max.unwrap_or(d.angle_max),
            scale_min: scale_min.unwrap_or(d.scale_min),
            scale_max: scale_max.unwrap_or(d.scale_max),
            polarity: polarity.unwrap_or(d.polarity),
            min_points_per_level: min_points_per_level.unwrap_or(d.min_points_per_level),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapeModelConfig(num_levels={:?}, max_points={:?}, angle=({:.3},{:.3}), scale=({:.3},{:.3}), polarity='{}')",
            self.num_levels,
            self.max_points,
            self.angle_min,
            self.angle_max,
            self.scale_min,
            self.scale_max,
            self.polarity
        )
    }
}

impl ShapeModelConfig {
    pub fn to_native(&self) -> PyResult<NativeShapeModelConfig> {
        if self.scale_min <= 0.0 || self.scale_max < self.scale_min {
            return Err(PyValueError::new_err(
                "scale_min must be > 0 and scale_max >= scale_min",
            ));
        }
        if self.angle_max < self.angle_min {
            return Err(PyValueError::new_err("angle_max must be >= angle_min"));
        }
        let pre_smooth = match self.pre_smooth.as_str() {
            "none" => NativePreSmooth::None,
            "binomial121" => NativePreSmooth::Binomial121,
            other => {
                return Err(PyValueError::new_err(format!(
                    "pre_smooth must be 'none' or 'binomial121', got {other:?}"
                )));
            }
        };
        Ok(NativeShapeModelConfig {
            num_levels: self.num_levels.and_then(NonZeroUsize::new),
            edge: self.edge.to_native()?,
            pre_smooth,
            min_contrast: self.min_contrast.to_native(),
            max_points: self.max_points.and_then(NonZeroUsize::new),
            origin: self.origin.map(|(x, y)| Point2f::new(x, y)),
            reference_angle: self.reference_angle,
            angle_range: (self.angle_min, self.angle_max),
            scale_range: (self.scale_min, self.scale_max),
            polarity: parse_polarity(&self.polarity)?,
            min_points_per_level: self.min_points_per_level,
        })
    }
}

/// Mirrors `vision_metrology::matching::ShapeSearchTuning` — search *effort*,
/// as opposed to `ShapeSearchConfig`'s *what to look for*.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct ShapeSearchTuning {
    pub greediness: f32,
    /// `None` derives the level-0 angle step from the model radius.
    pub angle_step: Option<f32>,
    /// `None` derives the level-0 scale step from the model radius.
    pub scale_step: Option<f32>,
    pub last_level: usize,
    pub max_candidates: usize,
    pub coarse_score_factor: f32,
}

impl Default for ShapeSearchTuning {
    fn default() -> Self {
        let t = NativeShapeSearchTuning::default();
        Self {
            greediness: t.greediness,
            angle_step: t.angle_step,
            scale_step: t.scale_step,
            last_level: t.last_level,
            max_candidates: t.max_candidates,
            coarse_score_factor: t.coarse_score_factor,
        }
    }
}

#[pymethods]
impl ShapeSearchTuning {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        greediness=None,
        angle_step=None,
        scale_step=None,
        last_level=None,
        max_candidates=None,
        coarse_score_factor=None
    ))]
    pub fn new(
        greediness: Option<f32>,
        angle_step: Option<f32>,
        scale_step: Option<f32>,
        last_level: Option<usize>,
        max_candidates: Option<usize>,
        coarse_score_factor: Option<f32>,
    ) -> Self {
        let d = Self::default();
        Self {
            greediness: greediness.unwrap_or(d.greediness),
            angle_step: angle_step.or(d.angle_step),
            scale_step: scale_step.or(d.scale_step),
            last_level: last_level.unwrap_or(d.last_level),
            max_candidates: max_candidates.unwrap_or(d.max_candidates),
            coarse_score_factor: coarse_score_factor.unwrap_or(d.coarse_score_factor),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapeSearchTuning(greediness={:.2}, last_level={}, max_candidates={})",
            self.greediness, self.last_level, self.max_candidates
        )
    }
}

impl ShapeSearchTuning {
    pub fn to_native(&self) -> NativeShapeSearchTuning {
        NativeShapeSearchTuning {
            greediness: self.greediness,
            angle_step: self.angle_step,
            scale_step: self.scale_step,
            last_level: self.last_level,
            max_candidates: self.max_candidates,
            coarse_score_factor: self.coarse_score_factor,
        }
    }
}

/// Mirrors `vision_metrology::matching::ShapeSearchConfig`.
///
/// The top-level fields say **what** is being looked for; `tuning` says
/// **how hard** the search works — the same split as the Rust type.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone)]
pub struct ShapeSearchConfig {
    pub min_score: f32,
    /// `None` reports every instance.
    pub max_matches: Option<usize>,
    pub max_overlap: f32,
    /// Region, in level-0 image coordinates, as `(x, y, width, height)`.
    /// `None` searches the whole image.
    pub roi: Option<(f32, f32, f32, f32)>,
    /// Rotation range to sweep, in radians. `None` uses the model's own range.
    pub angle_range: Option<(f32, f32)>,
    /// Scale range to sweep. `None` uses the model's own range.
    pub scale_range: Option<(f32, f32)>,
    pub min_contrast: Contrast,
    /// "none", "interpolate" or "least_squares".
    pub refinement: String,
    pub tuning: ShapeSearchTuning,
}

impl Default for ShapeSearchConfig {
    fn default() -> Self {
        let n = NativeShapeSearchConfig::default();
        Self {
            min_score: n.min_score,
            max_matches: n.max_matches.map(NonZeroUsize::get),
            max_overlap: n.max_overlap,
            roi: n.roi.map(|r| (r.x, r.y, r.width, r.height)),
            angle_range: n.angle_range,
            scale_range: n.scale_range,
            min_contrast: Contrast::from_native(n.min_contrast),
            refinement: "interpolate".to_string(),
            tuning: ShapeSearchTuning::default(),
        }
    }
}

#[pymethods]
impl ShapeSearchConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        min_score=None,
        max_matches=None,
        max_overlap=None,
        roi=None,
        angle_range=None,
        scale_range=None,
        min_contrast=None,
        refinement=None,
        tuning=None
    ))]
    pub fn new(
        min_score: Option<f32>,
        max_matches: Option<usize>,
        max_overlap: Option<f32>,
        roi: Option<(f32, f32, f32, f32)>,
        angle_range: Option<(f32, f32)>,
        scale_range: Option<(f32, f32)>,
        min_contrast: Option<Contrast>,
        refinement: Option<String>,
        tuning: Option<ShapeSearchTuning>,
    ) -> Self {
        let d = Self::default();
        Self {
            min_score: min_score.unwrap_or(d.min_score),
            max_matches: max_matches.or(d.max_matches),
            max_overlap: max_overlap.unwrap_or(d.max_overlap),
            roi: roi.or(d.roi),
            angle_range: angle_range.or(d.angle_range),
            scale_range: scale_range.or(d.scale_range),
            min_contrast: min_contrast.unwrap_or(d.min_contrast),
            refinement: refinement.unwrap_or(d.refinement),
            tuning: tuning.unwrap_or(d.tuning),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapeSearchConfig(min_score={:.3}, max_matches={:?}, refinement='{}', tuning={})",
            self.min_score,
            self.max_matches,
            self.refinement,
            self.tuning.__repr__()
        )
    }
}

impl ShapeSearchConfig {
    pub fn to_native(&self) -> PyResult<NativeShapeSearchConfig> {
        if !(0.0..=1.0).contains(&self.min_score) {
            return Err(PyValueError::new_err("min_score must be in [0, 1]"));
        }
        if !(0.0..=1.0).contains(&self.tuning.greediness) {
            return Err(PyValueError::new_err("tuning.greediness must be in [0, 1]"));
        }
        let refinement = match self.refinement.as_str() {
            "none" => NativeRefinement::None,
            "interpolate" => NativeRefinement::Interpolate,
            "least_squares" => NativeRefinement::LeastSquares,
            other => {
                return Err(PyValueError::new_err(format!(
                    "refinement must be 'none', 'interpolate' or 'least_squares', got '{other}'"
                )));
            }
        };
        Ok(NativeShapeSearchConfig {
            min_score: self.min_score,
            max_matches: self.max_matches.and_then(NonZeroUsize::new),
            max_overlap: self.max_overlap,
            roi: self.roi.map(to_rect),
            angle_range: self.angle_range,
            scale_range: self.scale_range,
            min_contrast: self.min_contrast.to_native(),
            refinement,
            tuning: self.tuning.to_native(),
        })
    }
}
