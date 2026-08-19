//! Python bindings for shape-based object detection.

use pyo3::prelude::*;
use pyo3::types::PyList;
use vision_metrology::matching::{
    ShapeMatcher as NativeShapeMatcher, ShapeModel as NativeShapeModel, ShapeModelBuilder,
};
use vm_primitives::Rect2f;

use crate::config::{ShapeModelConfig, ShapeSearchConfig};
use crate::convert::{any_image_from_numpy, with_any_image};
use crate::types::ShapeMatch;

fn to_rect(roi: (f32, f32, f32, f32)) -> Rect2f {
    Rect2f {
        x: roi.0,
        y: roi.1,
        width: roi.2,
        height: roi.3,
    }
}

/// A rotation- and scale-searchable model of an object's edge geometry.
///
/// Build it from a reference image and a rectangular ROI, then hand it to
/// :class:`ShapeMatcher`.
#[pyclass]
pub struct ShapeModel {
    inner: NativeShapeModel,
}

#[pymethods]
impl ShapeModel {
    /// Build from a `(H, W)` `uint8`/`uint16`/`float32` array and an
    /// `(x, y, width, height)` ROI.
    #[new]
    #[pyo3(signature = (image, roi, config=None))]
    pub fn new(
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        roi: (f32, f32, f32, f32),
        config: Option<ShapeModelConfig>,
    ) -> PyResult<Self> {
        let cfg = config.unwrap_or_default().to_native()?;
        let any = any_image_from_numpy(py, image)?;
        let inner = with_any_image!(any, view => {
            ShapeModelBuilder::new().build(&view, to_rect(roi), &cfg)
        })
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of pyramid levels.
    #[getter]
    pub fn num_levels(&self) -> usize {
        self.inner.num_levels()
    }

    /// Reference point in reference-image coordinates, as `(x, y)`.
    #[getter]
    pub fn origin(&self) -> (f32, f32) {
        let o = self.inner.origin();
        (o.x, o.y)
    }

    /// Points per pyramid level, finest first.
    #[getter]
    pub fn point_counts(&self) -> Vec<usize> {
        self.inner
            .levels()
            .iter()
            .map(|l| l.points().len())
            .collect()
    }

    /// Level-0 model points in reference-image coordinates, as a list of `(x, y)`.
    pub fn reference_points(&self) -> Vec<(f32, f32)> {
        self.inner
            .reference_points()
            .into_iter()
            .map(|p| (p.x, p.y))
            .collect()
    }

    /// Persist the model to a file.
    ///
    /// The encoding is opaque and versioned; :meth:`load` refuses a document
    /// written by an incompatible format instead of mis-reading it.
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Load a model previously written by :meth:`save`.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner = NativeShapeModel::load(path)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapeModel(levels={}, points={:?}, origin=({:.1}, {:.1}))",
            self.num_levels(),
            self.point_counts(),
            self.inner.origin().x,
            self.inner.origin().y
        )
    }
}

/// Reusable coarse-to-fine shape matcher.
#[pyclass]
pub struct ShapeMatcher {
    matcher: NativeShapeMatcher,
    cfg: ShapeSearchConfig,
}

#[pymethods]
impl ShapeMatcher {
    /// Create a matcher, optionally with an explicit search config.
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<ShapeSearchConfig>) -> Self {
        Self {
            matcher: NativeShapeMatcher::new(),
            cfg: config.unwrap_or_default(),
        }
    }

    /// Locate `model` in a `(H, W)` `uint8`/`uint16`/`float32` image; returns
    /// a list of `ShapeMatch`.
    pub fn find<'py>(
        &mut self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
        model: &ShapeModel,
    ) -> PyResult<Bound<'py, PyList>> {
        let native = self.cfg.to_native()?;
        let any = any_image_from_numpy(py, image)?;
        let out = with_any_image!(any, view => self.matcher.find(&view, &model.inner, &native));
        PyList::new(py, out.into_iter().map(ShapeMatch::from))
    }

    /// Whether the last search discarded candidates at `max_candidates`.
    ///
    /// A truncated search can miss the true instance, which distinguishes
    /// "not present" from "gave up".
    #[getter]
    pub fn truncated(&self) -> bool {
        self.matcher.truncated()
    }

    /// Current search configuration.
    #[getter]
    pub fn config(&self) -> ShapeSearchConfig {
        self.cfg.clone()
    }

    #[setter]
    pub fn set_config(&mut self, config: ShapeSearchConfig) {
        self.cfg = config;
    }
}

/// One-shot convenience: build a model and search a scene in a single call.
#[pyfunction]
#[pyo3(signature = (model_image, roi, scene_image, model_config=None, search_config=None))]
pub fn find_shape_model<'py>(
    py: Python<'py>,
    model_image: &Bound<'py, PyAny>,
    roi: (f32, f32, f32, f32),
    scene_image: &Bound<'py, PyAny>,
    model_config: Option<ShapeModelConfig>,
    search_config: Option<ShapeSearchConfig>,
) -> PyResult<Bound<'py, PyList>> {
    let model = ShapeModel::new(py, model_image, roi, model_config)?;
    let mut matcher = ShapeMatcher::new(search_config);
    matcher.find(py, scene_image, &model)
}
