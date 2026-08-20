//! Python bindings for shape-based object detection.

use pyo3::prelude::*;
use pyo3::types::PyList;
use vision_metrology::matching::{
    CropSpec as NativeCropSpec, ShapeMatcher as NativeShapeMatcher, ShapeModel as NativeShapeModel,
    ShapeModelBuilder,
};
use vm_primitives::Rect2f;

use crate::config::{ShapeModelConfig, ShapeSearchConfig};
use crate::convert::{any_image_from_numpy, image_from_numpy_u8, with_any_image};
use crate::types::ShapeMatch;

fn to_rect(roi: (f32, f32, f32, f32)) -> Rect2f {
    Rect2f {
        x: roi.0,
        y: roi.1,
        width: roi.2,
        height: roi.3,
    }
}

/// Fixed crop geometry for :meth:`ShapeMatch.model_frame_map` — rectify a
/// located match into a canonical, model-frame patch.
///
/// `rect` is `(x, y, width, height)` in **model-frame coordinates** (the
/// same reference-image frame :attr:`ShapeModel.reference_points` reports).
/// `px_per_unit` sets the destination resolution; `normalize_scale=True`
/// (default) renders the crop at model scale regardless of the found scale,
/// `False` renders at the found scale. The output pixel size —
/// :attr:`output_size` — depends only on `rect` and `px_per_unit`, never on
/// any particular match: that is what makes crops from different frames
/// directly comparable.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct CropSpec {
    /// Crop rectangle in model-frame coordinates, `(x, y, width, height)`.
    pub rect: (f32, f32, f32, f32),
    /// Destination pixels per one model-frame unit (pixel).
    pub px_per_unit: f32,
    /// Render at model scale (`True`, canonical) or found scale (`False`).
    pub normalize_scale: bool,
}

#[pymethods]
impl CropSpec {
    #[new]
    #[pyo3(signature = (rect, px_per_unit, normalize_scale=true))]
    pub fn new(rect: (f32, f32, f32, f32), px_per_unit: f32, normalize_scale: bool) -> Self {
        Self {
            rect,
            px_per_unit,
            normalize_scale,
        }
    }

    /// Output `(width, height)` in pixels — fixed by this spec alone.
    #[getter]
    pub fn output_size(&self) -> (usize, usize) {
        self.to_native().output_size()
    }

    fn __repr__(&self) -> String {
        format!(
            "CropSpec(rect={:?}, px_per_unit={:.3}, normalize_scale={})",
            self.rect, self.px_per_unit, self.normalize_scale
        )
    }
}

impl CropSpec {
    pub(crate) fn to_native(self) -> NativeCropSpec {
        NativeCropSpec {
            rect: to_rect(self.rect),
            px_per_unit: self.px_per_unit,
            normalize_scale: self.normalize_scale,
        }
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
    ///
    /// `mask`, when given, is a `(H, W)` `uint8` array the same size as
    /// `image`: an edge point enters the model only where the mask is
    /// non-zero. Use it when the part is not rectangular — the ROI's corners
    /// otherwise contribute background edges that no instance can match, and
    /// every one of them dilutes the score, whose denominator is the model's
    /// own point count.
    #[new]
    #[pyo3(signature = (image, roi, config=None, mask=None))]
    pub fn new(
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        roi: (f32, f32, f32, f32),
        config: Option<ShapeModelConfig>,
        mask: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let cfg = config.unwrap_or_default().to_native()?;
        let any = any_image_from_numpy(py, image)?;
        let mask_img = mask
            .map(|m| image_from_numpy_u8(py, &m.extract()?))
            .transpose()?;
        let inner = with_any_image!(any, view => {
            ShapeModelBuilder::new().build_with_mask(
                &view,
                to_rect(roi),
                mask_img.as_ref().map(|m| m.as_view()).as_ref(),
                &cfg,
            )
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

    /// The canonical orientation, in radians, the model frame is rotated onto
    /// relative to the reference image (see `ShapeModelConfig.reference_angle`).
    #[getter]
    pub fn reference_angle(&self) -> f32 {
        self.inner.reference_angle()
    }

    /// Model points at `level` in **model-frame** coordinates, as a list of
    /// `(x, y, dx, dy)` — position plus unit gradient direction.
    ///
    /// This is the frame a match's pose consumes, so these are the points to
    /// transform by :meth:`ShapeMatch.matrix` when drawing a found instance
    /// over a scene. Every level is reported in level-0 units, so the levels
    /// are directly comparable. Empty when `level >= num_levels`.
    pub fn model_geometry(&self, level: usize) -> Vec<(f32, f32, f32, f32)> {
        self.inner
            .model_geometry(level)
            .into_iter()
            .map(|(p, t)| (p.x, p.y, t.x, t.y))
            .collect()
    }

    /// Model points at `level` in **reference-image** coordinates, as a list
    /// of `(x, y, dx, dy)`.
    ///
    /// :meth:`model_geometry` with :attr:`reference_angle` undone — what to
    /// draw over the image the model was taught from. Identical to
    /// `model_geometry` for the usual `reference_angle == 0.0` model. Empty
    /// when `level >= num_levels`.
    pub fn reference_geometry(&self, level: usize) -> Vec<(f32, f32, f32, f32)> {
        self.inner
            .reference_geometry(level)
            .into_iter()
            .map(|(p, t)| (p.x, p.y, t.x, t.y))
            .collect()
    }

    /// The `dst -> src` map that rectifies this model's own reference image
    /// into the same canonical crop :meth:`ShapeMatch.model_frame_map`
    /// produces for a found instance — the untouched half of a side-by-side.
    pub fn reference_frame_map(&self, spec: &CropSpec) -> crate::warp_py::Map {
        crate::warp_py::Map::from_native(self.inner.reference_frame_map(&spec.to_native()))
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

    /// Number of teach-time points :meth:`resample_at` has to work with, or
    /// `0` for a model loaded from a format-3 document (predating that
    /// stored data).
    #[getter]
    pub fn teach_point_count(&self) -> usize {
        self.inner.teach_point_count()
    }

    /// Rebuild this model with every point resampled at scale `s` (roadmap
    /// W7, "estimate-then-verify"). The result's own `scale_range` is a
    /// narrow band around 1.0 — search that, and multiply a found match's
    /// own `scale` by `s` to recover scale relative to *this* model.
    ///
    /// Raises :class:`ValueError` if `s` is not finite and positive, or if
    /// this model has no stored teach data (see :attr:`teach_point_count`).
    pub fn resample_at(&self, s: f32) -> PyResult<Self> {
        let inner = self
            .inner
            .resample_at(s)
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

impl ShapeModel {
    /// Borrow the native model — `scale_py`'s estimators/`find_scale_invariant`
    /// take `&vision_metrology::matching::ShapeModel` directly.
    pub(crate) fn native(&self) -> &NativeShapeModel {
        &self.inner
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
#[pyo3(signature = (model_image, roi, scene_image, model_config=None, search_config=None, model_mask=None))]
pub fn find_shape_model<'py>(
    py: Python<'py>,
    model_image: &Bound<'py, PyAny>,
    roi: (f32, f32, f32, f32),
    scene_image: &Bound<'py, PyAny>,
    model_config: Option<ShapeModelConfig>,
    search_config: Option<ShapeSearchConfig>,
    model_mask: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let model = ShapeModel::new(py, model_image, roi, model_config, model_mask)?;
    let mut matcher = ShapeMatcher::new(search_config);
    matcher.find(py, scene_image, &model)
}
