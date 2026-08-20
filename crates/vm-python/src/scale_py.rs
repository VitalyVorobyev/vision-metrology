//! Python bindings for `scale`: `estimate_scale_moments`,
//! `estimate_scale_logpolar`, `find_scale_invariant_roi`/
//! `find_scale_invariant_center` (roadmap W7, "estimate-then-verify").
//!
//! `u8`-only, like `corr` — both estimators build on `segment`/`corr`,
//! which are `u8`-only in this workspace already.
//!
//! `ScaleHint` (a Rust enum) is two functions here instead of one function
//! plus a tagged-union argument: `find_scale_invariant_roi` (moments) and
//! `find_scale_invariant_center` (log-polar) — simpler for a Python caller
//! than modelling a two-variant enum as its own class for a single call
//! site.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use vision_metrology::scale::{
    ScaleHint, estimate_scale_logpolar as native_estimate_scale_logpolar,
    estimate_scale_moments as native_estimate_scale_moments,
    find_scale_invariant as native_find_scale_invariant,
};
use vm_primitives::{Point2f, Rect2f};

use crate::config::{LogPolarScaleConfig, MomentScaleConfig, ScaleInvariantConfig};
use crate::convert::image_from_numpy_u8;
use crate::match_py::ShapeModel;
use crate::types::ShapeMatch;
use numpy::PyReadonlyArray2;

fn to_rect(roi: (f32, f32, f32, f32)) -> Rect2f {
    Rect2f {
        x: roi.0,
        y: roi.1,
        width: roi.2,
        height: roi.3,
    }
}

/// One estimator's answer: an approximate scale, optionally a rotation, and
/// a confidence-ish score whose meaning differs between estimators — see
/// each function's own docs, not this class.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct ScaleEstimate {
    pub scale: f32,
    pub angle: Option<f32>,
    pub score: f32,
}

impl From<vision_metrology::scale::ScaleEstimate> for ScaleEstimate {
    fn from(e: vision_metrology::scale::ScaleEstimate) -> Self {
        Self {
            scale: e.scale,
            angle: e.angle,
            score: e.score,
        }
    }
}

#[pymethods]
impl ScaleEstimate {
    fn __repr__(&self) -> String {
        format!(
            "ScaleEstimate(scale={:.4}, angle={:?}, score={:.4})",
            self.scale, self.angle, self.score
        )
    }
}

/// Estimate scale from a segmented scene blob's spatial spread, compared to
/// the taught model's own — see the Rust `estimate_scale_moments` docs for
/// the full algorithm and what `score` means here (a fill-fraction
/// diagnostic, not a probability). `scene` is a `(H, W)` `uint8` array;
/// `roi` is `(x, y, width, height)`.
#[pyfunction]
#[pyo3(signature = (model, scene, roi, config=None))]
pub fn estimate_scale_moments(
    py: Python<'_>,
    model: &ShapeModel,
    scene: PyReadonlyArray2<'_, u8>,
    roi: (f32, f32, f32, f32),
    config: Option<MomentScaleConfig>,
) -> PyResult<ScaleEstimate> {
    let img = image_from_numpy_u8(py, &scene)?;
    let cfg = config.unwrap_or_default().to_native()?;
    let est = native_estimate_scale_moments(model.native(), &img.as_view(), to_rect(roi), &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(est.into())
}

/// Estimate scale (and, with `config.angle_margin` set, rotation) via
/// log-polar ZNCC correlation — see the Rust `estimate_scale_logpolar` docs.
/// `scene` is a `(H, W)` `uint8` array; `approx_center` is `(x, y)`.
///
/// Requires `model.teach_point_count > 0` (format-4 teach data) — raises
/// :class:`ValueError` otherwise, same requirement as
/// :meth:`ShapeModel.resample_at`.
#[pyfunction]
#[pyo3(signature = (model, scene, approx_center, config=None))]
pub fn estimate_scale_logpolar(
    py: Python<'_>,
    model: &ShapeModel,
    scene: PyReadonlyArray2<'_, u8>,
    approx_center: (f32, f32),
    config: Option<LogPolarScaleConfig>,
) -> PyResult<ScaleEstimate> {
    let img = image_from_numpy_u8(py, &scene)?;
    let cfg = config.unwrap_or_default().to_native()?;
    let est = native_estimate_scale_logpolar(
        model.native(),
        &img.as_view(),
        Point2f::new(approx_center.0, approx_center.1),
        &cfg,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(est.into())
}

/// Estimate scale via [`estimate_scale_moments`] over `roi`, resample
/// `model` at that estimate, and verify in a narrow band — the whole
/// estimate-then-verify strategy as one call. Returns a list of
/// `ShapeMatch`, empty (not an error) when nothing scores above
/// `config.search.min_score`.
#[pyfunction]
#[pyo3(signature = (model, scene, roi, config=None))]
pub fn find_scale_invariant_roi<'py>(
    py: Python<'py>,
    model: &ShapeModel,
    scene: PyReadonlyArray2<'py, u8>,
    roi: (f32, f32, f32, f32),
    config: Option<ScaleInvariantConfig>,
) -> PyResult<Bound<'py, PyList>> {
    let img = image_from_numpy_u8(py, &scene)?;
    let cfg = config.unwrap_or_default().to_native()?;
    let hint = ScaleHint::Roi(to_rect(roi));
    let out = native_find_scale_invariant(model.native(), &img.as_view(), hint, &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyList::new(py, out.into_iter().map(ShapeMatch::from))
}

/// Same as [`find_scale_invariant_roi`], estimating scale via
/// [`estimate_scale_logpolar`] around `center` instead of segmenting `roi`.
#[pyfunction]
#[pyo3(signature = (model, scene, center, config=None))]
pub fn find_scale_invariant_center<'py>(
    py: Python<'py>,
    model: &ShapeModel,
    scene: PyReadonlyArray2<'py, u8>,
    center: (f32, f32),
    config: Option<ScaleInvariantConfig>,
) -> PyResult<Bound<'py, PyList>> {
    let img = image_from_numpy_u8(py, &scene)?;
    let cfg = config.unwrap_or_default().to_native()?;
    let hint = ScaleHint::Center(Point2f::new(center.0, center.1));
    let out = native_find_scale_invariant(model.native(), &img.as_view(), hint, &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyList::new(py, out.into_iter().map(ShapeMatch::from))
}
