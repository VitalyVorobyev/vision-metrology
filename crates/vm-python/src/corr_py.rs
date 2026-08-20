//! Python bindings for `corr`: cross-correlation matching (`CorrTemplate` /
//! `find` / `find_topk`) and inter-frame subpixel `displacement`.
//!
//! `u8`-only, unlike most of this workspace's `ImageAny`-dispatched entry
//! points — corrmatch's published API (0.2.5) is `u8`-only, so a `uint16`/
//! `float32` array is a `ValueError` here rather than a silent quantization.

use numpy::PyReadonlyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use vision_metrology::corr::{
    CorrTemplate as NativeCorrTemplate, displacement as native_displacement, find as native_find,
    find_topk as native_find_topk,
};
use vm_primitives::Rect2f;

use crate::config::{CorrConfig, CorrTemplateConfig, DisplacementConfig};
use crate::convert::image_from_numpy_u8;

fn to_rect(roi: (f32, f32, f32, f32)) -> Rect2f {
    Rect2f {
        x: roi.0,
        y: roi.1,
        width: roi.2,
        height: roi.3,
    }
}

/// One match: the template's centre in scene coordinates, its rotation
/// (radians, zero unless the search enabled rotation), and a correlation
/// score. Not `ShapeMatch`: `score` is a raw correlation coefficient (ZNCC
/// in `[-1, 1]`, or a negative SSE for `metric='ssd'`), not an
/// occlusion-robust `1 - occluded_fraction`, and there is no scale.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct CorrMatch {
    pub x: f32,
    pub y: f32,
    pub angle: f32,
    pub score: f32,
}

impl From<vision_metrology::corr::CorrMatch> for CorrMatch {
    fn from(m: vision_metrology::corr::CorrMatch) -> Self {
        Self {
            x: m.position.x,
            y: m.position.y,
            angle: m.angle,
            score: m.score,
        }
    }
}

#[pymethods]
impl CorrMatch {
    fn __repr__(&self) -> String {
        format!(
            "CorrMatch(x={:.3}, y={:.3}, angle={:.4}, score={:.4})",
            self.x, self.y, self.angle, self.score
        )
    }
}

/// Compiled matching assets for one reference patch: corrmatch's own
/// pyramid, and — when `config.rotation` is set — a per-level angle bank of
/// masked ZNCC/SSD plans. Build once, reuse across many `find`/`find_topk`
/// calls.
#[pyclass]
pub struct CorrTemplate {
    inner: NativeCorrTemplate,
}

#[pymethods]
impl CorrTemplate {
    /// Compiles a reference patch out of a `(H, W)` `uint8` array. `rect` is
    /// `(x, y, width, height)`, rounded to whole pixels and clamped to the
    /// image.
    #[new]
    #[pyo3(signature = (image, rect, config=None))]
    pub fn new(
        py: Python<'_>,
        image: PyReadonlyArray2<'_, u8>,
        rect: (f32, f32, f32, f32),
        config: Option<CorrTemplateConfig>,
    ) -> PyResult<Self> {
        let img = image_from_numpy_u8(py, &image)?;
        let cfg = config.unwrap_or_default().to_native();
        let inner = NativeCorrTemplate::from_image(&img.as_view(), to_rect(rect), &cfg)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Template width in pixels, as compiled (the rounded `rect`).
    #[getter]
    pub fn width(&self) -> usize {
        self.inner.width()
    }

    /// Template height in pixels, as compiled (the rounded `rect`).
    #[getter]
    pub fn height(&self) -> usize {
        self.inner.height()
    }

    /// Whether this template was compiled with rotation support.
    #[getter]
    pub fn is_rotated(&self) -> bool {
        self.inner.is_rotated()
    }

    fn __repr__(&self) -> String {
        format!(
            "CorrTemplate(width={}, height={}, rotated={})",
            self.inner.width(),
            self.inner.height(),
            self.inner.is_rotated()
        )
    }
}

/// Finds `template`'s best match in a `(H, W)` `uint8` scene.
#[pyfunction]
#[pyo3(signature = (template, scene, config=None))]
pub fn find(
    py: Python<'_>,
    template: &CorrTemplate,
    scene: PyReadonlyArray2<'_, u8>,
    config: Option<CorrConfig>,
) -> PyResult<CorrMatch> {
    let img = image_from_numpy_u8(py, &scene)?;
    let cfg = config.unwrap_or_default().to_native()?;
    let m = native_find(&template.inner, &img.as_view(), &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(m.into())
}

/// Finds up to `k` matches in a `(H, W)` `uint8` scene, best score first.
#[pyfunction]
#[pyo3(signature = (template, scene, k, config=None))]
pub fn find_topk<'py>(
    py: Python<'py>,
    template: &CorrTemplate,
    scene: PyReadonlyArray2<'py, u8>,
    k: usize,
    config: Option<CorrConfig>,
) -> PyResult<Bound<'py, PyList>> {
    let img = image_from_numpy_u8(py, &scene)?;
    let cfg = config.unwrap_or_default().to_native()?;
    let ms = native_find_topk(&template.inner, &img.as_view(), k, &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyList::new(py, ms.into_iter().map(CorrMatch::from))
}

/// Subpixel shift from `prev` to `curr`: `(dx, dy)` in pixels, plus the
/// stage-1 (corrmatch ZNCC) score at the accepted match. Stage 2
/// (Lucas-Kanade, when `config.refine` requests it) only refines
/// `(dx, dy)` — it has no score of its own.
#[pyclass(get_all, skip_from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct Displacement {
    pub dx: f32,
    pub dy: f32,
    pub score: f32,
}

#[pymethods]
impl Displacement {
    fn __repr__(&self) -> String {
        format!(
            "Displacement(dx={:.4}, dy={:.4}, score={:.4})",
            self.dx, self.dy, self.score
        )
    }
}

/// Tracks `config.window` (in `prev`) into `curr`. Both `(H, W)` `uint8`.
///
/// Two-stage: a bounded ZNCC search (corrmatch, rotation off) around the
/// window's position in `prev`, then — by default — translation-only
/// inverse-compositional Lucas-Kanade on top, which removes the quadratic
/// peak's bias toward integer pixel positions.
#[pyfunction]
pub fn displacement(
    py: Python<'_>,
    prev: PyReadonlyArray2<'_, u8>,
    curr: PyReadonlyArray2<'_, u8>,
    config: &DisplacementConfig,
) -> PyResult<Displacement> {
    let prev_img = image_from_numpy_u8(py, &prev)?;
    let curr_img = image_from_numpy_u8(py, &curr)?;
    let cfg = config.to_native();
    let d = native_displacement(&prev_img.as_view(), &curr_img.as_view(), &cfg)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(Displacement {
        dx: d.shift.x,
        dy: d.shift.y,
        score: d.score,
    })
}
