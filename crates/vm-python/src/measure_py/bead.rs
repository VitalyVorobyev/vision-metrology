//! `BeadTracker`, `TrackedBead` and `BeadPass`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vision_metrology::measure::{
    BeadPass as NativeBeadPass, BeadReject, BeadTracker as NativeBeadTracker,
    TrackedBead as NativeTrackedBead,
};
use vm_primitives::Point2f;

use crate::config::BeadConfig;
use crate::convert::{any_image_from_numpy, with_any_image};

/// The prior: an `(N, 2)` `float32` array of `(x, y)` rows.
fn prior_from_numpy(prior: &PyReadonlyArray2<'_, f32>) -> PyResult<Vec<Point2f>> {
    let shape = prior.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "prior must be an (N, 2) float32 array, got shape {shape:?}"
        )));
    }
    let slice = prior
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("prior array not C-contiguous: {e}")))?;
    Ok((0..shape[0])
        .map(|i| Point2f::new(slice[2 * i], slice[2 * i + 1]))
        .collect())
}

/// Reject counts as a dict, in the fixed reason order.
fn rejects_dict<'py>(py: Python<'py>, rejects: &[(BeadReject, usize)]) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    for &(r, n) in rejects {
        d.set_item(r.as_str(), n)?;
    }
    Ok(d.unbind())
}

/// `(N, 2)` rows as a `float32` array.
fn rows(py: Python<'_>, xy: Vec<f32>) -> PyResult<Py<PyArray2<f32>>> {
    let n = xy.len() / 2;
    let arr =
        Array2::from_shape_vec((n, 2), xy).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).unbind())
}

/// One tracking pass — mirrors `vision_metrology::measure::BeadPass`.
///
/// Lengths are in px. `step_scale` is the fraction of the solved correction applied (0
/// when the pass did not move the curve); `rejects` counts the stations rejected, by
/// reason.
#[pyclass(get_all, skip_from_py_object)]
pub struct BeadPass {
    pub n_valid: usize,
    pub support: f32,
    pub longest_gap: f32,
    pub correction_rms: f32,
    pub correction_max: f32,
    pub residual_rms: f32,
    pub residual_max: f32,
    pub step_scale: f32,
    pub irls_iters: usize,
    pub rejects: Py<PyDict>,
}

impl BeadPass {
    fn from_native(py: Python<'_>, p: &NativeBeadPass) -> PyResult<Self> {
        Ok(Self {
            n_valid: p.n_valid,
            support: p.support,
            longest_gap: p.longest_gap,
            correction_rms: p.correction_rms,
            correction_max: p.correction_max,
            residual_rms: p.residual_rms,
            residual_max: p.residual_max,
            step_scale: p.step_scale,
            irls_iters: p.irls_iters,
            rejects: rejects_dict(py, &p.rejects)?,
        })
    }
}

#[pymethods]
impl BeadPass {
    fn __repr__(&self) -> String {
        format!(
            "BeadPass(n_valid={}, correction_max={:.4}, residual_rms={:.4}, step_scale={:.3})",
            self.n_valid, self.correction_max, self.residual_rms, self.step_scale
        )
    }
}

/// A tracked bead — mirrors `vision_metrology::measure::TrackedBead`, as arrays.
///
/// Every array has one row per station of the refined curve. `centerline` is the next
/// call's prior as it stands. `normals` are the unit normals the final strips scanned
/// along; `offset`, `width`, `confidence`, `center`, `first` and `second` (the edges on
/// the −n and +n sides) come from the final stage and are NaN where it rejected, and
/// `reject` names the reason there (`None` at a hit). The statistics over the hits are
/// `None` when there are none.
#[pyclass(get_all, skip_from_py_object)]
pub struct TrackedBead {
    pub centerline: Py<PyArray2<f32>>,
    pub spacing: f32,
    pub normals: Py<PyArray2<f32>>,
    pub offset: Py<PyArray1<f32>>,
    pub width: Py<PyArray1<f32>>,
    pub confidence: Py<PyArray1<f32>>,
    pub center: Py<PyArray2<f32>>,
    pub first: Py<PyArray2<f32>>,
    pub second: Py<PyArray2<f32>>,
    pub reject: Vec<Option<&'static str>>,
    pub support: f32,
    pub longest_gap: f32,
    pub n_used: usize,
    pub center_rms: Option<f32>,
    pub center_max_dev: Option<f32>,
    pub width_mean: Option<f32>,
    pub width_std: Option<f32>,
    pub width_min: Option<f32>,
    pub width_max: Option<f32>,
    pub rejects: Py<PyDict>,
    /// "converged", "pass_limit" or "too_few_valid".
    pub stop: &'static str,
    pub passes: Vec<Py<BeadPass>>,
}

impl TrackedBead {
    fn from_native(py: Python<'_>, b: NativeTrackedBead) -> PyResult<Self> {
        let n = b.samples.len();
        let (mut normals, mut center, mut first, mut second) = (
            Vec::with_capacity(2 * n),
            Vec::with_capacity(2 * n),
            Vec::with_capacity(2 * n),
            Vec::with_capacity(2 * n),
        );
        let (mut offset, mut width, mut confidence) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        let mut reject = Vec::with_capacity(n);
        for s in &b.samples {
            normals.extend([s.normal.x, s.normal.y]);
            match &s.hit {
                Ok(h) => {
                    center.extend([h.pair.center.x, h.pair.center.y]);
                    first.extend([h.pair.first.p.x, h.pair.first.p.y]);
                    second.extend([h.pair.second.p.x, h.pair.second.p.y]);
                    offset.push(h.offset);
                    width.push(h.pair.width);
                    confidence.push(h.confidence);
                    reject.push(None);
                }
                Err(r) => {
                    for v in [&mut center, &mut first, &mut second] {
                        v.extend([f32::NAN, f32::NAN]);
                    }
                    offset.push(f32::NAN);
                    width.push(f32::NAN);
                    confidence.push(f32::NAN);
                    reject.push(Some(r.as_str()));
                }
            }
        }
        let centerline = b.centerline.iter().flat_map(|p| [p.x, p.y]).collect();
        let stats = b.summary.stats;
        let passes = b
            .track
            .passes
            .iter()
            .map(|p| Py::new(py, BeadPass::from_native(py, p)?))
            .collect::<PyResult<_>>()?;
        Ok(Self {
            centerline: rows(py, centerline)?,
            spacing: b.spacing,
            normals: rows(py, normals)?,
            offset: offset.into_pyarray(py).unbind(),
            width: width.into_pyarray(py).unbind(),
            confidence: confidence.into_pyarray(py).unbind(),
            center: rows(py, center)?,
            first: rows(py, first)?,
            second: rows(py, second)?,
            reject,
            support: b.summary.support,
            longest_gap: b.summary.longest_gap,
            n_used: stats.map_or(0, |s| s.n_used),
            center_rms: stats.map(|s| s.center_rms),
            center_max_dev: stats.map(|s| s.center_max_dev),
            width_mean: stats.map(|s| s.width_mean),
            width_std: stats.map(|s| s.width_std),
            width_min: stats.map(|s| s.width_min),
            width_max: stats.map(|s| s.width_max),
            rejects: rejects_dict(py, &b.summary.rejects)?,
            stop: b.track.stop.as_str(),
            passes,
        })
    }
}

#[pymethods]
impl TrackedBead {
    fn __repr__(&self) -> String {
        format!(
            "TrackedBead(stations={}, support={:.3}, stop='{}', width_mean={:?})",
            self.reject.len(),
            self.support,
            self.stop,
            self.width_mean
        )
    }
}

/// Tracks a bead along a prior curve and measures it — mirrors
/// `vision_metrology::measure::BeadTracker`.
///
/// `BeadTracker(config)` raises `ValueError` for an invalid config. `track(image, prior)`
/// takes a `uint8`, `uint16` or `float32` image and an `(N, 2)` `float32` prior, and
/// raises `ValueError` for a prior with fewer than two points, a non-finite point or no
/// length; a bead that is not there is a `TrackedBead` with every station rejected.
#[pyclass]
pub struct BeadTracker {
    inner: NativeBeadTracker,
    config: BeadConfig,
}

#[pymethods]
impl BeadTracker {
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<BeadConfig>) -> PyResult<Self> {
        let config = config.unwrap_or_default();
        let inner = NativeBeadTracker::new(config.to_native()?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner, config })
    }

    /// The config the tracker runs with; assigning one validates it as the constructor
    /// does.
    #[getter]
    pub fn config(&self) -> BeadConfig {
        self.config.clone()
    }

    #[setter]
    pub fn set_config(&mut self, config: BeadConfig) -> PyResult<()> {
        self.inner
            .set_config(config.to_native()?)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        self.config = config;
        Ok(())
    }

    /// Track the bead on `image` from `prior` and measure it along the refined curve.
    pub fn track(
        &mut self,
        py: Python<'_>,
        image: &Bound<'_, PyAny>,
        prior: PyReadonlyArray2<'_, f32>,
    ) -> PyResult<TrackedBead> {
        let prior = prior_from_numpy(&prior)?;
        let any = any_image_from_numpy(py, image)?;
        let bead = with_any_image!(any, view => self.inner.track(&view, &prior))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        TrackedBead::from_native(py, bead)
    }
}
