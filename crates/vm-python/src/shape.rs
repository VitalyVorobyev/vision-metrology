//! Python bindings for `LsdDetector` and `ConicFitter`.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use vm_core::Point2f;
use vm_shape::{ConicFitConfig, ConicFitter, LsdConfig, LsdDetector};

use crate::convert::image_from_numpy_u8;

// ---------------------------------------------------------------------------
// PyLsdDetector
// ---------------------------------------------------------------------------

/// Python-facing Line Segment Detector.
///
/// ## Example (Python)
/// ```python
/// det = vm_python.PyLsdDetector()
/// segs = det.detect(img)  # list of dicts
/// ```
#[pyclass]
pub struct PyLsdDetector {
    det: LsdDetector,
    cfg: LsdConfig,
}

#[pymethods]
impl PyLsdDetector {
    /// Create a new LSD detector with default configuration.
    #[new]
    pub fn new() -> Self {
        Self {
            det: LsdDetector::new(),
            cfg: LsdConfig::default(),
        }
    }

    /// Detect line segments in a 2-D `uint8` numpy array.
    ///
    /// Returns a list of dicts, each with keys:
    /// - `x1`, `y1` (float): first endpoint.
    /// - `x2`, `y2` (float): second endpoint.
    /// - `width` (float): estimated line width.
    /// - `nfa` (float): log₁₀(NFA) — negative means significant.
    /// - `angle` (float): orientation angle in radians.
    pub fn detect<'py>(
        &mut self,
        py: Python<'py>,
        img: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyList>> {
        let image = image_from_numpy_u8(py, &img)?;
        let segs = self.det.detect(&image.as_view(), &self.cfg);

        let list = PyList::empty_bound(py);
        for s in &segs {
            let d = PyDict::new_bound(py);
            d.set_item("x1", s.p1.x)?;
            d.set_item("y1", s.p1.y)?;
            d.set_item("x2", s.p2.x)?;
            d.set_item("y2", s.p2.y)?;
            d.set_item("width", s.width)?;
            d.set_item("nfa", s.nfa)?;
            d.set_item("angle", s.angle)?;
            list.append(d)?;
        }
        Ok(list)
    }
}

// ---------------------------------------------------------------------------
// PyConicFitter
// ---------------------------------------------------------------------------

/// Python-facing conic/ellipse fitter.
///
/// ## Example (Python)
/// ```python
/// fitter = vm_python.PyConicFitter()
/// pts = np.array([[x, y], ...], dtype=np.float32)
/// result = fitter.fit_ellipse(pts)  # dict or None
/// ```
#[pyclass]
pub struct PyConicFitter {
    fitter: ConicFitter,
    cfg: ConicFitConfig,
}

#[pymethods]
impl PyConicFitter {
    /// Create a new conic fitter with default configuration.
    #[new]
    pub fn new() -> Self {
        Self {
            fitter: ConicFitter::new(),
            cfg: ConicFitConfig::default(),
        }
    }

    /// Fit an ellipse to a set of 2-D points.
    ///
    /// `pts` must be a `(N, 2)` float32 numpy array.
    ///
    /// Returns a dict with keys `cx`, `cy`, `a`, `b`, `angle` on success,
    /// or `None` if fitting fails or the result is not an ellipse.
    pub fn fit_ellipse<'py>(
        &mut self,
        py: Python<'py>,
        pts: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let shape = pts.shape();
        if shape.len() != 2 || shape[1] != 2 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "pts must be a (N, 2) float32 array",
            ));
        }
        let n = shape[0];
        let slice = pts.as_slice().map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("array not C-contiguous: {e}"))
        })?;

        let points: Vec<Point2f> = (0..n)
            .map(|i| Point2f {
                x: slice[i * 2],
                y: slice[i * 2 + 1],
            })
            .collect();

        let result = self.fitter.fit_ellipse_ransac(&points, &self.cfg);
        match result {
            Ok(ellipse) => {
                let d = PyDict::new_bound(py);
                d.set_item("cx", ellipse.center.x)?;
                d.set_item("cy", ellipse.center.y)?;
                d.set_item("a", ellipse.semi_axes.x)?;
                d.set_item("b", ellipse.semi_axes.y)?;
                d.set_item("angle", ellipse.angle)?;
                Ok(Some(d))
            }
            Err(_) => Ok(None),
        }
    }
}
