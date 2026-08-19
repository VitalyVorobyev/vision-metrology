//! Python bindings for `LsdDetector` and `ConicFitter`.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use vision_metrology::{ConicFitter as NativeConicFitter, LsdDetector as NativeLsdDetector};
use vm_primitives::Point2f;

use crate::config_py::{ConicFitConfig, LsdConfig};
use crate::convert::image_from_numpy_u8;
use crate::types::{Ellipse, LineSegment};

fn segments_to_pylist<'py>(
    py: Python<'py>,
    segs: &[vision_metrology::LineSegment2f],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for s in segs {
        let dx = s.p2.x - s.p1.x;
        let dy = s.p2.y - s.p1.y;
        let length = (dx * dx + dy * dy).sqrt();
        let obj = Py::new(
            py,
            LineSegment {
                x1: s.p1.x,
                y1: s.p1.y,
                x2: s.p2.x,
                y2: s.p2.y,
                width: s.width,
                nfa: s.nfa,
                angle: s.angle,
                length,
            },
        )?;
        list.append(obj)?;
    }
    Ok(list)
}

fn points_from_numpy(pts: PyReadonlyArray2<'_, f32>) -> PyResult<Vec<Point2f>> {
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

    Ok((0..n)
        .map(|i| Point2f::new(slice[i * 2], slice[i * 2 + 1]))
        .collect())
}

/// Python-facing Line Segment Detector.
#[pyclass]
pub struct LsdDetector {
    det: NativeLsdDetector,
    cfg: LsdConfig,
}

#[pymethods]
impl LsdDetector {
    /// Create a new LSD detector with optional explicit config.
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<LsdConfig>) -> Self {
        Self {
            det: NativeLsdDetector::new(),
            cfg: config.unwrap_or_default(),
        }
    }

    /// Detect line segments in a 2-D `uint8` numpy array.
    pub fn detect_u8<'py>(
        &mut self,
        py: Python<'py>,
        img: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyList>> {
        let image = image_from_numpy_u8(py, &img)?;
        let cfg = self.cfg.to_native()?;
        let segs = self.det.detect(&image.as_view(), &cfg);
        segments_to_pylist(py, &segs)
    }

    /// Return a copy of the current detector config.
    pub fn config(&self) -> LsdConfig {
        self.cfg.clone()
    }

    /// Replace detector configuration.
    pub fn set_config(&mut self, config: LsdConfig) {
        self.cfg = config;
    }
}

/// Python-facing conic/ellipse fitter.
#[pyclass]
pub struct ConicFitter {
    fitter: NativeConicFitter,
    cfg: ConicFitConfig,
}

#[pymethods]
impl ConicFitter {
    /// Create a new conic fitter with optional explicit config.
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<ConicFitConfig>) -> Self {
        Self {
            fitter: NativeConicFitter::new(),
            cfg: config.unwrap_or_default(),
        }
    }

    /// Fit an ellipse to a set of 2-D points.
    ///
    /// `pts` must be a `(N, 2)` float32 numpy array.
    /// Returns an `Ellipse` on success, or `None` if fitting fails.
    pub fn fit_ellipse<'py>(
        &mut self,
        py: Python<'py>,
        pts: PyReadonlyArray2<'py, f32>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let points = points_from_numpy(pts)?;
        let cfg = self.cfg.to_native()?;
        let result = self.fitter.fit_ellipse_ransac(&points, &cfg);

        match result {
            Ok(ellipse) => {
                let obj = Py::new(
                    py,
                    Ellipse {
                        cx: ellipse.center.x,
                        cy: ellipse.center.y,
                        a: ellipse.semi_major(),
                        b: ellipse.semi_minor(),
                        angle: ellipse.angle,
                    },
                )?;
                Ok(Some(obj.into()))
            }
            Err(_) => Ok(None),
        }
    }

    /// Return a copy of the current fitter config.
    pub fn config(&self) -> ConicFitConfig {
        self.cfg.clone()
    }

    /// Replace fitter configuration.
    pub fn set_config(&mut self, config: ConicFitConfig) {
        self.cfg = config;
    }
}

pub(crate) fn detect_line_segments_u8_impl<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    config: LsdConfig,
) -> PyResult<Bound<'py, PyList>> {
    let mut det = LsdDetector::new(Some(config));
    det.detect_u8(py, img)
}

pub(crate) fn fit_ellipse_impl<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f32>,
    config: ConicFitConfig,
) -> PyResult<Option<Py<PyAny>>> {
    let mut fitter = ConicFitter::new(Some(config));
    fitter.fit_ellipse(py, pts)
}
