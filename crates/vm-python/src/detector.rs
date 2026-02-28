//! Python binding for `Edge2DDetector`.
//!
//! `PyEdgeDetector.detect_u8(img)` accepts a 2-D `numpy.ndarray` of `uint8`
//! and returns a list of dicts with keys `x`, `y`, `nx`, `ny`, `strength`.
//!
//! The GIL is held during detection; see `convert.rs` for the rationale.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use vm_edge::edge2d::{Edge2DConfig, Edge2DDetector};

use crate::convert::image_from_numpy_u8;

/// Python-facing 2-D edge detector.
///
/// Wraps `vm_edge::Edge2DDetector` with the default `Edge2DConfig`.
/// Re-uses internal scratch buffers across calls.
///
/// ## Example (Python)
/// ```python
/// import vm_python, numpy as np
/// det = vm_python.PyEdgeDetector()
/// img = np.zeros((64, 64), dtype=np.uint8)
/// img[:, 32:] = 200
/// edgels = det.detect_u8(img)  # list of dicts
/// print(edgels[0])  # {'x': ..., 'y': ..., 'nx': ..., 'ny': ..., 'strength': ...}
/// ```
#[pyclass]
pub struct PyEdgeDetector {
    det: Edge2DDetector,
    cfg: Edge2DConfig,
}

#[pymethods]
impl PyEdgeDetector {
    /// Create a new detector with default configuration.
    #[new]
    pub fn new() -> Self {
        Self {
            det: Edge2DDetector::new(),
            cfg: Edge2DConfig::default(),
        }
    }

    /// Detect edges in a 2-D `uint8` numpy array.
    ///
    /// Returns a list of dicts, each with keys:
    /// - `x` (float): subpixel x coordinate (pixel-center convention).
    /// - `y` (float): subpixel y coordinate.
    /// - `nx` (float): x component of the unit normal (dark-to-bright).
    /// - `ny` (float): y component of the unit normal.
    /// - `strength` (float): edge strength (NMS response).
    pub fn detect_u8<'py>(
        &mut self,
        py: Python<'py>,
        img: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyList>> {
        let image = image_from_numpy_u8(py, &img)?;
        let edgels = self.det.detect_u8(&image.as_view(), &self.cfg);

        let list = PyList::empty_bound(py);
        for e in &edgels {
            let d = PyDict::new_bound(py);
            d.set_item("x", e.p.x)?;
            d.set_item("y", e.p.y)?;
            d.set_item("nx", e.n.x)?;
            d.set_item("ny", e.n.y)?;
            d.set_item("strength", e.strength)?;
            list.append(d)?;
        }
        Ok(list)
    }
}
