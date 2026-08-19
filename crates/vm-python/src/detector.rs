//! Python binding for `Edge2DDetector`.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyList;
use vm_primitives::edge::Edge2DDetector;

use crate::config::EdgeConfig;
use crate::convert::image_from_numpy_u8;
use crate::types::Edgel;

fn edgels_to_pylist<'py>(
    py: Python<'py>,
    edgels: &[vm_primitives::edge::Edgel],
) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for e in edgels {
        let obj = Py::new(
            py,
            Edgel {
                x: e.p.x,
                y: e.p.y,
                nx: e.n.x,
                ny: e.n.y,
                strength: e.strength,
            },
        )?;
        list.append(obj)?;
    }
    Ok(list)
}

/// Python-facing 2-D edge detector.
#[pyclass]
pub struct EdgeDetector {
    det: Edge2DDetector,
    cfg: EdgeConfig,
}

#[pymethods]
impl EdgeDetector {
    /// Create a detector with an optional explicit configuration.
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<EdgeConfig>) -> Self {
        Self {
            det: Edge2DDetector::new(),
            cfg: config.unwrap_or_default(),
        }
    }

    /// Detect edges in a 2-D `uint8` numpy array.
    pub fn detect_u8<'py>(
        &mut self,
        py: Python<'py>,
        img: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyList>> {
        let image = image_from_numpy_u8(py, &img)?;
        let cfg = self.cfg.to_native()?;
        let edgels = self.det.detect(&image.as_view(), &cfg);
        edgels_to_pylist(py, &edgels)
    }

    /// Return a copy of the current detector config.
    pub fn config(&self) -> EdgeConfig {
        self.cfg.clone()
    }

    /// Replace detector configuration.
    pub fn set_config(&mut self, config: EdgeConfig) {
        self.cfg = config;
    }
}

pub(crate) fn detect_edges_u8_impl<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    config: EdgeConfig,
) -> PyResult<Bound<'py, PyList>> {
    let mut det = EdgeDetector::new(Some(config));
    det.detect_u8(py, img)
}
