//! Python binding for `MultiScaleEdgeDetector`.

use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use pyo3::types::PyList;
use vision_metrology::MultiScaleEdgeDetector;

use crate::config_py::MultiScaleConfig;
use crate::convert::image_from_numpy_u8;
use crate::types::Edgel;

fn edgels_to_pylist<'py>(
    py: Python<'py>,
    edgels: &[vision_metrology::ScaleAnnotatedEdgel],
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

/// Python-facing multi-scale 2-D edge detector.
#[pyclass]
pub struct MultiScaleDetector {
    det: MultiScaleEdgeDetector,
    cfg: MultiScaleConfig,
}

#[pymethods]
impl MultiScaleDetector {
    /// Create a multi-scale detector with an optional explicit config.
    #[new]
    #[pyo3(signature = (config=None))]
    pub fn new(config: Option<MultiScaleConfig>) -> Self {
        Self {
            det: MultiScaleEdgeDetector::new(),
            cfg: config.unwrap_or_default(),
        }
    }

    /// Detect edges in a 2-D `uint8` numpy array across all pyramid levels.
    pub fn detect_u8<'py>(
        &mut self,
        py: Python<'py>,
        img: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyList>> {
        let image = image_from_numpy_u8(py, &img)?;
        let cfg = self.cfg.to_native()?;
        let edgels = self.det.detect_u8(&image.as_view(), &cfg);
        edgels_to_pylist(py, &edgels)
    }

    /// Return a copy of the current detector config.
    pub fn config(&self) -> MultiScaleConfig {
        self.cfg.clone()
    }

    /// Replace detector configuration.
    pub fn set_config(&mut self, config: MultiScaleConfig) {
        self.cfg = config;
    }
}

pub(crate) fn detect_multiscale_edges_u8_impl<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    config: MultiScaleConfig,
) -> PyResult<Bound<'py, PyList>> {
    let mut det = MultiScaleDetector::new(Some(config));
    det.detect_u8(py, img)
}
