//! Python bindings for `contour`: edgel graph construction and smoothing.
//!
//! Minimal surface: build a graph from a `uint8` image via
//! [`build_contour_graph`] (equivalent to
//! `build_graph_from_detector_output` in Rust — edgel detection is `u8`-only
//! there, so no dtype dispatch applies), read its edges back as `(N, 2)`
//! `float32` polylines, and [`smooth_polyline`] them.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, ndarray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vision_metrology::contour::{
    Connectivity, ContourBuildConfig, ContourGraph as NativeContourGraph,
    build_graph_from_detector_output, smooth_polyline as native_smooth_polyline,
};
use vm_primitives::Point2f;
use vm_primitives::edge::{Edge2DConfig, Edge2DDetector};

use crate::config::EdgeConfig;
use crate::convert::image_from_numpy_u8;

fn points_to_pyarray<'py>(py: Python<'py>, pts: &[Point2f]) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let mut flat = Vec::with_capacity(pts.len() * 2);
    for p in pts {
        flat.push(p.x);
        flat.push(p.y);
    }
    let arr = ndarray::Array2::from_shape_vec((pts.len(), 2), flat)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(arr.into_pyarray(py))
}

/// A junction-aware graph of subpixel edgel chains.
#[pyclass]
pub struct ContourGraph {
    inner: NativeContourGraph,
}

#[pymethods]
impl ContourGraph {
    #[getter]
    pub fn num_nodes(&self) -> usize {
        self.inner.nodes.len()
    }

    #[getter]
    pub fn num_edges(&self) -> usize {
        self.inner.edges.len()
    }

    #[getter]
    pub fn num_junctions(&self) -> usize {
        self.inner.num_junctions()
    }

    /// Every edge's polyline, as `(N, 2)` `float32` arrays in edge order.
    pub fn polylines<'py>(&self, py: Python<'py>) -> PyResult<Vec<Bound<'py, PyArray2<f32>>>> {
        self.inner
            .edges
            .iter()
            .map(|e| points_to_pyarray(py, &e.points))
            .collect()
    }

    /// Arc length of each edge, in pixels, in the same order as `polylines`.
    pub fn edge_lengths(&self) -> Vec<f32> {
        self.inner.edges.iter().map(|e| e.length).collect()
    }
}

/// Run the 2-D edge detector on `img` and build a contour graph in one step.
///
/// `img` must be `uint8` — the underlying `build_graph_from_detector_output`
/// takes no other pixel type.
#[pyfunction]
#[pyo3(signature = (
    img,
    edge_config=None,
    connectivity=None,
    min_component_size=None,
    record_geometry=None,
    thin=None
))]
#[allow(clippy::too_many_arguments)]
pub fn build_contour_graph(
    py: Python<'_>,
    img: PyReadonlyArray2<'_, u8>,
    edge_config: Option<EdgeConfig>,
    connectivity: Option<String>,
    min_component_size: Option<usize>,
    record_geometry: Option<bool>,
    thin: Option<bool>,
) -> PyResult<ContourGraph> {
    let image = image_from_numpy_u8(py, &img)?;
    let edge_cfg: Edge2DConfig = edge_config.unwrap_or_default().to_native()?;
    let connectivity = match connectivity.as_deref() {
        None | Some("c8") => Connectivity::C8,
        Some("c4") => Connectivity::C4,
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "connectivity must be 'c4' or 'c8', got '{other}'"
            )));
        }
    };
    let contour_cfg = ContourBuildConfig {
        connectivity,
        min_component_size: min_component_size.unwrap_or(2),
        record_strengths: false,
        record_geometry: record_geometry.unwrap_or(false),
        thin: thin.unwrap_or(true),
    };
    let mut detector = Edge2DDetector::new();
    let inner =
        build_graph_from_detector_output(&image.as_view(), &mut detector, &edge_cfg, &contour_cfg);
    Ok(ContourGraph { inner })
}

/// Gaussian-smooth a polyline in arc-length space.
///
/// `points` is an `(N, 2)` `float32` array; returns an array of the same
/// shape.
#[pyfunction]
pub fn smooth_polyline<'py>(
    py: Python<'py>,
    points: PyReadonlyArray2<'py, f32>,
    sigma: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let shape = points.shape();
    if shape.len() != 2 || shape[1] != 2 {
        return Err(PyValueError::new_err(
            "points must be an (N, 2) float32 array",
        ));
    }
    let slice = points
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("array not C-contiguous: {e}")))?;
    // `.0` drops a remainder the shape check above already ruled out: the
    // array is (N, 2), so the slice length is even.
    let pts: Vec<Point2f> = slice
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[x, y]| Point2f::new(x, y))
        .collect();
    let smoothed = native_smooth_polyline(&pts, sigma);
    points_to_pyarray(py, &smoothed)
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ContourGraph>()?;
    m.add_function(pyo3::wrap_pyfunction!(build_contour_graph, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(smooth_polyline, m)?)?;
    Ok(())
}
