//! Declarative module-level APIs.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

use crate::config::{EdgeConfig, FitConfig, LsdConfig};
use crate::detector::detect_edges_impl;
use crate::segment::{
    component_stats_impl, label_components_impl, otsu_threshold_impl, threshold_binary_impl,
};
use crate::shape::{detect_line_segments_impl, fit_ellipse_impl, fit_line_impl};

/// Detect edges in a `uint8`, `uint16` or `float32` image.
#[pyfunction]
pub fn detect_edges<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    config: EdgeConfig,
) -> PyResult<Bound<'py, PyList>> {
    detect_edges_impl(py, img, config)
}

/// Detect line segments in a `uint8`, `uint16` or `float32` image.
#[pyfunction]
pub fn detect_line_segments<'py>(
    py: Python<'py>,
    img: &Bound<'py, PyAny>,
    config: LsdConfig,
) -> PyResult<Bound<'py, PyList>> {
    detect_line_segments_impl(py, img, config)
}

#[pyfunction]
pub fn fit_ellipse<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f32>,
    config: FitConfig,
) -> PyResult<Option<Py<PyAny>>> {
    fit_ellipse_impl(py, pts, config)
}

#[pyfunction]
pub fn fit_line<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f32>,
    config: FitConfig,
) -> PyResult<Option<Py<PyAny>>> {
    fit_line_impl(py, pts, config)
}

#[pyfunction]
pub fn otsu_threshold(py: Python<'_>, img: PyReadonlyArray2<'_, u8>) -> PyResult<u8> {
    otsu_threshold_impl(py, img)
}

#[pyfunction]
pub fn threshold_binary<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    threshold: u8,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    threshold_binary_impl(py, img, threshold)
}

#[pyfunction]
#[pyo3(signature = (img, connectivity=8))]
pub fn label_components<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    connectivity: u8,
) -> PyResult<(Bound<'py, PyArray2<i32>>, u32)> {
    label_components_impl(py, img, Some(connectivity))
}

#[pyfunction]
#[pyo3(signature = (label_img, n_labels, min_area=1))]
pub fn component_stats<'py>(
    py: Python<'py>,
    label_img: PyReadonlyArray2<'py, i32>,
    n_labels: u32,
    min_area: u32,
) -> PyResult<Bound<'py, PyList>> {
    component_stats_impl(py, label_img, n_labels, Some(min_area))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(detect_edges, m)?)?;
    m.add_function(wrap_pyfunction!(detect_line_segments, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ellipse, m)?)?;
    m.add_function(wrap_pyfunction!(fit_line, m)?)?;
    m.add_function(wrap_pyfunction!(crate::match_py::find_shape_model, m)?)?;
    m.add_function(wrap_pyfunction!(otsu_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_binary, m)?)?;
    m.add_function(wrap_pyfunction!(label_components, m)?)?;
    m.add_function(wrap_pyfunction!(component_stats, m)?)?;
    crate::contour_py::register(m)?;
    crate::morph_py::register(m)?;
    Ok(())
}
