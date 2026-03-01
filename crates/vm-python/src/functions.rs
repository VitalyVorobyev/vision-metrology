//! Declarative module-level APIs.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3::wrap_pyfunction;

use crate::config_py::{ConicFitConfig, EdgeConfig, LsdConfig, MultiScaleConfig, RigidMatchConfig};
use crate::detector::detect_edges_u8_impl;
use crate::match_py::match_rigid_model_impl;
use crate::multiscale::detect_multiscale_edges_u8_impl;
use crate::segment::{
    component_stats_impl, label_components_impl, otsu_threshold_impl, threshold_binary_impl,
};
use crate::shape::{detect_line_segments_u8_impl, fit_ellipse_impl};

#[pyfunction]
pub fn detect_edges_u8<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    config: EdgeConfig,
) -> PyResult<Bound<'py, PyList>> {
    detect_edges_u8_impl(py, img, config)
}

#[pyfunction]
pub fn detect_multiscale_edges_u8<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    config: MultiScaleConfig,
) -> PyResult<Bound<'py, PyList>> {
    detect_multiscale_edges_u8_impl(py, img, config)
}

#[pyfunction]
pub fn detect_line_segments_u8<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    config: LsdConfig,
) -> PyResult<Bound<'py, PyList>> {
    detect_line_segments_u8_impl(py, img, config)
}

#[pyfunction]
pub fn fit_ellipse<'py>(
    py: Python<'py>,
    pts: PyReadonlyArray2<'py, f32>,
    config: ConicFitConfig,
) -> PyResult<Option<Py<PyAny>>> {
    fit_ellipse_impl(py, pts, config)
}

#[pyfunction]
pub fn match_rigid_model<'py>(
    py: Python<'py>,
    model_edgels: PyReadonlyArray2<'py, f32>,
    scene_img: PyReadonlyArray2<'py, u8>,
    edge_config: EdgeConfig,
    match_config: RigidMatchConfig,
) -> PyResult<Option<Py<PyAny>>> {
    match_rigid_model_impl(py, model_edgels, scene_img, edge_config, match_config)
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
    m.add_function(wrap_pyfunction!(detect_edges_u8, m)?)?;
    m.add_function(wrap_pyfunction!(detect_multiscale_edges_u8, m)?)?;
    m.add_function(wrap_pyfunction!(detect_line_segments_u8, m)?)?;
    m.add_function(wrap_pyfunction!(fit_ellipse, m)?)?;
    m.add_function(wrap_pyfunction!(match_rigid_model, m)?)?;
    m.add_function(wrap_pyfunction!(otsu_threshold, m)?)?;
    m.add_function(wrap_pyfunction!(threshold_binary, m)?)?;
    m.add_function(wrap_pyfunction!(label_components, m)?)?;
    m.add_function(wrap_pyfunction!(component_stats, m)?)?;
    Ok(())
}
