//! Python bindings for `vm_primitives::filter`: exact box and binomial smoothing on `uint8`,
//! and Gaussian on `float32`.
//!
//! The border is a string (`"clamp"`, `"reflect101"` or `"constant"`); `"constant"` takes its
//! fill value from `value`, and the other two ignore it.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vm_primitives::BorderMode;
use vm_primitives::filter::{binomial_u8, box_mean_u8, gaussian_f32};

use crate::convert::{image_from_numpy, image_from_numpy_u8};

fn parse_border<T: Copy>(mode: &str, value: T) -> PyResult<BorderMode<T>> {
    match mode {
        "clamp" => Ok(BorderMode::Clamp),
        "reflect101" => Ok(BorderMode::Reflect101),
        "constant" => Ok(BorderMode::Constant(value)),
        other => Err(PyValueError::new_err(format!(
            "border must be 'clamp', 'reflect101' or 'constant', got '{other}'"
        ))),
    }
}

fn to_array<'py, T: numpy::Element>(
    py: Python<'py>,
    w: usize,
    h: usize,
    data: Vec<T>,
) -> PyResult<Bound<'py, PyArray2<T>>> {
    Ok(numpy::ndarray::Array2::from_shape_vec((h, w), data)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?
        .into_pyarray(py))
}

/// Box mean of width `2·radius + 1` on a `uint8` image, rounded half up. Exact in integers.
#[pyfunction]
#[pyo3(signature = (img, radius, border="clamp", value=0))]
pub fn box_mean<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    radius: usize,
    border: &str,
    value: u8,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let image = image_from_numpy_u8(py, &img)?;
    let border = parse_border(border, value)?;
    let out = box_mean_u8(&image.as_view(), radius, border)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    to_array(py, out.width(), out.height(), out.data().to_vec())
}

/// Binomial smoothing of even `order` (2 to 12) on a `uint8` image. Exact in integers.
#[pyfunction]
#[pyo3(signature = (img, order, border="clamp", value=0))]
pub fn binomial<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    order: usize,
    border: &str,
    value: u8,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let image = image_from_numpy_u8(py, &img)?;
    let border = parse_border(border, value)?;
    let out = binomial_u8(&image.as_view(), order, border)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    to_array(py, out.width(), out.height(), out.data().to_vec())
}

/// Gaussian of standard deviation `sigma` (pixels) on a `float32` image. Below σ = 8 this is the
/// exact FIR truncated at `ceil(3σ)`; from σ = 8 on, a three-box cascade.
#[pyfunction]
#[pyo3(signature = (img, sigma, border="clamp", value=0.0))]
pub fn gaussian<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, f32>,
    sigma: f64,
    border: &str,
    value: f32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let image = image_from_numpy(py, &img)?;
    let border = parse_border(border, value)?;
    let out = gaussian_f32(&image.as_view(), sigma, border)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    to_array(py, out.width(), out.height(), out.data().to_vec())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(box_mean, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(binomial, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(gaussian, m)?)?;
    Ok(())
}
