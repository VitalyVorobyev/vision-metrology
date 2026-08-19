//! Python bindings for `morph`: binary morphology, chamfer distance, thinning.
//!
//! Every function here operates on `uint8` binary images (`> 0` is
//! foreground) — the underlying Rust functions are genuinely `u8`-only
//! (`erode_binary_u8`, `chamfer_distance_u8`, ...), so there is no dtype
//! dispatch to add; the Python names simply drop the now-redundant `_u8`
//! suffix, matching every other single-dtype entry point in this module.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vm_primitives::morph::{
    StructuringElement, chamfer_distance_u8, close_binary_u8, dilate_binary_u8, erode_binary_u8,
    open_binary_u8, thin_binary_u8,
};

use crate::convert::image_from_numpy_u8;

fn parse_se(shape: &str, radius: usize) -> PyResult<StructuringElement> {
    match shape {
        "square" => Ok(StructuringElement::Square(radius)),
        "disk" => Ok(StructuringElement::Disk(radius)),
        other => Err(PyValueError::new_err(format!(
            "shape must be 'square' or 'disk', got '{other}'"
        ))),
    }
}

macro_rules! morph_fn {
    ($name:ident, $native:ident, $py_name:literal) => {
        #[pyfunction]
        #[pyo3(name = $py_name, signature = (img, shape="square", radius=1))]
        pub fn $name<'py>(
            py: Python<'py>,
            img: PyReadonlyArray2<'py, u8>,
            shape: &str,
            radius: usize,
        ) -> PyResult<Bound<'py, PyArray2<u8>>> {
            let image = image_from_numpy_u8(py, &img)?;
            let se = parse_se(shape, radius)?;
            let out = $native(&image.as_view(), &se);
            Ok(numpy::ndarray::Array2::from_shape_vec(
                (out.height(), out.width()),
                out.data().to_vec(),
            )
            .map_err(|e| PyValueError::new_err(format!("{e}")))?
            .into_pyarray(py))
        }
    };
}

morph_fn!(erode, erode_binary_u8, "erode");
morph_fn!(dilate, dilate_binary_u8, "dilate");
morph_fn!(open_, open_binary_u8, "open");
morph_fn!(close_, close_binary_u8, "close");

/// Zhang-Suen skeletonisation of a binary image.
#[pyfunction]
pub fn thin<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let image = image_from_numpy_u8(py, &img)?;
    let out = thin_binary_u8(&image.as_view());
    Ok(
        numpy::ndarray::Array2::from_shape_vec((out.height(), out.width()), out.data().to_vec())
            .map_err(|e| PyValueError::new_err(format!("{e}")))?
            .into_pyarray(py),
    )
}

/// 3-4-5 chamfer approximation to the Euclidean distance transform.
#[pyfunction]
pub fn chamfer_distance<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let image = image_from_numpy_u8(py, &img)?;
    let out = chamfer_distance_u8(&image.as_view());
    Ok(
        numpy::ndarray::Array2::from_shape_vec((out.height(), out.width()), out.data().to_vec())
            .map_err(|e| PyValueError::new_err(format!("{e}")))?
            .into_pyarray(py),
    )
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(erode, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(dilate, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(open_, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(close_, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(thin, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(chamfer_distance, m)?)?;
    Ok(())
}
