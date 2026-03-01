//! Python bindings for segmentation operations.

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyList;
use vm_contour::Connectivity;
use vm_core::Image;
use vm_segment::{CcLabel, component_stats, label_connected_components_u8, otsu_threshold_u8};

use crate::convert::image_from_numpy_u8;
use crate::types::ComponentStats;

pub(crate) fn otsu_threshold_impl(py: Python<'_>, img: PyReadonlyArray2<'_, u8>) -> PyResult<u8> {
    let image = image_from_numpy_u8(py, &img)?;
    Ok(otsu_threshold_u8(&image.as_view()))
}

pub(crate) fn threshold_binary_impl<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    threshold: u8,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let shape = img.shape();
    let h = shape[0];
    let w = shape[1];
    let slice = img.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array not C-contiguous: {e}"))
    })?;
    let data: Vec<u8> = slice
        .iter()
        .map(|&v| if v > threshold { 255u8 } else { 0u8 })
        .collect();
    let out = Image::from_vec(w, h, data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("image error: {e}")))?;
    let flat = out.data().to_vec();
    let arr = numpy::ndarray::Array2::from_shape_vec((h, w), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Ok(arr.into_pyarray(py))
}

pub(crate) fn label_components_impl<'py>(
    py: Python<'py>,
    img: PyReadonlyArray2<'py, u8>,
    connectivity: Option<u8>,
) -> PyResult<(Bound<'py, PyArray2<i32>>, u32)> {
    let image = image_from_numpy_u8(py, &img)?;
    let conn = match connectivity.unwrap_or(8) {
        4 => Connectivity::C4,
        8 => Connectivity::C8,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "connectivity must be 4 or 8, got {other}"
            )));
        }
    };

    let ccl = label_connected_components_u8(&image.as_view(), conn);
    let flat: Vec<i32> = ccl.label_map.data().iter().map(|&v| v as i32).collect();
    let arr = numpy::ndarray::Array2::from_shape_vec((image.height(), image.width()), flat)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?;
    Ok((arr.into_pyarray(py), ccl.num_labels))
}

pub(crate) fn component_stats_impl<'py>(
    py: Python<'py>,
    label_img: PyReadonlyArray2<'py, i32>,
    n_labels: u32,
    min_area: Option<u32>,
) -> PyResult<Bound<'py, PyList>> {
    let shape = label_img.shape();
    let h = shape[0];
    let w = shape[1];
    let slice = label_img.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array not C-contiguous: {e}"))
    })?;
    let data: Vec<u32> = slice.iter().map(|&v| v as u32).collect();
    let label_map = vm_core::Image::from_vec(w, h, data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("image error: {e}")))?;

    let ccl = CcLabel {
        label_map,
        num_labels: n_labels,
    };
    let stats = component_stats(&ccl, min_area.unwrap_or(1));

    let list = PyList::empty(py);
    for s in &stats {
        let obj = Py::new(
            py,
            ComponentStats {
                label: s.label,
                pixel_count: s.pixel_count,
                cx: s.centroid.x,
                cy: s.centroid.y,
                bbox_x: s.bbox.x,
                bbox_y: s.bbox.y,
                bbox_w: s.bbox.width,
                bbox_h: s.bbox.height,
            },
        )?;
        list.append(obj)?;
    }
    Ok(list)
}

/// Python-facing segmentation helper.
#[pyclass]
pub struct Segmenter {}

#[pymethods]
impl Segmenter {
    #[new]
    pub fn new() -> Self {
        Self {}
    }

    /// Compute the Otsu threshold for an 8-bit image.
    pub fn otsu_threshold(&self, py: Python<'_>, img: PyReadonlyArray2<'_, u8>) -> PyResult<u8> {
        otsu_threshold_impl(py, img)
    }

    /// Apply a global threshold: pixels > `threshold` become 255, else 0.
    pub fn threshold_binary<'py>(
        &self,
        py: Python<'py>,
        img: PyReadonlyArray2<'py, u8>,
        threshold: u8,
    ) -> PyResult<Bound<'py, PyArray2<u8>>> {
        threshold_binary_impl(py, img, threshold)
    }

    /// Label connected components in a binary image.
    pub fn label_components<'py>(
        &self,
        py: Python<'py>,
        img: PyReadonlyArray2<'py, u8>,
        connectivity: Option<u8>,
    ) -> PyResult<(Bound<'py, PyArray2<i32>>, u32)> {
        label_components_impl(py, img, connectivity)
    }

    /// Compute per-component statistics from a label image.
    pub fn component_stats<'py>(
        &self,
        py: Python<'py>,
        label_img: PyReadonlyArray2<'py, i32>,
        n_labels: u32,
        min_area: Option<u32>,
    ) -> PyResult<Bound<'py, PyList>> {
        component_stats_impl(py, label_img, n_labels, min_area)
    }
}
