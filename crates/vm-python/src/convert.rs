//! Conversion helpers between numpy arrays and `vm_primitives` image types.
//!
//! ## GIL strategy (OQ-4)
//! All image I/O uses `PyReadonlyArray2<u8>` (zero-copy view). The GIL must
//! be held for the duration of any detect call. This is the simplest correct
//! approach; callers that need multi-threaded processing should release the GIL
//! in Python and copy image data before calling into this module.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use vm_primitives::Image;

/// Convert a 2-D numpy `uint8` array to an owned `Image<u8>`.
///
/// The array must be C-contiguous (row-major). Returns a `PyErr` if the array
/// is not 2-D or not C-contiguous.
pub fn image_from_numpy_u8(_py: Python<'_>, array: &PyReadonlyArray2<u8>) -> PyResult<Image<u8>> {
    let shape = array.shape();
    let h = shape[0];
    let w = shape[1];
    // Ensure C-contiguous layout.
    let slice = array.as_slice().map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("array not C-contiguous: {e}"))
    })?;
    let data = slice.to_vec();
    Image::from_vec(w, h, data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("image error: {e}")))
}
