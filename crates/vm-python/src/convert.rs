//! Conversion helpers between numpy arrays and `vm_primitives` image types.
//!
//! ## GIL strategy (OQ-4)
//! All image I/O uses `PyReadonlyArray2<T>` (zero-copy view). The GIL must
//! be held for the duration of any detect call. This is the simplest correct
//! approach; callers that need multi-threaded processing should release the GIL
//! in Python and copy image data before calling into this module.
//!
//! ## Dtype dispatch
//! Every algorithm in this workspace that is generic over `vm_primitives::Pixel`
//! (sealed over `u8`/`u16`/`f32`) is bound the same way here: [`AnyImage`] holds
//! whichever concrete `Image<P>` the input array's dtype selected, and
//! [`with_any_image!`] runs one generic function body against it — the same
//! trick as matching on a closed enum, since a generic closure over `P` cannot
//! be written directly in stable Rust. An unsupported dtype is a `ValueError`
//! naming the three that are.

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vm_primitives::Image;

/// Convert a 2-D numpy array of element type `T` to an owned `Image<T>`.
///
/// The array must be C-contiguous (row-major). Returns a `PyErr` if the array
/// is not 2-D or not C-contiguous.
pub fn image_from_numpy<'py, T: numpy::Element + Clone>(
    _py: Python<'py>,
    array: &PyReadonlyArray2<'py, T>,
) -> PyResult<Image<T>> {
    let shape = array.shape();
    let h = shape[0];
    let w = shape[1];
    let slice = array
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("array not C-contiguous: {e}")))?;
    let data = slice.to_vec();
    Image::from_vec(w, h, data).map_err(|e| PyValueError::new_err(format!("image error: {e}")))
}

/// Convert a 2-D numpy `uint8` array to an owned `Image<u8>`.
pub fn image_from_numpy_u8(py: Python<'_>, array: &PyReadonlyArray2<u8>) -> PyResult<Image<u8>> {
    image_from_numpy(py, array)
}

/// An image of one of the three dtypes every generic detector accepts.
pub enum AnyImage {
    U8(Image<u8>),
    U16(Image<u16>),
    F32(Image<f32>),
}

/// Dispatch a numpy array of unknown dtype to the matching `Image<P>` variant.
///
/// Tries `uint8`, then `uint16`, then `float32`; numpy's own `FromPyObject`
/// for `PyReadonlyArray2<T>` does the dtype check, so this only has to try
/// each in turn. Anything else is a `ValueError` naming the three that work.
pub fn any_image_from_numpy<'py>(py: Python<'py>, img: &Bound<'py, PyAny>) -> PyResult<AnyImage> {
    if let Ok(a) = img.extract::<PyReadonlyArray2<'py, u8>>() {
        return Ok(AnyImage::U8(image_from_numpy(py, &a)?));
    }
    if let Ok(a) = img.extract::<PyReadonlyArray2<'py, u16>>() {
        return Ok(AnyImage::U16(image_from_numpy(py, &a)?));
    }
    if let Ok(a) = img.extract::<PyReadonlyArray2<'py, f32>>() {
        return Ok(AnyImage::F32(image_from_numpy(py, &a)?));
    }
    Err(PyValueError::new_err(
        "unsupported array dtype: expected uint8, uint16 or float32",
    ))
}

/// Run `$body` with `$view` bound to the `ImageView<'_, P>` inside an
/// [`AnyImage`], once per pixel type.
///
/// `$body` is written once in source but expanded three times — each
/// expansion is monomorphized independently against its own concrete `P`,
/// which is what lets this stand in for a generic closure over `Pixel`.
macro_rules! with_any_image {
    ($any:expr, $view:ident => $body:expr) => {
        match $any {
            $crate::convert::AnyImage::U8(img) => {
                let $view = img.as_view();
                $body
            }
            $crate::convert::AnyImage::U16(img) => {
                let $view = img.as_view();
                $body
            }
            $crate::convert::AnyImage::F32(img) => {
                let $view = img.as_view();
                $body
            }
        }
    };
}
pub(crate) use with_any_image;
