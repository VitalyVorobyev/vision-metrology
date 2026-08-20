//! Python bindings for `warp`: build a `dst -> src` coordinate map once,
//! apply it to as many frames as needed.
//!
//! `from_fn` is not bound — an arbitrary per-pixel Python callback would pay
//! the GIL round-trip cost on every destination pixel, defeating the point
//! of precomputing the map once. `affine` and `projective` take the
//! `dst -> src` matrix as a `(3, 3)` `float32` numpy array (row-major,
//! homogeneous); `projective` accepts a true homography, `affine` requires
//! its bottom row to be `[0, 0, 1]` (any other bottom row is a `ValueError`
//! — that is what distinguishes the two).

use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use vision_metrology::warp::{Interp as NativeInterp, Map as NativeMap};
use vision_metrology::{Affine2f, BorderMode, Pixel, Point2f, Projective2f};

use crate::convert::{any_image_from_numpy, with_any_image};

fn matrix3_from_numpy(arr: &PyReadonlyArray2<'_, f32>) -> PyResult<nalgebra::Matrix3<f32>> {
    if arr.shape() != [3, 3] {
        return Err(PyValueError::new_err(format!(
            "matrix must be shape (3, 3), got {:?}",
            arr.shape()
        )));
    }
    let slice = arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("matrix not C-contiguous: {e}")))?;
    // `Matrix3::from_row_slice` reads row-major, matching numpy's default
    // C-contiguous layout — no transpose needed.
    Ok(nalgebra::Matrix3::from_row_slice(slice))
}

fn parse_interp(interp: &str) -> PyResult<NativeInterp> {
    match interp {
        "nearest" => Ok(NativeInterp::Nearest),
        "bilinear" => Ok(NativeInterp::Bilinear),
        other => Err(PyValueError::new_err(format!(
            "interp must be 'nearest' or 'bilinear', got '{other}'"
        ))),
    }
}

fn parse_border<P: Pixel>(border_mode: &str, border_constant: f32) -> PyResult<BorderMode<P>> {
    match border_mode {
        "clamp" => Ok(BorderMode::Clamp),
        "reflect101" => Ok(BorderMode::Reflect101),
        "constant" => Ok(BorderMode::Constant(P::from_f32_sat(border_constant))),
        other => Err(PyValueError::new_err(format!(
            "border_mode must be 'clamp', 'reflect101' or 'constant', got '{other}'"
        ))),
    }
}

/// A precomputed `dst -> src` coordinate map — build once with
/// :meth:`affine`, :meth:`projective` or :meth:`polar`, then call
/// :meth:`apply` per frame.
///
/// Mirrors `vision_metrology::warp::Map`: every builder states the mapping
/// from a **destination** pixel to the **source** coordinate it samples —
/// see the Rust module docs for why, and for how to invert a forward
/// (source-into-destination) transform first.
#[pyclass(skip_from_py_object)]
#[derive(Clone)]
pub struct Map {
    inner: NativeMap,
}

impl Map {
    /// Wrap an already-built native map (e.g. from
    /// `ShapeMatch::model_frame_map`) without going through a Python
    /// constructor.
    pub(crate) fn from_native(inner: NativeMap) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl Map {
    /// Build from a `dst -> src` affine transform: a `(3, 3)` `float32`
    /// array whose bottom row is `[0, 0, 1]`.
    #[staticmethod]
    pub fn affine(w: usize, h: usize, matrix: PyReadonlyArray2<'_, f32>) -> PyResult<Self> {
        let m3 = matrix3_from_numpy(&matrix)?;
        let bottom = m3.row(2);
        if (bottom[0], bottom[1], bottom[2]) != (0.0, 0.0, 1.0) {
            return Err(PyValueError::new_err(
                "affine matrix's bottom row must be [0, 0, 1]; use Map.projective for a \
                 general homography",
            ));
        }
        let m = Affine2f::from_matrix_unchecked(m3);
        Ok(Self {
            inner: NativeMap::affine(w, h, &m),
        })
    }

    /// Build from a `dst -> src` homography: a `(3, 3)` `float32` array.
    #[staticmethod]
    pub fn projective(w: usize, h: usize, matrix: PyReadonlyArray2<'_, f32>) -> PyResult<Self> {
        let m3 = matrix3_from_numpy(&matrix)?;
        let h_ = Projective2f::from_matrix_unchecked(m3);
        Ok(Self {
            inner: NativeMap::projective(w, h, &h_),
        })
    }

    /// Build a polar-unwrap map: destination x sweeps `phi` (radians),
    /// destination y sweeps `r` (pixels), both linearly at bin centers —
    /// see the Rust `Map::polar` docs for the exact formula.
    #[staticmethod]
    #[pyo3(signature = (center, r, phi, w, h))]
    pub fn polar(center: (f32, f32), r: (f32, f32), phi: (f32, f32), w: usize, h: usize) -> Self {
        Self {
            inner: NativeMap::polar(
                Point2f::new(center.0, center.1),
                r.0..r.1,
                phi.0..phi.1,
                w,
                h,
            ),
        }
    }

    /// Destination width in pixels.
    #[getter]
    pub fn width(&self) -> usize {
        self.inner.width()
    }

    /// Destination height in pixels.
    #[getter]
    pub fn height(&self) -> usize {
        self.inner.height()
    }

    /// Resample `img` (`uint8`/`uint16`/`float32`) through this map,
    /// returning a same-dtype `(height, width)` array. Out-of-source taps
    /// are filled per `border_mode`; see :meth:`apply_with_mask` to also
    /// learn which pixels that happened to.
    #[pyo3(signature = (img, interp="bilinear", border_mode="clamp", border_constant=0.0))]
    pub fn apply<'py>(
        &self,
        py: Python<'py>,
        img: &Bound<'py, PyAny>,
        interp: &str,
        border_mode: &str,
        border_constant: f32,
    ) -> PyResult<Py<PyAny>> {
        let any = any_image_from_numpy(py, img)?;
        let interp = parse_interp(interp)?;
        let (w, h) = (self.inner.width(), self.inner.height());

        with_any_image!(any, view => {
            let border = parse_border(border_mode, border_constant)?;
            let mut dst = vec![Default::default(); w * h];
            self.inner
                .apply(&view, &mut dst, interp, border)
                .map_err(|e| PyValueError::new_err(format!("warp apply error: {e}")))?;
            let arr = numpy::ndarray::Array2::from_shape_vec((h, w), dst)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            Ok(arr.into_pyarray(py).into_any().unbind())
        })
    }

    /// Same as :meth:`apply`, additionally returning a `(height, width)`
    /// `uint8` validity mask: `255` where every interpolation tap fell
    /// inside `img`, `0` otherwise. Returns `(image, mask)`.
    #[pyo3(signature = (img, interp="bilinear", border_mode="clamp", border_constant=0.0))]
    #[allow(clippy::type_complexity)]
    pub fn apply_with_mask<'py>(
        &self,
        py: Python<'py>,
        img: &Bound<'py, PyAny>,
        interp: &str,
        border_mode: &str,
        border_constant: f32,
    ) -> PyResult<(Py<PyAny>, Py<PyArray2<u8>>)> {
        let any = any_image_from_numpy(py, img)?;
        let interp = parse_interp(interp)?;
        let (w, h) = (self.inner.width(), self.inner.height());

        with_any_image!(any, view => {
            let border = parse_border(border_mode, border_constant)?;
            let mut dst = vec![Default::default(); w * h];
            let mut mask = vec![0u8; w * h];
            self.inner
                .apply_with_mask(&view, &mut dst, &mut mask, interp, border)
                .map_err(|e| PyValueError::new_err(format!("warp apply error: {e}")))?;
            let arr = numpy::ndarray::Array2::from_shape_vec((h, w), dst)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            let mask_arr = numpy::ndarray::Array2::from_shape_vec((h, w), mask)
                .map_err(|e| PyValueError::new_err(format!("{e}")))?;
            Ok((arr.into_pyarray(py).into_any().unbind(), mask_arr.into_pyarray(py).unbind()))
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Map(width={}, height={})",
            self.inner.width(),
            self.inner.height()
        )
    }
}
