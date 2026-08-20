//! Python bindings for `metric`: the calibration bridge.
//!
//! **`Pose3` is not a Python class.** Every function here that takes or
//! returns a camera-from-reference pose uses a plain `(4, 4)` `float64`
//! numpy array (row-major, homogeneous: top-left 3x3 rotation, last column
//! translation in millimetres) — the numpy-friendly representation the
//! roadmap plan calls for, and the shape every other pose-shaped array in
//! this codebase (calibration exports, robotics stacks) already uses.
//! [`pose_from_numpy`]/[`pose_to_numpy`] are the two conversion points.

use nalgebra::{Quaternion, Translation3, UnitQuaternion};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use vision_metrology::metric::{
    BrownConrady5 as NativeBrownConrady5, CameraModel as NativeCameraModel,
    PinholeIntrinsics as NativePinholeIntrinsics, Plane3 as NativePlane3,
    PlaneGrid as NativePlaneGrid, Pose3, distort_pixel, io,
};
use vision_metrology::{Point2f, Point3f, Vec3f};

use crate::warp_py::Map;

// ── Pose3 <-> numpy ─────────────────────────────────────────────────────

/// Parse a `(4, 4)` `float64` row-major homogeneous transform into a `Pose3`.
pub fn pose_from_numpy(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<Pose3> {
    use numpy::PyUntypedArrayMethods;
    if arr.shape() != [4, 4] {
        return Err(PyValueError::new_err(format!(
            "pose must be shape (4, 4), got {:?}",
            arr.shape()
        )));
    }
    let slice = arr
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("pose array not C-contiguous: {e}")))?;
    let m = nalgebra::Matrix4::from_row_slice(slice);
    let mut rot64 = nalgebra::Matrix3::<f64>::zeros();
    for i in 0..3 {
        for j in 0..3 {
            rot64[(i, j)] = m[(i, j)];
        }
    }
    let q64 = UnitQuaternion::from_matrix(&rot64);
    let rotation = UnitQuaternion::new_normalize(Quaternion::new(
        q64.scalar() as f32,
        q64.imag().x as f32,
        q64.imag().y as f32,
        q64.imag().z as f32,
    ));
    let translation = Translation3::new(m[(0, 3)] as f32, m[(1, 3)] as f32, m[(2, 3)] as f32);
    Ok(Pose3::from_parts(translation, rotation))
}

/// Serialize a `Pose3` to a `(4, 4)` `float64` row-major homogeneous
/// transform.
pub fn pose_to_numpy(py: Python<'_>, pose: &Pose3) -> Py<PyArray2<f64>> {
    let r = pose.rotation.to_rotation_matrix();
    let t = pose.translation.vector;
    let arr = numpy::ndarray::Array2::from_shape_fn((4, 4), |(i, j)| {
        if i < 3 && j < 3 {
            r.matrix()[(i, j)] as f64
        } else if i < 3 && j == 3 {
            match i {
                0 => t.x as f64,
                1 => t.y as f64,
                _ => t.z as f64,
            }
        } else if i == 3 && j == 3 {
            1.0
        } else {
            0.0
        }
    });
    arr.into_pyarray(py).unbind()
}

// ── Canonical types ─────────────────────────────────────────────────────

/// Pinhole camera intrinsics, in pixels.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PinholeIntrinsics {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub skew: f32,
}

#[pymethods]
impl PinholeIntrinsics {
    #[new]
    #[pyo3(signature = (fx, fy, cx, cy, skew=0.0))]
    fn new(fx: f32, fy: f32, cx: f32, cy: f32, skew: f32) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            skew,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PinholeIntrinsics(fx={:.3}, fy={:.3}, cx={:.3}, cy={:.3}, skew={:.3})",
            self.fx, self.fy, self.cx, self.cy, self.skew
        )
    }
}

impl From<PinholeIntrinsics> for NativePinholeIntrinsics {
    fn from(v: PinholeIntrinsics) -> Self {
        Self {
            fx: v.fx,
            fy: v.fy,
            cx: v.cx,
            cy: v.cy,
            skew: v.skew,
        }
    }
}

impl From<NativePinholeIntrinsics> for PinholeIntrinsics {
    fn from(v: NativePinholeIntrinsics) -> Self {
        Self {
            fx: v.fx,
            fy: v.fy,
            cx: v.cx,
            cy: v.cy,
            skew: v.skew,
        }
    }
}

/// Brown-Conrady 5-parameter radial-tangential distortion.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Copy, Default)]
pub struct BrownConrady5 {
    pub k1: f32,
    pub k2: f32,
    pub k3: f32,
    pub p1: f32,
    pub p2: f32,
}

#[pymethods]
impl BrownConrady5 {
    #[new]
    #[pyo3(signature = (k1=0.0, k2=0.0, k3=0.0, p1=0.0, p2=0.0))]
    fn new(k1: f32, k2: f32, k3: f32, p1: f32, p2: f32) -> Self {
        Self { k1, k2, k3, p1, p2 }
    }

    fn __repr__(&self) -> String {
        format!(
            "BrownConrady5(k1={:.4}, k2={:.4}, k3={:.4}, p1={:.4}, p2={:.4})",
            self.k1, self.k2, self.k3, self.p1, self.p2
        )
    }
}

impl From<BrownConrady5> for NativeBrownConrady5 {
    fn from(v: BrownConrady5) -> Self {
        Self {
            k1: v.k1,
            k2: v.k2,
            k3: v.k3,
            p1: v.p1,
            p2: v.p2,
        }
    }
}

impl From<NativeBrownConrady5> for BrownConrady5 {
    fn from(v: NativeBrownConrady5) -> Self {
        Self {
            k1: v.k1,
            k2: v.k2,
            k3: v.k3,
            p1: v.p1,
            p2: v.p2,
        }
    }
}

/// A calibrated camera: intrinsics + distortion.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct CameraModel {
    pub intrinsics: PinholeIntrinsics,
    pub distortion: BrownConrady5,
}

#[pymethods]
impl CameraModel {
    #[new]
    #[pyo3(signature = (intrinsics, distortion=None))]
    fn new(intrinsics: PinholeIntrinsics, distortion: Option<BrownConrady5>) -> Self {
        Self {
            intrinsics,
            distortion: distortion.unwrap_or_default(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "CameraModel(intrinsics={}, distortion={})",
            self.intrinsics.__repr__(),
            self.distortion.__repr__()
        )
    }
}

impl From<CameraModel> for NativeCameraModel {
    fn from(v: CameraModel) -> Self {
        Self {
            intrinsics: v.intrinsics.into(),
            distortion: v.distortion.into(),
        }
    }
}

impl From<NativeCameraModel> for CameraModel {
    fn from(v: NativeCameraModel) -> Self {
        Self {
            intrinsics: v.intrinsics.into(),
            distortion: v.distortion.into(),
        }
    }
}

/// An oriented plane in the reference frame: `n . X + d = 0`.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct Plane3 {
    pub n: (f32, f32, f32),
    pub d: f32,
}

#[pymethods]
impl Plane3 {
    #[new]
    fn new(n: (f32, f32, f32), d: f32) -> Self {
        Self { n, d }
    }

    /// The reference frame's `z = 0` plane — the default `plane_grid_map`
    /// and `homography_plane_to_image` assume.
    #[staticmethod]
    fn xy() -> Self {
        NativePlane3::xy().into()
    }

    fn __repr__(&self) -> String {
        format!("Plane3(n={:?}, d={:.3})", self.n, self.d)
    }
}

impl From<Plane3> for NativePlane3 {
    fn from(v: Plane3) -> Self {
        Self {
            n: Vec3f::new(v.n.0, v.n.1, v.n.2),
            d: v.d,
        }
    }
}

impl From<NativePlane3> for Plane3 {
    fn from(v: NativePlane3) -> Self {
        Self {
            n: (v.n.x, v.n.y, v.n.z),
            d: v.d,
        }
    }
}

/// A metric raster on the reference frame's `z = 0` plane.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Debug, Clone, Copy)]
pub struct PlaneGrid {
    pub origin_mm: (f32, f32),
    pub mm_per_px: f32,
    pub w: usize,
    pub h: usize,
}

#[pymethods]
impl PlaneGrid {
    #[new]
    fn new(origin_mm: (f32, f32), mm_per_px: f32, w: usize, h: usize) -> Self {
        Self {
            origin_mm,
            mm_per_px,
            w,
            h,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "PlaneGrid(origin_mm={:?}, mm_per_px={:.4}, w={}, h={})",
            self.origin_mm, self.mm_per_px, self.w, self.h
        )
    }
}

impl From<PlaneGrid> for NativePlaneGrid {
    fn from(v: PlaneGrid) -> Self {
        Self {
            origin_mm: Point2f::new(v.origin_mm.0, v.origin_mm.1),
            mm_per_px: v.mm_per_px,
            w: v.w,
            h: v.h,
        }
    }
}

// ── Functions ────────────────────────────────────────────────────────────

/// Project pixels onto `plane` under `camera`/`pose`, vectorized.
///
/// `pixels` is an `(N, 2)` `float32` array of `(x, y)` pixel coordinates.
/// Returns an `(N, 2)` `float64` array of plane-frame `(x_mm, y_mm)`
/// coordinates; a pixel whose ray misses the plane (parallel, or behind the
/// camera) gets a `NaN` row rather than raising, so a batch never fails as a
/// whole for one bad point.
#[pyfunction]
pub fn pixel_to_plane<'py>(
    py: Python<'py>,
    camera: CameraModel,
    pose: PyReadonlyArray2<'py, f64>,
    plane: Plane3,
    pixels: PyReadonlyArray2<'py, f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    use numpy::PyUntypedArrayMethods;
    let camera: NativeCameraModel = camera.into();
    let pose = pose_from_numpy(&pose)?;
    let plane: NativePlane3 = plane.into();

    if pixels.shape().len() != 2 || pixels.shape()[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "pixels must be shape (N, 2), got {:?}",
            pixels.shape()
        )));
    }
    let slice = pixels
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("pixels array not C-contiguous: {e}")))?;
    let n = pixels.shape()[0];

    // Pure-Rust per-point loop, no Python calls inside — computed with the
    // GIL held, same convention as every other detector in this crate (see
    // `convert.rs`'s "GIL strategy" note).
    let mut out = vec![0.0f64; n * 2];
    for i in 0..n {
        let px = Point2f::new(slice[2 * i], slice[2 * i + 1]);
        match vision_metrology::metric::pixel_to_plane(&camera, &pose, &plane, px) {
            Some(p) => {
                out[2 * i] = p.x as f64;
                out[2 * i + 1] = p.y as f64;
            }
            None => {
                out[2 * i] = f64::NAN;
                out[2 * i + 1] = f64::NAN;
            }
        }
    }

    let arr = numpy::ndarray::Array2::from_shape_vec((n, 2), out)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(arr.into_pyarray(py).unbind())
}

/// The runtime bird's-eye `dst -> src` map: destination is `grid` pixel
/// coordinates, source is the raw (distorted) camera image. Composes the
/// ideal plane->image homography with forward Brown-Conrady distortion.
#[pyfunction]
pub fn plane_grid_map<'py>(
    camera: CameraModel,
    pose: PyReadonlyArray2<'py, f64>,
    grid: PlaneGrid,
) -> PyResult<Map> {
    let camera: NativeCameraModel = camera.into();
    let pose = pose_from_numpy(&pose)?;
    let grid: NativePlaneGrid = grid.into();
    let native = vision_metrology::metric::plane_grid_map(&camera, &pose, &grid);
    Ok(Map::from_native(native))
}

/// Build a `dst -> src` undistortion map for a `w x h` image shot by
/// `camera`.
#[pyfunction]
pub fn undistort_map(camera: CameraModel, w: usize, h: usize) -> Map {
    let camera: NativeCameraModel = camera.into();
    let native = vision_metrology::metric::undistort_map(&camera, w, h);
    Map::from_native(native)
}

/// Project points on the reference frame's `z = 0` plane into a camera's
/// raw (distorted) pixel space: `pose * (x_mm, y_mm, 0)`, perspective
/// divide, then [`distort_pixel`](vision_metrology::metric::distort_pixel) —
/// the exact forward geometry [`plane_grid_map`] composes internally,
/// exposed pointwise/vectorized here for the lab's mosaic compositing.
///
/// Choosing which camera "owns" a destination grid pixel under a
/// nearest-camera-centre priority rule needs, for *every* candidate camera,
/// where that plane point would land in that camera's own raw pixel space —
/// not just the one source pixel `plane_grid_map` samples for whichever
/// camera the caller already picked. This is the tiny binding that makes
/// that computable in Python without duplicating the distortion formula
/// there (see `lab/backend/src/vm_lab/routers/mosaic.py`).
///
/// `points_mm` is an `(N, 2)` `float32` array of `(x_mm, y_mm)` reference-
/// frame plane coordinates. Returns an `(N, 2)` `float64` array of raw
/// pixel coordinates; a point behind the camera (`z <= 0`) gets a `NaN` row.
#[pyfunction]
pub fn project_plane_points<'py>(
    py: Python<'py>,
    camera: CameraModel,
    pose: PyReadonlyArray2<'py, f64>,
    points_mm: PyReadonlyArray2<'py, f32>,
) -> PyResult<Py<PyArray2<f64>>> {
    use numpy::PyUntypedArrayMethods;
    let camera: NativeCameraModel = camera.into();
    let pose = pose_from_numpy(&pose)?;

    if points_mm.shape().len() != 2 || points_mm.shape()[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "points_mm must be shape (N, 2), got {:?}",
            points_mm.shape()
        )));
    }
    let slice = points_mm
        .as_slice()
        .map_err(|e| PyValueError::new_err(format!("points_mm array not C-contiguous: {e}")))?;
    let n = points_mm.shape()[0];

    // Pure-Rust per-point loop, no Python calls inside — same convention as
    // `pixel_to_plane` above.
    let mut out = vec![0.0f64; n * 2];
    for i in 0..n {
        let p_ref = Point3f::new(slice[2 * i], slice[2 * i + 1], 0.0);
        let p_cam = pose * p_ref;
        if p_cam.z <= 0.0 {
            out[2 * i] = f64::NAN;
            out[2 * i + 1] = f64::NAN;
            continue;
        }
        let normalized = Point2f::new(p_cam.x / p_cam.z, p_cam.y / p_cam.z);
        let px = distort_pixel(&camera, normalized);
        out[2 * i] = px.x as f64;
        out[2 * i + 1] = px.y as f64;
    }

    let arr = numpy::ndarray::Array2::from_shape_vec((n, 2), out)
        .map_err(|e| PyValueError::new_err(format!("{e}")))?;
    Ok(arr.into_pyarray(py).unbind())
}

fn bytes_arg(py: Python<'_>, source: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(b) = source.cast::<PyBytes>() {
        return Ok(b.as_bytes().to_vec());
    }
    if let Ok(path) = source.extract::<String>() {
        return std::fs::read(&path)
            .map_err(|e| PyIOError::new_err(format!("could not read {path}: {e}")));
    }
    let _ = py;
    Err(PyValueError::new_err(
        "expected a file path (str) or raw JSON bytes",
    ))
}

fn cameras_to_py(
    py: Python<'_>,
    pairs: Vec<(NativeCameraModel, Pose3)>,
) -> Vec<(CameraModel, Py<PyArray2<f64>>)> {
    pairs
        .into_iter()
        .map(|(cam, pose)| (cam.into(), pose_to_numpy(py, &pose)))
        .collect()
}

/// Load a calibration-rs `RigExtrinsicsExport` JSON document (path or raw
/// bytes). Returns a list of `(CameraModel, pose)` pairs, one per camera,
/// `pose` a `(4, 4)` `float64` camera-from-reference transform in
/// millimetres.
#[pyfunction]
pub fn load_rig_extrinsics(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
) -> PyResult<Vec<(CameraModel, Py<PyArray2<f64>>)>> {
    let bytes = bytes_arg(py, source)?;
    let pairs = io::import_rig_extrinsics(&bytes)
        .map_err(|e| PyValueError::new_err(format!("load_rig_extrinsics: {e}")))?;
    Ok(cameras_to_py(py, pairs))
}

/// Load a `table_calibration` `calibration.json` document (path or raw
/// bytes). Returns a list of `(CameraModel, pose)` pairs sorted by camera
/// index, `pose` a `(4, 4)` `float64` camera-from-reference transform in
/// millimetres.
#[pyfunction]
pub fn load_table_calibration(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
) -> PyResult<Vec<(CameraModel, Py<PyArray2<f64>>)>> {
    let bytes = bytes_arg(py, source)?;
    let pairs = io::import_table_calibration(&bytes)
        .map_err(|e| PyValueError::new_err(format!("load_table_calibration: {e}")))?;
    Ok(cameras_to_py(py, pairs))
}
