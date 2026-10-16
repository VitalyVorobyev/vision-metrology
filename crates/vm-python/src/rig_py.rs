//! Python bindings for `metric::rig`: planar rig substrate poses, numpy in and out.
//!
//! A sequence is passed whole: per step and view arrays, step-major. A step-view whose
//! `d_px` row holds a NaN is a missing shift. Poses are `(tx, ty, theta)` rows.

use numpy::ndarray::{Array, IxDyn};
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use vision_metrology::metric::rig;
use vm_primitives::{Isometry2f, Point2f, Projective2f, Vec2f};

use crate::config::{RigPoseConfig, RigSmootherConfig};

type Arr = Py<PyArrayDyn<f64>>;

fn array(py: Python<'_>, shape: &[usize], data: Vec<f64>) -> Arr {
    Array::from_shape_vec(IxDyn(shape), data)
        .expect("shape matches the data")
        .into_pyarray(py)
        .unbind()
}

fn pose_row(p: &Isometry2f) -> [f64; 3] {
    [
        p.translation.x as f64,
        p.translation.y as f64,
        p.rotation.angle() as f64,
    ]
}

fn opt(v: Option<f32>) -> f64 {
    v.map_or(f64::NAN, f64::from)
}

fn shaped<'a, T: numpy::Element>(
    a: &'a PyReadonlyArrayDyn<'_, T>,
    name: &str,
    shape: &[usize],
) -> PyResult<&'a [T]> {
    if a.shape() != shape {
        return Err(PyValueError::new_err(format!(
            "{name} must be shape {shape:?}, got {:?}",
            a.shape()
        )));
    }
    a.as_slice()
        .map_err(|e| PyValueError::new_err(format!("{name} not C-contiguous: {e}")))
}

fn native_err(e: vm_primitives::Error) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn path_points(external: &PyReadonlyArrayDyn<'_, f64>, frames: usize) -> PyResult<Vec<Point2f>> {
    let e = shaped(external, "external", &[frames, 2])?;
    Ok(e.as_chunks::<2>()
        .0
        .iter()
        .map(|p| Point2f::new(p[0] as f32, p[1] as f32))
        .collect())
}

/// A rig sequence's poses with their provenance. Per-frame arrays have one row per
/// frame (the anchor first); per-view arrays one row per step.
#[pyclass(skip_from_py_object)]
pub struct RigTrajectory {
    inner: rig::RigTrajectory,
}

#[pymethods]
impl RigTrajectory {
    /// `(F, 3)`: substrate→plane `(tx, ty, theta)`.
    #[getter]
    fn poses(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        array(
            py,
            &[f.len(), 3],
            f.iter().flat_map(|f| pose_row(&f.pose)).collect(),
        )
    }

    /// `(F, 3, 3)`: pose covariance over `(tx, ty, theta)`.
    #[getter]
    fn covariances(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        let data = f
            .iter()
            .flat_map(|f| {
                f.covariance
                    .transpose()
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>()
            })
            .collect();
        array(py, &[f.len(), 3, 3], data)
    }

    /// `(F, 3)`: the plane-frame increment from the previous frame.
    #[getter]
    fn increments(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        array(
            py,
            &[f.len(), 3],
            f.iter().flat_map(|f| pose_row(&f.increment)).collect(),
        )
    }

    /// `(F, 3, 3)`: increment covariance.
    #[getter]
    fn increment_covariances(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        let data = f
            .iter()
            .flat_map(|f| {
                f.increment_covariance
                    .transpose()
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>()
            })
            .collect();
        array(py, &[f.len(), 3, 3], data)
    }

    /// Per frame: `'anchor'`, `'measured'`, `'single_view'` or `'bridged'`.
    #[getter]
    fn steps(&self) -> Vec<&'static str> {
        self.inner.frames.iter().map(|f| f.step.as_str()).collect()
    }

    #[getter]
    fn views_used(&self) -> Vec<usize> {
        self.inner.frames.iter().map(|f| f.views_used).collect()
    }

    /// `(F,)`: RMS used-view residual, mm; NaN without redundancy.
    #[getter]
    fn residual_rms_mm(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        array(
            py,
            &[f.len()],
            f.iter().map(|f| opt(f.residual_rms_mm)).collect(),
        )
    }

    #[getter]
    fn residual_max_mm(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        array(
            py,
            &[f.len()],
            f.iter().map(|f| opt(f.residual_max_mm)).collect(),
        )
    }

    /// `(F,)`: the residual scale the increment covariance was inflated by (≥ 1).
    #[getter]
    fn residual_scale(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        array(
            py,
            &[f.len()],
            f.iter().map(|f| f.residual_scale as f64).collect(),
        )
    }

    /// `(S, V)` outcome names: `'used'`, `'missing'`, `'not_accepted'`,
    /// `'closure_failed'`, `'unliftable'` or `'rejected'`.
    #[getter]
    fn view_outcomes(&self) -> Vec<Vec<&'static str>> {
        (1..self.inner.frames.len())
            .map(|k| {
                self.inner
                    .views(k)
                    .iter()
                    .map(|u| u.outcome.as_str())
                    .collect()
            })
            .collect()
    }

    /// `(S, V)`: robust weight, 0 unless used.
    #[getter]
    fn view_weights(&self, py: Python<'_>) -> Arr {
        self.per_view(py, |u| vec![u.weight as f64])
    }

    /// `(S, V)`: residual at the lift point, mm; NaN when not lifted or bridged.
    #[getter]
    fn view_residuals_mm(&self, py: Python<'_>) -> Arr {
        self.per_view(py, |u| vec![opt(u.residual_mm)])
    }

    /// `(S, V, 2)`: the lift point on the plane, mm; NaN when not lifted.
    #[getter]
    fn view_levers_mm(&self, py: Python<'_>) -> Arr {
        self.per_view(py, |u| {
            u.lever_mm
                .map_or(vec![f64::NAN; 2], |p| vec![p.x as f64, p.y as f64])
        })
    }

    /// `(S, V, 2)`: the lifted plane displacement, mm; NaN when not lifted.
    #[getter]
    fn view_displacements_mm(&self, py: Python<'_>) -> Arr {
        self.per_view(py, |u| {
            u.displacement_mm
                .map_or(vec![f64::NAN; 2], |d| vec![d.x as f64, d.y as f64])
        })
    }

    #[getter]
    fn measured(&self) -> usize {
        self.inner.stats.measured
    }
    #[getter]
    fn single_view(&self) -> usize {
        self.inner.stats.single_view
    }
    #[getter]
    fn bridged(&self) -> usize {
        self.inner.stats.bridged
    }
    #[getter]
    fn longest_bridge(&self) -> usize {
        self.inner.stats.longest_bridge
    }
    #[getter]
    fn residual_median_mm(&self) -> Option<f32> {
        self.inner.stats.residual_median_mm
    }
    #[getter]
    fn residual_max_overall_mm(&self) -> Option<f32> {
        self.inner.stats.residual_max_mm
    }

    /// Path length of substrate point `point` over that of `external` `(F, 2)`; `None`
    /// when the external path has zero length. A scale diagnostic.
    fn path_length_ratio(
        &self,
        point: (f32, f32),
        external: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<Option<f32>> {
        let ext = path_points(&external, self.inner.frames.len())?;
        self.inner
            .path_length_ratio(Point2f::new(point.0, point.1), &ext)
            .map_err(native_err)
    }
}

impl RigTrajectory {
    fn per_view(&self, py: Python<'_>, f: impl Fn(&rig::ViewUse) -> Vec<f64>) -> Arr {
        let steps = self.inner.frames.len().saturating_sub(1);
        let v = self.inner.view_count();
        let mut data = Vec::new();
        let mut width = 1;
        for k in 1..=steps {
            for u in self.inner.views(k) {
                let row = f(u);
                width = row.len();
                data.extend(row);
            }
        }
        if width == 1 {
            array(py, &[steps, v], data)
        } else {
            array(py, &[steps, v, width], data)
        }
    }
}

/// A smoothed rig sequence.
#[pyclass(skip_from_py_object)]
pub struct RigSmoothed {
    inner: rig::RigSmoothed,
}

#[pymethods]
impl RigSmoothed {
    /// `(F, 3)`: substrate→plane `(tx, ty, theta)` from the smoothed increments.
    #[getter]
    fn poses(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        array(
            py,
            &[f.len(), 3],
            f.iter().flat_map(|f| pose_row(&f.pose)).collect(),
        )
    }

    #[getter]
    fn increments(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        array(
            py,
            &[f.len(), 3],
            f.iter().flat_map(|f| pose_row(&f.increment)).collect(),
        )
    }

    #[getter]
    fn increment_covariances(&self, py: Python<'_>) -> Arr {
        let f = &self.inner.frames;
        let data = f
            .iter()
            .flat_map(|f| {
                f.increment_covariance
                    .transpose()
                    .iter()
                    .map(|&v| v as f64)
                    .collect::<Vec<_>>()
            })
            .collect();
        array(py, &[f.len(), 3, 3], data)
    }

    /// Per frame: the step's own increment was rejected by the gate.
    #[getter]
    fn gated(&self) -> Vec<bool> {
        self.inner.frames.iter().map(|f| f.gated).collect()
    }

    #[getter]
    fn gated_count(&self) -> usize {
        self.inner.gated
    }

    fn path_length_ratio(
        &self,
        point: (f32, f32),
        external: PyReadonlyArrayDyn<'_, f64>,
    ) -> PyResult<Option<f32>> {
        let ext = path_points(&external, self.inner.frames.len())?;
        self.inner
            .path_length_ratio(Point2f::new(point.0, point.1), &ext)
            .map_err(native_err)
    }
}

/// Planar rig substrate pose: one rigid SE(2) pose per frame from per-view translations.
#[pyclass]
pub struct RigPoseEstimator {
    inner: rig::RigPoseEstimator,
    config: RigPoseConfig,
}

#[pymethods]
impl RigPoseEstimator {
    #[new]
    #[pyo3(signature=(config=None))]
    fn new(config: Option<RigPoseConfig>) -> Self {
        Self {
            inner: rig::RigPoseEstimator::new(),
            config: config.unwrap_or_default(),
        }
    }

    /// `plane_to_image` `(V, 3, 3)`; `d_px`, `centroid` `(S, V, 2)`; `covariance`
    /// `(S, V, 3)` as `[xx, xy, yy]` px²; `accepted` `(S, V)` bool; `closure` `(S, V)`
    /// px, NaN where absent. All float64. A NaN in `d_px` marks a missing shift.
    #[pyo3(signature=(plane_to_image, d_px, covariance, centroid, accepted, closure=None))]
    fn estimate(
        &mut self,
        plane_to_image: PyReadonlyArrayDyn<'_, f64>,
        d_px: PyReadonlyArrayDyn<'_, f64>,
        covariance: PyReadonlyArrayDyn<'_, f64>,
        centroid: PyReadonlyArrayDyn<'_, f64>,
        accepted: PyReadonlyArrayDyn<'_, bool>,
        closure: Option<PyReadonlyArrayDyn<'_, f64>>,
    ) -> PyResult<RigTrajectory> {
        let hs = plane_to_image.shape();
        if hs.len() != 3 || hs[1] != 3 || hs[2] != 3 {
            return Err(PyValueError::new_err(format!(
                "plane_to_image must be shape (V, 3, 3), got {hs:?}"
            )));
        }
        let v = hs[0];
        let ds = d_px.shape();
        if ds.len() != 3 || ds[1] != v || ds[2] != 2 {
            return Err(PyValueError::new_err(format!(
                "d_px must be shape (S, {v}, 2), got {ds:?}"
            )));
        }
        let s = ds[0];
        let h = shaped(&plane_to_image, "plane_to_image", &[v, 3, 3])?;
        let d = shaped(&d_px, "d_px", &[s, v, 2])?;
        let cov = shaped(&covariance, "covariance", &[s, v, 3])?;
        let cen = shaped(&centroid, "centroid", &[s, v, 2])?;
        let acc = shaped(&accepted, "accepted", &[s, v])?;
        let clo = closure
            .as_ref()
            .map(|c| shaped(c, "closure", &[s, v]))
            .transpose()?;

        let views: Vec<rig::RigView> = h
            .as_chunks::<9>()
            .0
            .iter()
            .map(|m| rig::RigView {
                plane_to_image: Projective2f::from_matrix_unchecked(
                    nalgebra::Matrix3::from_row_slice(m).cast::<f32>(),
                ),
            })
            .collect();
        let shifts: Vec<Option<rig::ViewShift>> = (0..s * v)
            .map(|i| {
                let (dx, dy) = (d[2 * i], d[2 * i + 1]);
                (!(dx.is_nan() || dy.is_nan())).then(|| rig::ViewShift {
                    d_px: Vec2f::new(dx as f32, dy as f32),
                    covariance: [
                        cov[3 * i] as f32,
                        cov[3 * i + 1] as f32,
                        cov[3 * i + 2] as f32,
                    ],
                    centroid: Point2f::new(cen[2 * i] as f32, cen[2 * i + 1] as f32),
                    accepted: acc[i],
                    closure_px: clo.and_then(|c| (!c[i].is_nan()).then_some(c[i] as f32)),
                })
            })
            .collect();
        let cfg = self.config.to_native()?;
        let inner = self
            .inner
            .estimate(&views, &shifts, &cfg)
            .map_err(native_err)?;
        Ok(RigTrajectory { inner })
    }

    /// Rauch–Tung–Striebel smoothing of a trajectory's increments.
    #[pyo3(signature=(trajectory, config=None))]
    fn smooth(
        &mut self,
        trajectory: &RigTrajectory,
        config: Option<RigSmootherConfig>,
    ) -> PyResult<RigSmoothed> {
        let cfg = config.unwrap_or_default().to_native();
        let inner = self
            .inner
            .smooth(&trajectory.inner, &cfg)
            .map_err(native_err)?;
        Ok(RigSmoothed { inner })
    }
}
