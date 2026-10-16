//! Python binding for `vm_primitives::edge::ProfileSampler`.

use numpy::ndarray::{Array2, ArrayView2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use vm_primitives::edge::{ProfileSampler as NativeSampler, ProfileSamplerConfig, ProfileStation};
use vm_primitives::{Point2f, Vec2f};

use crate::convert::{any_image_from_numpy, image_from_numpy_u8, with_any_image};

/// `(N, 2)` `float32` rows, in any memory layout.
fn pairs(a: ArrayView2<'_, f32>, name: &str) -> PyResult<Vec<(f32, f32)>> {
    if a.ncols() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (N, 2), got {:?}",
            a.shape()
        )));
    }
    Ok(a.rows().into_iter().map(|r| (r[0], r[1])).collect())
}

/// The three `(N, M)` arrays `ProfileSampler.sample` returns.
type Profiles<'py> = (
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<bool>>,
);

/// Continuous derivative-of-Gaussian profiles along arbitrary directions — mirrors
/// `vm_primitives::edge::ProfileSampler` with its config's `sigma`.
#[pyclass]
pub struct ProfileSampler {
    inner: NativeSampler,
    sigma: f32,
}

fn checked_sigma(sigma: f32) -> PyResult<f32> {
    if sigma.is_finite() && sigma > 0.0 {
        Ok(sigma)
    } else {
        Err(PyValueError::new_err(format!(
            "sigma must be finite and positive, got {sigma}"
        )))
    }
}

#[pymethods]
impl ProfileSampler {
    #[new]
    #[pyo3(signature = (sigma = ProfileSamplerConfig::default().sigma))]
    fn new(sigma: f32) -> PyResult<Self> {
        Ok(Self {
            inner: NativeSampler::new(),
            sigma: checked_sigma(sigma)?,
        })
    }

    /// Gaussian σ in pixels; the support reaches 4σ from each sample.
    #[getter]
    fn sigma(&self) -> f32 {
        self.sigma
    }

    #[setter]
    fn set_sigma(&mut self, sigma: f32) -> PyResult<()> {
        self.sigma = checked_sigma(sigma)?;
        Ok(())
    }

    /// Sample station `i` at `points[i] + offsets[j] · normals[i]` (normalised).
    ///
    /// Returns `(value, derivative, valid)`, each `(N, M)`: the smoothed intensity, its
    /// derivative along the normal (input units per pixel), and whether the sample's
    /// support lay inside the image and the mask (non-zero = valid). Invalid samples
    /// read NaN.
    #[pyo3(signature = (img, points, normals, offsets, mask = None))]
    fn sample<'py>(
        &mut self,
        py: Python<'py>,
        img: &Bound<'py, PyAny>,
        points: PyReadonlyArray2<'py, f32>,
        normals: PyReadonlyArray2<'py, f32>,
        offsets: PyReadonlyArray1<'py, f32>,
        mask: Option<PyReadonlyArray2<'py, u8>>,
    ) -> PyResult<Profiles<'py>> {
        let p = pairs(points.as_array(), "points")?;
        let n = pairs(normals.as_array(), "normals")?;
        if p.len() != n.len() {
            return Err(PyValueError::new_err(format!(
                "points and normals must have the same length, got {} and {}",
                p.len(),
                n.len()
            )));
        }
        let stations: Vec<ProfileStation> = p
            .iter()
            .zip(&n)
            .map(|(&(x, y), &(nx, ny))| ProfileStation {
                p: Point2f::new(x, y),
                n: Vec2f::new(nx, ny),
            })
            .collect();
        let offsets: Vec<f32> = offsets.as_array().iter().copied().collect();
        let mask = mask.map(|m| image_from_numpy_u8(py, &m)).transpose()?;
        let any = any_image_from_numpy(py, img)?;
        let cfg = ProfileSamplerConfig { sigma: self.sigma };
        let mut out = vec![None; stations.len() * offsets.len()];
        let sampler = &mut self.inner;
        with_any_image!(any, view => sampler.sample(
            &view,
            mask.as_ref().map(|m| m.as_view()).as_ref(),
            &stations,
            &offsets,
            &cfg,
            &mut out,
        ))
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let shape = (stations.len(), offsets.len());
        let value = out
            .iter()
            .map(|s| s.map_or(f32::NAN, |s| s.value))
            .collect();
        let derivative = out
            .iter()
            .map(|s| s.map_or(f32::NAN, |s| s.derivative))
            .collect();
        let valid = out.iter().map(Option::is_some).collect();
        let array =
            |v| Array2::from_shape_vec(shape, v).map_err(|e| PyValueError::new_err(e.to_string()));
        Ok((
            array(value)?.into_pyarray(py),
            array(derivative)?.into_pyarray(py),
            Array2::from_shape_vec(shape, valid)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
                .into_pyarray(py),
        ))
    }

    fn __repr__(&self) -> String {
        format!("ProfileSampler(sigma={})", self.sigma)
    }
}
