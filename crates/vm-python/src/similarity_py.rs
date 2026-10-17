//! Masked local similarity registration, with native reusable scratch.
use crate::config::SimilarityConfig;
use crate::convert::{AnyImage, any_image_from_numpy, image_from_numpy};
use numpy::PyReadonlyArray2;
use pyo3::{exceptions::PyValueError, prelude::*};
use vision_metrology::corr;
use vm_primitives::{Similarity2f, Vec2f, similarity_parts};

/// Previous-to-current (tx, ty, angle_radians, scale), and diagnostics.
#[pyclass(get_all, skip_from_py_object)]
pub struct Similarity {
    pub transform: Option<(f32, f32, f32, f32)>,
    pub score: Option<f32>,
    pub margin: Option<f32>,
    pub support: usize,
    pub residual_rms: Option<f32>,
    pub residual_max: Option<f32>,
    pub gain: Option<f32>,
    pub offset: Option<f32>,
    pub covariance: Option<[[f32; 4]; 4]>,
    pub regions: usize,
    pub region_error: Option<f32>,
    pub region_chi2: Option<f32>,
    pub status: &'static str,
}
impl From<corr::Similarity> for Similarity {
    fn from(n: corr::Similarity) -> Self {
        Self {
            transform: n.transform.map(|p| {
                let (t, a, s) = similarity_parts(&p);
                (t.x, t.y, a, s)
            }),
            score: n.score,
            margin: n.margin,
            support: n.support,
            residual_rms: n.residual_rms,
            residual_max: n.residual_max,
            gain: n.gain,
            offset: n.offset,
            covariance: n.covariance,
            regions: n.regions,
            region_error: n.region_error,
            region_chi2: n.region_chi2,
            status: n.status.as_str(),
        }
    }
}

/// Reusable native masked similarity estimator. Python inputs are copied by
/// the usual numpy conversion; native scratch is reused across calls.
#[pyclass]
pub struct SimilarityEstimator {
    inner: corr::SimilarityEstimator,
    config: SimilarityConfig,
}

#[pymethods]
impl SimilarityEstimator {
    #[new]
    #[pyo3(signature=(config=None))]
    fn new(config: Option<SimilarityConfig>) -> Self {
        Self {
            inner: corr::SimilarityEstimator::new(),
            config: config.unwrap_or_default(),
        }
    }

    /// Images must share dtype (uint8/uint16/float32). Masks are uint8;
    /// nonzero pixels are valid. Prediction is (tx, ty, angle_radians, positive_scale).
    #[pyo3(signature=(previous,current,*,previous_mask=None,current_mask=None,prediction=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        previous: &Bound<'_, PyAny>,
        current: &Bound<'_, PyAny>,
        previous_mask: Option<PyReadonlyArray2<'_, u8>>,
        current_mask: Option<PyReadonlyArray2<'_, u8>>,
        prediction: Option<(f32, f32, f32, f32)>,
    ) -> PyResult<Similarity> {
        let a = any_image_from_numpy(py, previous)?;
        let b = any_image_from_numpy(py, current)?;
        let am = previous_mask
            .as_ref()
            .map(|m| image_from_numpy(py, m))
            .transpose()?;
        let bm = current_mask
            .as_ref()
            .map(|m| image_from_numpy(py, m))
            .transpose()?;
        let av = am.as_ref().map(|m| m.as_view());
        let bv = bm.as_ref().map(|m| m.as_view());
        if prediction.is_some_and(|(x, y, angle, scale)| {
            ![x, y, angle, scale].iter().all(|v| v.is_finite()) || scale <= 0.0
        }) {
            return Err(PyValueError::new_err(
                "similarity prediction must be finite with positive scale",
            ));
        }
        let p = prediction
            .map(|(x, y, angle, scale)| Similarity2f::new(Vec2f::new(x, y), angle, scale));
        let cfg = self.config.to_native()?;
        let result = match (&a, &b) {
            (AnyImage::U8(a), AnyImage::U8(b)) => self.inner.estimate(
                &a.as_view(),
                &b.as_view(),
                av.as_ref(),
                bv.as_ref(),
                p,
                &cfg,
            ),
            (AnyImage::U16(a), AnyImage::U16(b)) => self.inner.estimate(
                &a.as_view(),
                &b.as_view(),
                av.as_ref(),
                bv.as_ref(),
                p,
                &cfg,
            ),
            (AnyImage::F32(a), AnyImage::F32(b)) => self.inner.estimate(
                &a.as_view(),
                &b.as_view(),
                av.as_ref(),
                bv.as_ref(),
                p,
                &cfg,
            ),
            _ => {
                return Err(PyValueError::new_err(
                    "similarity images must have matching dtypes",
                ));
            }
        };
        result
            .map(Into::into)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
