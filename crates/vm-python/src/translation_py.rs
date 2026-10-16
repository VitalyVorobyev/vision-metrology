//! Masked registration, generic over the image dtype; Python image conversion
//! copies inputs, while the native estimator reuses its numerical scratch.

use numpy::PyReadonlyArray2;
use pyo3::{exceptions::PyValueError, prelude::*};
use vision_metrology::corr;
use vm_primitives::Vec2f;

use crate::config::TranslationConfig;
use crate::convert::{AnyImage, any_image_from_numpy, image_from_numpy};

/// Content motion and the evidence qualifying it. Only `status == 'accepted'`
/// permits using the shift as a fixture. Covariance is a local approximation.
#[pyclass(get_all, skip_from_py_object)]
pub struct Translation {
    pub shift: Option<(f32, f32)>,
    pub score: Option<f32>,
    pub margin: Option<f32>,
    pub support: usize,
    pub residual_rms: Option<f32>,
    pub residual_max: Option<f32>,
    pub gain: Option<f32>,
    pub offset: Option<f32>,
    pub covariance: Option<(f32, f32, f32)>,
    pub regions: usize,
    pub region_error: Option<f32>,
    pub status: &'static str,
}

impl From<corr::Translation> for Translation {
    fn from(n: corr::Translation) -> Self {
        Self {
            shift: n.shift.map(|p| (p.x, p.y)),
            score: n.score,
            margin: n.margin,
            support: n.support,
            residual_rms: n.residual_rms,
            residual_max: n.residual_max,
            gain: n.gain,
            offset: n.offset,
            covariance: n.covariance.map(|c| (c[0], c[1], c[2])),
            regions: n.regions,
            region_error: n.region_error,
            status: n.status.as_str(),
        }
    }
}

/// Reusable native masked translation estimator. Python inputs are copied by
/// the usual numpy conversion; native scratch is reused across calls.
#[pyclass]
pub struct TranslationEstimator {
    inner: corr::TranslationEstimator,
    config: TranslationConfig,
}

#[pymethods]
impl TranslationEstimator {
    #[new]
    #[pyo3(signature=(config=None))]
    fn new(config: Option<TranslationConfig>) -> Self {
        Self {
            inner: corr::TranslationEstimator::new(),
            config: config.unwrap_or_default(),
        }
    }

    /// Images must share dtype (uint8/uint16/float32). Masks are uint8;
    /// nonzero pixels are valid. Prediction centers the bounded search.
    #[pyo3(signature=(previous,current,*,previous_mask=None,current_mask=None,prediction=None))]
    fn estimate(
        &mut self,
        py: Python<'_>,
        previous: &Bound<'_, PyAny>,
        current: &Bound<'_, PyAny>,
        previous_mask: Option<PyReadonlyArray2<'_, u8>>,
        current_mask: Option<PyReadonlyArray2<'_, u8>>,
        prediction: Option<(f32, f32)>,
    ) -> PyResult<Translation> {
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
        let p = prediction.map(|(x, y)| Vec2f::new(x, y));
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
                    "translation images must have matching dtypes",
                ));
            }
        };
        result
            .map(Into::into)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
