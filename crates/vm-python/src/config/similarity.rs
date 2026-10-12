//! Python configuration for local masked similarity registration.
use super::TranslationConfig;
use pyo3::prelude::*;
use vision_metrology::corr::SimilarityConfig as NativeConfig;

/// Bounded local similarity near a prediction. Angles are radians; scale bound is log-scale.
#[pyclass(get_all, set_all, from_py_object)]
#[derive(Clone)]
pub struct SimilarityConfig {
    pub translation: TranslationConfig,
    pub max_angle: f32,
    pub max_log_scale: f32,
}
impl Default for SimilarityConfig {
    fn default() -> Self {
        let n = NativeConfig::default();
        let mut translation = TranslationConfig::default();
        translation.tuning.iterations = n.translation.tuning.iterations.get();
        Self {
            translation,
            max_angle: n.max_angle,
            max_log_scale: n.max_log_scale,
        }
    }
}
#[pymethods]
impl SimilarityConfig {
    #[new]
    #[pyo3(signature=(*,translation=None,max_angle=0.05,max_log_scale=0.05))]
    fn new(translation: Option<TranslationConfig>, max_angle: f32, max_log_scale: f32) -> Self {
        Self {
            translation: translation.unwrap_or_else(|| Self::default().translation),
            max_angle,
            max_log_scale,
        }
    }
}
impl SimilarityConfig {
    pub(crate) fn to_native(&self) -> PyResult<NativeConfig> {
        Ok(NativeConfig {
            translation: self.translation.to_native()?,
            max_angle: self.max_angle,
            max_log_scale: self.max_log_scale,
        })
    }
}
