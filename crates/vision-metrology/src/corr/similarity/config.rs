use crate::corr::TranslationConfig;
use vm_primitives::Error;

/// Local masked similarity registration, with bounded departures from a prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityConfig {
    /// Masked translation initialization, support, score, margin and numerical tuning.
    /// `max_region_error` also gates disagreement of independent similarity fits.
    pub translation: TranslationConfig,
    /// Maximum rotation departure from the prediction, in radians.
    pub max_angle: f32,
    /// Maximum absolute log-scale departure from the prediction.
    pub max_log_scale: f32,
}
impl Default for SimilarityConfig {
    fn default() -> Self {
        let mut translation = TranslationConfig::default();
        translation.tuning.iterations = std::num::NonZeroUsize::new(20).unwrap();
        Self {
            translation,
            max_angle: 0.05,
            max_log_scale: 0.05,
        }
    }
}
impl SimilarityConfig {
    pub(super) fn validate(&self) -> Result<(), Error> {
        self.translation.validate()?;
        if !self.max_angle.is_finite()
            || self.max_angle <= 0.0
            || self.max_angle > 0.5
            || !self.max_log_scale.is_finite()
            || self.max_log_scale <= 0.0
            || self.max_log_scale > 0.5
        {
            return Err(Error::InvalidConfig("invalid similarity bounds"));
        }
        Ok(())
    }
}
