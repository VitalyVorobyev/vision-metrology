use crate::corr::{RegionCheck, TranslationConfig};
use vm_primitives::Error;

/// Local masked similarity registration, with bounded departures from a prediction.
#[derive(Debug, Clone, PartialEq)]
pub struct SimilarityConfig {
    /// Masked translation initialization, support, score, margin and numerical tuning.
    /// `region_check` gates each regional similarity fit against the global one, at the
    /// region's own centroid and RMS radius; its defaults here differ from translation's.
    pub translation: TranslationConfig,
    /// Maximum rotation departure from the prediction, in radians.
    pub max_angle: f32,
    /// Maximum absolute log-scale departure from the prediction.
    pub max_log_scale: f32,
}
impl Default for SimilarityConfig {
    fn default() -> Self {
        let mut translation = TranslationConfig::default();
        translation.tuning.iterations = std::num::NonZeroUsize::new(30).unwrap();
        // 18.5 is the 99.9% quantile of chi-square with 4 dof. The floor keeps false
        // rejects of true similarities at or under 1% on the synthetic study
        // (`tools/registration_eval/false_reject.py`) and on HPatches development
        // self-pairs (`hpatches.py --pairs self`), where bilinear interpolation of real
        // texture adds regional bias.
        translation.region_check = Some(RegionCheck {
            max_chi2: 18.5,
            bias_floor_px: 0.06,
        });
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
