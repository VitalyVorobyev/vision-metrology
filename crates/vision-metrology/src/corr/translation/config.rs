use std::num::NonZeroUsize;

use vm_primitives::Error;

/// Bounded masked translation search and its acceptance gates.
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationConfig {
    /// Half extent of the search around the prediction, in full-resolution pixels.
    pub search_radius: NonZeroUsize,
    /// Minimum number of jointly valid pixels at full resolution.
    pub min_support: NonZeroUsize,
    /// Minimum integer-search ZNCC, in `[-1, 1]`.
    pub min_score: f32,
    /// Required peak-minus-competitor ZNCC; competitors are at least two pixels away.
    pub min_margin: f32,
    /// Maximum disagreement between independently refined regions, in pixels.
    /// `None` disables the regional check.
    pub max_region_error: Option<f32>,
    /// Numerical effort and dimensionless conditioning gates.
    pub tuning: TranslationTuning,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            search_radius: NonZeroUsize::new(12).unwrap(),
            min_support: NonZeroUsize::new(400).unwrap(),
            min_score: 0.5,
            min_margin: 0.02,
            max_region_error: Some(0.5),
            tuning: TranslationTuning::default(),
        }
    }
}

/// Effort controls for [`TranslationConfig`].
#[derive(Debug, Clone, PartialEq)]
pub struct TranslationTuning {
    /// Maximum number of 2× box-mean pyramid levels, including level zero.
    pub max_levels: NonZeroUsize,
    /// Maximum robust Gauss–Newton iterations per refinement.
    pub iterations: NonZeroUsize,
    /// Translation update below which refinement stops, in pixels.
    pub tolerance: f32,
    /// Huber cutoff in robust residual standard deviations.
    pub huber: f32,
    /// Minimum eigenvalue per pixel of the photometry-eliminated translation Hessian.
    /// Intensities are standardized first, so this is independent of pixel units.
    pub min_texture: f32,
}

impl Default for TranslationTuning {
    fn default() -> Self {
        Self {
            max_levels: NonZeroUsize::new(3).unwrap(),
            iterations: NonZeroUsize::new(12).unwrap(),
            tolerance: 0.005,
            huber: 1.5,
            min_texture: 1e-5,
        }
    }
}

impl TranslationConfig {
    pub(super) fn validate(&self) -> Result<(), Error> {
        if self.search_radius.get() > 4096
            || self.tuning.max_levels.get() > 12
            || !self.min_score.is_finite()
            || !(-1.0..=1.0).contains(&self.min_score)
            || !self.min_margin.is_finite()
            || !(0.0..=2.0).contains(&self.min_margin)
            || self
                .max_region_error
                .is_some_and(|v| !v.is_finite() || v <= 0.0)
            || [
                self.tuning.tolerance,
                self.tuning.huber,
                self.tuning.min_texture,
            ]
            .iter()
            .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::InvalidConfig(
                "invalid masked translation configuration",
            ));
        }
        Ok(())
    }
}
