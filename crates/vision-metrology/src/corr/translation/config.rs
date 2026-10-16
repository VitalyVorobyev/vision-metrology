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
    /// Covariance-normalized consistency of independently refined image quadrants.
    /// `None` disables the regional check.
    pub region_check: Option<RegionCheck>,
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
            region_check: Some(RegionCheck::default()),
            tuning: TranslationTuning::default(),
        }
    }
}

/// Covariance-normalized regional consistency test.
///
/// Independently refined image quadrants must agree with one motion model. Each compared
/// discrepancy `d` (in px) is normalized by its covariance, the sum of the compared fits'
/// local covariances plus `bias_floor_px²` per component, and the largest
/// `χ² = dᵀ S⁻¹ d` must not exceed `max_chi2`. Translation compares every pair of regional
/// shifts (2 degrees of freedom); similarity compares each regional fit with the global
/// one at the region's own centroid and RMS radius (4 degrees of freedom).
///
/// The local covariance assumes independent residuals; the floor carries model bias the
/// covariance does not (interpolation, filtering, unmodelled deformation). With
/// negligible noise the test tolerates about `bias_floor_px · sqrt(2 · max_chi2)` px of
/// disagreement in a translation pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RegionCheck {
    /// Largest accepted χ² of a regional discrepancy; dimensionless.
    pub max_chi2: f32,
    /// Model-bias floor added in quadrature to each discrepancy component, in pixels.
    pub bias_floor_px: f32,
}

impl Default for RegionCheck {
    /// Translation's defaults, chosen for about 1% false rejects of true translations on
    /// the synthetic study in `tools/registration_eval/false_reject.py`.
    fn default() -> Self {
        Self {
            max_chi2: 13.8,
            bias_floor_px: 0.05,
        }
    }
}

impl RegionCheck {
    pub(crate) fn is_valid(&self) -> bool {
        self.max_chi2.is_finite()
            && self.max_chi2 > 0.0
            && self.bias_floor_px.is_finite()
            && self.bias_floor_px >= 0.0
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
    pub(crate) fn validate(&self) -> Result<(), Error> {
        if self.search_radius.get() > 4096
            || self.tuning.max_levels.get() > 12
            || !self.min_score.is_finite()
            || !(-1.0..=1.0).contains(&self.min_score)
            || !self.min_margin.is_finite()
            || !(0.0..=2.0).contains(&self.min_margin)
            || self.region_check.is_some_and(|c| !c.is_valid())
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
