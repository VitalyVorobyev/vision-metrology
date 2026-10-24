use crate::corr::TranslationStatus;
use vm_primitives::Similarity2f;

/// Previous-to-current similarity and the evidence qualifying it.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Similarity {
    /// `q_current = transform * p_previous`. Check `status` before using it.
    pub transform: Option<Similarity2f>,
    /// Prediction-warped integer translation-search ZNCC.
    pub score: Option<f32>,
    /// Minimum competing translation peak margin over searched levels.
    pub margin: Option<f32>,
    /// Jointly valid native pixels in the last refinement.
    pub support: usize,
    /// Photometric residual RMS, in previous-image units.
    pub residual_rms: Option<f32>,
    /// Maximum absolute photometric residual, in previous-image units.
    pub residual_max: Option<f32>,
    /// `previous(p) ≈ gain * current(transform * p) + offset`.
    pub gain: Option<f32>,
    /// Photometric offset, in previous-image units.
    pub offset: Option<f32>,
    /// Local 4×4 covariance for centre displacement x/y, angle and log-scale.
    /// Units are px², rad², log-scale² and their cross terms. Not manufacturing uncertainty.
    pub covariance: Option<[[f32; 4]; 4]>,
    /// Independently supported regional similarity fits.
    pub regions: usize,
    /// Largest disagreement of a regional fit with the global map within that region's
    /// RMS radius about its centroid, in pixels: `|Δq(centroid)| + ρ |(Δangle, Δlog-scale)|`.
    /// Absent with fewer than two supported regions or when checks are disabled.
    pub region_error: Option<f32>,
    /// Largest covariance-normalized χ² of a regional fit against the global one
    /// (4 degrees of freedom), the statistic `RegionCheck` tests. `None` when
    /// `region_error` is.
    pub region_chi2: Option<f32>,
    /// Acceptance or reason the map is diagnostic only.
    pub status: TranslationStatus,
}
impl Similarity {
    pub(super) fn empty(status: TranslationStatus) -> Self {
        Self {
            transform: None,
            score: None,
            margin: None,
            support: 0,
            residual_rms: None,
            residual_max: None,
            gain: None,
            offset: None,
            covariance: None,
            regions: 0,
            region_error: None,
            region_chi2: None,
            status,
        }
    }
}
