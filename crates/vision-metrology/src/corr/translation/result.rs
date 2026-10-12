use vm_primitives::Vec2f;

/// Outcome of masked registration. A failed estimate is diagnostic evidence, not a pose.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TranslationStatus {
    /// All configured checks passed.
    Accepted,
    /// Too few jointly visible pixels.
    InsufficientSupport,
    /// Intensity or gradient evidence cannot constrain both displacement axes.
    LowTexture,
    /// Weak correlation or a competing correlation peak.
    Ambiguous,
    /// The optimum reached the configured search boundary.
    SearchBoundary,
    /// The robust refinement did not converge to a finite, supported solution.
    RefinementFailed,
    /// Spatially separated regions disagree with one translation.
    ModelMismatch,
}

impl TranslationStatus {
    /// Stable diagnostic name, also used by Python.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Accepted => "accepted",
            Self::InsufficientSupport => "insufficient_support",
            Self::LowTexture => "low_texture",
            Self::Ambiguous => "ambiguous",
            Self::SearchBoundary => "search_boundary",
            Self::RefinementFailed => "refinement_failed",
            Self::ModelMismatch => "model_mismatch",
        }
    }
}

/// Masked content displacement, photometry, and the evidence qualifying them.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Translation {
    /// Content displacement: `p_current = p_previous + shift`, in pixel-centre coordinates.
    /// Absent when no supported estimate could be formed. Check `status` before use.
    pub shift: Option<Vec2f>,
    /// Integer-search ZNCC peak.
    pub score: Option<f32>,
    /// Minimum peak-minus-competitor margin across searched levels. Competitors
    /// are at least two pixels away at their level (at least two in full resolution).
    pub margin: Option<f32>,
    /// Jointly valid pixels used by the last refinement (or integer search on failure).
    pub support: usize,
    /// RMS of `previous - (gain * current + offset)` in previous-image pixel units.
    pub residual_rms: Option<f32>,
    /// Largest absolute photometric residual in previous-image pixel units.
    pub residual_max: Option<f32>,
    /// Photometric gain in `previous(x) ≈ gain * current(x + shift) + offset`.
    pub gain: Option<f32>,
    /// Photometric offset in previous-image pixel units.
    pub offset: Option<f32>,
    /// Local linearized covariance `[xx, xy, yy]` in px²; not a manufacturing tolerance.
    pub covariance: Option<[f32; 3]>,
    /// Regions with independent, supported refinements.
    pub regions: usize,
    /// Maximum distance between regional shifts. `None` with fewer than two regions
    /// or when the regional check is disabled.
    pub region_error: Option<f32>,
    /// Acceptance or the reason the estimate must not be used as a pose.
    pub status: TranslationStatus,
}

impl Translation {
    pub(super) fn empty(status: TranslationStatus) -> Self {
        Self {
            shift: None,
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
            status,
        }
    }
}
