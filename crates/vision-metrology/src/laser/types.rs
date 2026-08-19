//! Public data types and configuration for laser stripe extraction.

use vm_primitives::{BorderMode, Edge1DConfig, Point2f, SubpixRefine};

/// Which image axis to scan along when extracting a laser line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScanAxis {
    /// Scan horizontally: one detection per row.
    #[default]
    Rows,
    /// Scan vertically: one detection per column.
    Cols {
        /// Column memory access strategy.
        access: ColAccess,
    },
}

/// Memory access strategy for column-direction laser scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColAccess {
    /// Gather each column into a temporary buffer before detection.
    Gather,
    /// Use a pre-transposed image (row-major layout with axes swapped).
    Transposed,
}

/// Per-scan-line detection result for a laser stripe.
#[derive(Debug, Clone, PartialEq)]
pub struct LaserSample {
    /// Index of the row (or column) this sample was extracted from.
    pub scan_i: usize,
    /// Subpixel center of the stripe along the scan axis.
    pub center: f32,
    /// Stripe width in pixels (distance between opposite-polarity edge peaks).
    pub width: f32,
    /// Detection quality score (higher is better).
    pub score: f32,
    /// Left (or top) edge position in subpixel coordinates.
    pub left: f32,
    /// Right (or bottom) edge position in subpixel coordinates.
    pub right: f32,
    /// `false` when no valid stripe was found on this scan line.
    pub valid: bool,
}

/// Extracted laser line composed of per-scan-line samples and 2-D point list.
#[derive(Debug, Clone)]
pub struct LaserLine {
    /// Scan axis used during extraction.
    pub axis: ScanAxis,
    /// One entry per scanned row or column, including invalid samples.
    pub samples: Vec<LaserSample>,
    /// Valid subpixel stripe centres in image pixel coordinates.
    pub points: Vec<Point2f>,
}

/// Coarse centre-finding method applied before the precise DoG edge-pair search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoarseMethod {
    /// Use the position of the maximum intensity pixel as the coarse centre.
    Max,
    /// Compute the intensity-weighted centroid within a local window.
    CenterOfMass {
        /// Half-width of the centroid window in pixels.
        half_width: usize,
        /// Threshold as a fraction of the local maximum; pixels below are ignored.
        threshold_frac: f32,
    },
}

/// Smoothing applied to the extracted centre positions.
///
/// Post-processing, not detection: it runs over the finished samples and can
/// only move a centre that was already found. It used to be an
/// `enable_smoothing: bool` with the window hard-coded at 5, which meant the
/// only way to learn the strength of the filter was to read the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CenterSmoothing {
    /// Report the centres exactly as detected.
    #[default]
    None,
    /// Running median over each contiguous run of valid samples.
    ///
    /// A median rejects a single-scan outlier without rounding a genuine
    /// corner, which is why it is a median and not a mean. The window is
    /// `2 · half_window + 1` samples, clipped at the ends of the run; a
    /// `half_window` of 0 is a no-op.
    Median {
        /// Half the window length, in scan lines.
        half_window: usize,
    },
}

/// Configuration for [`LaserExtractor`](super::LaserExtractor).
///
/// The four fields here decide **what counts as a stripe**; everything about
/// *how* the extractor hunts for it lives in [`LaserExtractTuning`], reachable
/// as `tuning` and defaulted, so the common case stays
/// `LaserExtractConfig { axis, ..Default::default() }`.
#[derive(Debug, Clone)]
pub struct LaserExtractConfig {
    /// Which image axis to scan along (rows or columns).
    pub axis: ScanAxis,
    /// Minimum detection score to accept a sample as valid.
    pub min_score: f32,
    /// Minimum accepted stripe width in pixels.
    pub min_width: f32,
    /// Maximum accepted stripe width in pixels.
    pub max_width: f32,
    /// Search behaviour: the knobs a working setup rarely touches.
    pub tuning: LaserExtractTuning,
}

/// Advanced knobs for [`LaserExtractConfig`].
///
/// Separated because they describe the *search*, not the stripe: they trade
/// speed against robustness on a signal that is already being found, and a
/// caller who has not measured the difference should leave them alone.
#[derive(Debug, Clone)]
pub struct LaserExtractTuning {
    /// Coarse centre-finding method used to seed the ROI for each scan line.
    pub coarse: CoarseMethod,
    /// Half-width of the ROI (in pixels) centred on the coarse estimate.
    pub roi_half_width: usize,
    /// Maximum allowed position jump between adjacent scan lines (pixels).
    /// Jumps larger than this trigger a gap.
    pub max_jump_px: f32,
    /// Maximum number of consecutive invalid scan lines before the tracker resets.
    pub max_gap_scans: usize,
    /// Weight in `[0, 1]` blending the previous position into the coarse prior.
    /// `0.0` = no prior (full re-detection per scan line); `1.0` = frozen prior.
    pub prior_weight: f32,
    /// Configuration forwarded to the 1-D DoG edge detector.
    pub edge_cfg: Edge1DConfig,
    /// Smoothing applied to the output centre positions.
    pub smoothing: CenterSmoothing,
}

impl Default for LaserExtractConfig {
    fn default() -> Self {
        Self {
            axis: ScanAxis::Rows,
            min_score: 0.0,
            min_width: 2.0,
            max_width: 10.0,
            tuning: LaserExtractTuning::default(),
        }
    }
}

impl Default for LaserExtractTuning {
    fn default() -> Self {
        Self {
            coarse: CoarseMethod::CenterOfMass {
                half_width: 8,
                threshold_frac: 0.5,
            },
            roi_half_width: 32,
            max_jump_px: 8.0,
            max_gap_scans: 5,
            prior_weight: 0.2,
            edge_cfg: Edge1DConfig {
                sigma: 1.2,
                border: BorderMode::Clamp,
                pos_thresh: 0.0,
                neg_thresh: 0.0,
                refine: SubpixRefine::Parabolic3,
            },
            smoothing: CenterSmoothing::None,
        }
    }
}
