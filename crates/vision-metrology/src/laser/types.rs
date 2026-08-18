//! Public data types and configuration for laser stripe extraction.

use vm_primitives::{BorderMode, Edge1DConfig, Point2f, SubpixRefine};

/// Which image axis to scan along when extracting a laser line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanAxis {
    /// Scan horizontally: one detection per row.
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

/// Configuration for [`LaserExtractor`].
#[derive(Debug, Clone)]
pub struct LaserExtractConfig {
    /// Which image axis to scan along (rows or columns).
    pub axis: ScanAxis,
    /// Coarse centre-finding method used to seed the ROI for each scan line.
    pub coarse: CoarseMethod,
    /// Half-width of the ROI (in pixels) centred on the coarse estimate.
    pub roi_half_width: usize,
    /// Maximum allowed position jump between adjacent scan lines (pixels).
    /// Jumps larger than this trigger a gap.
    pub max_jump_px: f32,
    /// Maximum number of consecutive invalid scan lines before the tracker resets.
    pub max_gap_scans: usize,
    /// Minimum detection score to accept a sample as valid.
    pub min_score: f32,
    /// Minimum accepted stripe width in pixels.
    pub min_width: f32,
    /// Maximum accepted stripe width in pixels.
    pub max_width: f32,
    /// Configuration forwarded to the 1-D DoG edge detector.
    pub edge_cfg: Edge1DConfig,
    /// Weight in `[0, 1]` blending the previous position into the coarse prior.
    /// `0.0` = no prior (full re-detection per scan line); `1.0` = frozen prior.
    pub prior_weight: f32,
    /// If `true`, apply light smoothing to the output centre positions.
    pub enable_smoothing: bool,
}

impl Default for LaserExtractConfig {
    fn default() -> Self {
        Self {
            axis: ScanAxis::Rows,
            coarse: CoarseMethod::CenterOfMass {
                half_width: 8,
                threshold_frac: 0.5,
            },
            roi_half_width: 32,
            max_jump_px: 8.0,
            max_gap_scans: 5,
            min_score: 0.0,
            min_width: 2.0,
            max_width: 10.0,
            edge_cfg: Edge1DConfig {
                sigma: 1.2,
                border: BorderMode::Clamp,
                pos_thresh: 0.0,
                neg_thresh: 0.0,
                refine: SubpixRefine::Parabolic3,
            },
            prior_weight: 0.2,
            enable_smoothing: false,
        }
    }
}
