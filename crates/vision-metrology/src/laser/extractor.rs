//! The reusable [`LaserExtractor`] and its public entry points.

use std::ops::Range;

use vm_primitives::{Edge1DDetector, Error, ImageView, Pixel};

use super::postprocess::{build_points, smooth_valid_centers};
use super::scan::{extract_cols_gather_samples, extract_rows_samples};
use super::types::{CenterSmoothing, ColAccess, LaserExtractConfig, LaserLine, ScanAxis};

/// Reusable laser stripe extractor.
///
/// Owns scratch buffers for 1-D edge detection and column gathering. All
/// `extract_line_*` methods are allocation-free per row/column scan after
/// initial construction.
#[derive(Debug, Clone)]
pub struct LaserExtractor {
    detector: Edge1DDetector,
}

/// Resolve the transposed view that [`ColAccess::Transposed`] requires.
///
/// Both failures here are ordinary API misuse -- forgetting the argument, or
/// passing a view that is not actually the transpose -- rather than broken
/// internal invariants, so they are reported instead of asserted.
fn transposed_view<'t, T>(
    img: &ImageView<'_, T>,
    transposed: Option<&'t ImageView<'t, T>>,
) -> Result<&'t ImageView<'t, T>, Error> {
    let img_t = transposed.ok_or(Error::InvalidConfig(
        "ColAccess::Transposed requires a transposed image",
    ))?;
    if img_t.width() != img.height() || img_t.height() != img.width() {
        return Err(Error::InvalidConfig(
            "transposed image dimensions must be the original's, swapped",
        ));
    }
    Ok(img_t)
}

impl LaserExtractor {
    /// Create a new extractor with a DoG smoothing sigma.
    pub fn new(sigma: f32) -> Self {
        Self {
            detector: Edge1DDetector::new(sigma),
        }
    }

    /// Update the Gaussian sigma used by the internal 1-D edge detector.
    pub fn set_sigma(&mut self, sigma: f32) {
        self.detector.set_sigma(sigma);
    }

    /// Extract a laser stripe over `scan_range` rows or columns, from an image
    /// of any [`Pixel`] type.
    ///
    /// Pass `transposed` when using `ColAccess::Transposed` (the transposed
    /// image must have dimensions swapped relative to `img`).
    ///
    /// # Errors
    /// Returns [`Error::InvalidConfig`] when `cfg.axis` selects
    /// `ColAccess::Transposed` but `transposed` is `None`, or when the supplied
    /// view is not the transpose of `img`. All other axis modes never fail.
    pub fn extract_line<P: Pixel>(
        &mut self,
        img: &ImageView<'_, P>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
        transposed: Option<&ImageView<'_, P>>,
    ) -> Result<LaserLine, Error> {
        self.detector.set_sigma(cfg.tuning.edge_cfg.sigma);
        let mut samples = match cfg.axis {
            ScanAxis::Rows => extract_rows_samples(&mut self.detector, img, scan_range, cfg),
            ScanAxis::Cols {
                access: ColAccess::Gather,
            } => extract_cols_gather_samples(&mut self.detector, img, scan_range, cfg),
            ScanAxis::Cols {
                access: ColAccess::Transposed,
            } => {
                let img_t = transposed_view(img, transposed)?;
                extract_rows_samples(&mut self.detector, img_t, scan_range, cfg)
            }
        };

        if let CenterSmoothing::Median { half_window } = cfg.tuning.smoothing {
            smooth_valid_centers(&mut samples, half_window);
        }

        let points = build_points(&samples, cfg.axis);
        Ok(LaserLine {
            axis: cfg.axis,
            samples,
            points,
        })
    }
}
