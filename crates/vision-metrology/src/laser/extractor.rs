//! The reusable [`LaserExtractor`] and its public entry points.

use std::ops::Range;

use vm_primitives::{Edge1DDetector, Error, ImageView};

use super::postprocess::{build_points, smooth_valid_centers};
use super::scan::{ScanPixel, extract_cols_gather_samples, extract_rows_samples};
use super::types::{ColAccess, LaserExtractConfig, LaserLine, ScanAxis};

/// Reusable gather scratch, one buffer per supported pixel type.
#[derive(Debug, Clone, Default)]
pub(super) struct ColBufs {
    pub(super) u8: Vec<u8>,
    pub(super) u16: Vec<u16>,
    pub(super) f32: Vec<f32>,
}

/// Reusable laser stripe extractor.
///
/// Owns scratch buffers for 1-D edge detection and column gathering. All
/// `extract_line_*` methods are allocation-free per row/column scan after
/// initial construction.
#[derive(Debug, Clone)]
pub struct LaserExtractor {
    detector: Edge1DDetector,
    bufs: ColBufs,
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
            bufs: ColBufs::default(),
        }
    }

    /// Update the Gaussian sigma used by the internal 1-D edge detector.
    pub fn set_sigma(&mut self, sigma: f32) {
        self.detector.set_sigma(sigma);
    }

    /// Extract a laser stripe from a `u8` image over `scan_range` rows or columns.
    ///
    /// Pass `transposed` when using `ColAccess::Transposed` (the transposed
    /// image must have dimensions swapped relative to `img`).
    ///
    /// # Errors
    /// Returns [`Error::InvalidConfig`] when `cfg.axis` selects
    /// `ColAccess::Transposed` but `transposed` is `None`, or when the supplied
    /// view is not the transpose of `img`. All other axis modes never fail.
    pub fn extract_line_u8(
        &mut self,
        img: &ImageView<'_, u8>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
        transposed: Option<&ImageView<'_, u8>>,
    ) -> Result<LaserLine, Error> {
        self.extract_line(img, scan_range, cfg, transposed)
    }

    /// Extract a laser stripe from a `u16` image over `scan_range` rows or columns.
    ///
    /// See [`extract_line_u8`][Self::extract_line_u8] for parameter details.
    pub fn extract_line_u16(
        &mut self,
        img: &ImageView<'_, u16>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
        transposed: Option<&ImageView<'_, u16>>,
    ) -> Result<LaserLine, Error> {
        self.extract_line(img, scan_range, cfg, transposed)
    }

    /// Extract a laser stripe from an `f32` image over `scan_range` rows or columns.
    ///
    /// See [`extract_line_u8`][Self::extract_line_u8] for parameter details.
    pub fn extract_line_f32(
        &mut self,
        img: &ImageView<'_, f32>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
        transposed: Option<&ImageView<'_, f32>>,
    ) -> Result<LaserLine, Error> {
        self.extract_line(img, scan_range, cfg, transposed)
    }

    fn extract_line<T: ScanPixel>(
        &mut self,
        img: &ImageView<'_, T>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
        transposed: Option<&ImageView<'_, T>>,
    ) -> Result<LaserLine, Error> {
        self.detector.set_sigma(cfg.edge_cfg.sigma);
        let mut samples = match cfg.axis {
            ScanAxis::Rows => extract_rows_samples(&mut self.detector, img, scan_range, cfg),
            ScanAxis::Cols {
                access: ColAccess::Gather,
            } => extract_cols_gather_samples(
                &mut self.detector,
                &mut self.bufs,
                img,
                scan_range,
                cfg,
            ),
            ScanAxis::Cols {
                access: ColAccess::Transposed,
            } => {
                let img_t = transposed_view(img, transposed)?;
                extract_rows_samples(&mut self.detector, img_t, scan_range, cfg)
            }
        };

        if cfg.enable_smoothing {
            smooth_valid_centers(&mut samples);
        }

        let points = build_points(&samples, cfg.axis);
        Ok(LaserLine {
            axis: cfg.axis,
            samples,
            points,
        })
    }
}
