use std::ops::Range;

use vm_primitives::{BorderMode, Error, ImageView, Point2f};
use vm_primitives::{
    Edge1DConfig, Edge1DDetector, EdgePair1D, EdgePeak, EdgePolarity, SubpixRefine,
};

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

/// Reusable laser stripe extractor.
///
/// Owns scratch buffers for 1-D edge detection and column gathering. All
/// `extract_line_*` methods are allocation-free per row/column scan after
/// initial construction.
#[derive(Debug, Clone)]
pub struct LaserExtractor {
    detector: Edge1DDetector,
    col_u8: Vec<u8>,
    col_u16: Vec<u16>,
    col_f32: Vec<f32>,
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
            col_u8: Vec::new(),
            col_u16: Vec::new(),
            col_f32: Vec::new(),
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
        self.detector.set_sigma(cfg.edge_cfg.sigma);
        let mut samples = match cfg.axis {
            ScanAxis::Rows => self.extract_rows_u8_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Gather,
            } => self.extract_cols_gather_u8_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Transposed,
            } => {
                let img_t = transposed_view(img, transposed)?;
                self.extract_rows_u8_samples(img_t, scan_range.clone(), cfg)
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
        self.detector.set_sigma(cfg.edge_cfg.sigma);
        let mut samples = match cfg.axis {
            ScanAxis::Rows => self.extract_rows_u16_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Gather,
            } => self.extract_cols_gather_u16_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Transposed,
            } => {
                let img_t = transposed_view(img, transposed)?;
                self.extract_rows_u16_samples(img_t, scan_range.clone(), cfg)
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
        self.detector.set_sigma(cfg.edge_cfg.sigma);
        let mut samples = match cfg.axis {
            ScanAxis::Rows => self.extract_rows_f32_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Gather,
            } => self.extract_cols_gather_f32_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Transposed,
            } => {
                let img_t = transposed_view(img, transposed)?;
                self.extract_rows_f32_samples(img_t, scan_range.clone(), cfg)
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

    fn extract_rows_u8_samples(
        &mut self,
        img: &ImageView<'_, u8>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
    ) -> Vec<LaserSample> {
        assert!(
            scan_range.end <= img.height(),
            "scan range out of row bounds"
        );
        let mut samples = Vec::with_capacity(scan_range.end.saturating_sub(scan_range.start));

        let mut last_valid_center: Option<f32> = None;
        let mut gap_len = 0usize;

        for scan_i in scan_range {
            let line = img.row(scan_i);
            let n = line.len();

            let tracking = last_valid_center.is_some() && gap_len <= cfg.max_gap_scans;
            let predicted = if tracking {
                let last = last_valid_center.expect("checked");
                let (b0, b1) = roi_bounds(last, cfg.roi_half_width, n);
                coarse_center_u8_in_range(line, &cfg.coarse, b0, b1).unwrap_or(last)
            } else {
                match coarse_center_u8(line, &cfg.coarse) {
                    Some(v) => v,
                    None => {
                        samples.push(invalid_sample(scan_i, None));
                        gap_len += 1;
                        continue;
                    }
                }
            };

            let cand = detect_pair_u8(
                &mut self.detector,
                line,
                predicted,
                tracking,
                cfg,
                0,
                scan_i,
            );

            match cand {
                Some(s) => {
                    last_valid_center = Some(s.center);
                    gap_len = 0;
                    samples.push(s);
                }
                None => {
                    samples.push(invalid_sample(scan_i, Some(predicted)));
                    gap_len += 1;
                }
            }
        }

        samples
    }

    fn extract_rows_u16_samples(
        &mut self,
        img: &ImageView<'_, u16>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
    ) -> Vec<LaserSample> {
        assert!(
            scan_range.end <= img.height(),
            "scan range out of row bounds"
        );
        let mut samples = Vec::with_capacity(scan_range.end.saturating_sub(scan_range.start));

        let mut last_valid_center: Option<f32> = None;
        let mut gap_len = 0usize;

        for scan_i in scan_range {
            let line = img.row(scan_i);
            let n = line.len();

            let tracking = last_valid_center.is_some() && gap_len <= cfg.max_gap_scans;
            let predicted = if tracking {
                let last = last_valid_center.expect("checked");
                let (b0, b1) = roi_bounds(last, cfg.roi_half_width, n);
                coarse_center_u16_in_range(line, &cfg.coarse, b0, b1).unwrap_or(last)
            } else {
                match coarse_center_u16(line, &cfg.coarse) {
                    Some(v) => v,
                    None => {
                        samples.push(invalid_sample(scan_i, None));
                        gap_len += 1;
                        continue;
                    }
                }
            };

            let cand = detect_pair_u16(
                &mut self.detector,
                line,
                predicted,
                tracking,
                cfg,
                0,
                scan_i,
            );

            match cand {
                Some(s) => {
                    last_valid_center = Some(s.center);
                    gap_len = 0;
                    samples.push(s);
                }
                None => {
                    samples.push(invalid_sample(scan_i, Some(predicted)));
                    gap_len += 1;
                }
            }
        }

        samples
    }

    fn extract_rows_f32_samples(
        &mut self,
        img: &ImageView<'_, f32>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
    ) -> Vec<LaserSample> {
        assert!(
            scan_range.end <= img.height(),
            "scan range out of row bounds"
        );
        let mut samples = Vec::with_capacity(scan_range.end.saturating_sub(scan_range.start));

        let mut last_valid_center: Option<f32> = None;
        let mut gap_len = 0usize;

        for scan_i in scan_range {
            let line = img.row(scan_i);
            let n = line.len();

            let tracking = last_valid_center.is_some() && gap_len <= cfg.max_gap_scans;
            let predicted = if tracking {
                let last = last_valid_center.expect("checked");
                let (b0, b1) = roi_bounds(last, cfg.roi_half_width, n);
                coarse_center_f32_in_range(line, &cfg.coarse, b0, b1).unwrap_or(last)
            } else {
                match coarse_center_f32(line, &cfg.coarse) {
                    Some(v) => v,
                    None => {
                        samples.push(invalid_sample(scan_i, None));
                        gap_len += 1;
                        continue;
                    }
                }
            };

            let cand = detect_pair_f32(
                &mut self.detector,
                line,
                predicted,
                tracking,
                cfg,
                0,
                scan_i,
            );

            match cand {
                Some(s) => {
                    last_valid_center = Some(s.center);
                    gap_len = 0;
                    samples.push(s);
                }
                None => {
                    samples.push(invalid_sample(scan_i, Some(predicted)));
                    gap_len += 1;
                }
            }
        }

        samples
    }

    fn extract_cols_gather_u8_samples(
        &mut self,
        img: &ImageView<'_, u8>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
    ) -> Vec<LaserSample> {
        assert!(
            scan_range.end <= img.width(),
            "scan range out of col bounds"
        );
        let mut samples = Vec::with_capacity(scan_range.end.saturating_sub(scan_range.start));

        let detector = &mut self.detector;
        let col_buf = &mut self.col_u8;
        let mut last_valid_center: Option<f32> = None;
        let mut gap_len = 0usize;

        for scan_i in scan_range {
            let n = img.height();
            let tracking = last_valid_center.is_some() && gap_len <= cfg.max_gap_scans;

            let predicted = if tracking {
                let last = last_valid_center.expect("checked");
                let (b0, b1) = roi_bounds(last, cfg.roi_half_width, n);
                let line = gather_col_segment_u8(img, scan_i, b0, b1, col_buf);
                coarse_center_u8_in_range(line, &cfg.coarse, 0, line.len())
                    .map(|v| v + b0 as f32)
                    .unwrap_or(last)
            } else {
                let line = gather_col_segment_u8(img, scan_i, 0, n, col_buf);
                match coarse_center_u8(line, &cfg.coarse) {
                    Some(v) => v,
                    None => {
                        samples.push(invalid_sample(scan_i, None));
                        gap_len += 1;
                        continue;
                    }
                }
            };

            let (roi0, roi1) = roi_bounds(predicted, cfg.roi_half_width, n);
            if roi1.saturating_sub(roi0) < 3 {
                samples.push(invalid_sample(scan_i, Some(predicted)));
                gap_len += 1;
                continue;
            }

            let roi_line = gather_col_segment_u8(img, scan_i, roi0, roi1, col_buf);
            let cand = detect_pair_u8(detector, roi_line, predicted, tracking, cfg, roi0, scan_i);

            match cand {
                Some(s) => {
                    last_valid_center = Some(s.center);
                    gap_len = 0;
                    samples.push(s);
                }
                None => {
                    samples.push(invalid_sample(scan_i, Some(predicted)));
                    gap_len += 1;
                }
            }
        }

        samples
    }

    fn extract_cols_gather_u16_samples(
        &mut self,
        img: &ImageView<'_, u16>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
    ) -> Vec<LaserSample> {
        assert!(
            scan_range.end <= img.width(),
            "scan range out of col bounds"
        );
        let mut samples = Vec::with_capacity(scan_range.end.saturating_sub(scan_range.start));

        let detector = &mut self.detector;
        let col_buf = &mut self.col_u16;
        let mut last_valid_center: Option<f32> = None;
        let mut gap_len = 0usize;

        for scan_i in scan_range {
            let n = img.height();
            let tracking = last_valid_center.is_some() && gap_len <= cfg.max_gap_scans;

            let predicted = if tracking {
                let last = last_valid_center.expect("checked");
                let (b0, b1) = roi_bounds(last, cfg.roi_half_width, n);
                let line = gather_col_segment_u16(img, scan_i, b0, b1, col_buf);
                coarse_center_u16_in_range(line, &cfg.coarse, 0, line.len())
                    .map(|v| v + b0 as f32)
                    .unwrap_or(last)
            } else {
                let line = gather_col_segment_u16(img, scan_i, 0, n, col_buf);
                match coarse_center_u16(line, &cfg.coarse) {
                    Some(v) => v,
                    None => {
                        samples.push(invalid_sample(scan_i, None));
                        gap_len += 1;
                        continue;
                    }
                }
            };

            let (roi0, roi1) = roi_bounds(predicted, cfg.roi_half_width, n);
            if roi1.saturating_sub(roi0) < 3 {
                samples.push(invalid_sample(scan_i, Some(predicted)));
                gap_len += 1;
                continue;
            }

            let roi_line = gather_col_segment_u16(img, scan_i, roi0, roi1, col_buf);
            let cand = detect_pair_u16(detector, roi_line, predicted, tracking, cfg, roi0, scan_i);

            match cand {
                Some(s) => {
                    last_valid_center = Some(s.center);
                    gap_len = 0;
                    samples.push(s);
                }
                None => {
                    samples.push(invalid_sample(scan_i, Some(predicted)));
                    gap_len += 1;
                }
            }
        }

        samples
    }

    fn extract_cols_gather_f32_samples(
        &mut self,
        img: &ImageView<'_, f32>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
    ) -> Vec<LaserSample> {
        assert!(
            scan_range.end <= img.width(),
            "scan range out of col bounds"
        );
        let mut samples = Vec::with_capacity(scan_range.end.saturating_sub(scan_range.start));

        let detector = &mut self.detector;
        let col_buf = &mut self.col_f32;
        let mut last_valid_center: Option<f32> = None;
        let mut gap_len = 0usize;

        for scan_i in scan_range {
            let n = img.height();
            let tracking = last_valid_center.is_some() && gap_len <= cfg.max_gap_scans;

            let predicted = if tracking {
                let last = last_valid_center.expect("checked");
                let (b0, b1) = roi_bounds(last, cfg.roi_half_width, n);
                let line = gather_col_segment_f32(img, scan_i, b0, b1, col_buf);
                coarse_center_f32_in_range(line, &cfg.coarse, 0, line.len())
                    .map(|v| v + b0 as f32)
                    .unwrap_or(last)
            } else {
                let line = gather_col_segment_f32(img, scan_i, 0, n, col_buf);
                match coarse_center_f32(line, &cfg.coarse) {
                    Some(v) => v,
                    None => {
                        samples.push(invalid_sample(scan_i, None));
                        gap_len += 1;
                        continue;
                    }
                }
            };

            let (roi0, roi1) = roi_bounds(predicted, cfg.roi_half_width, n);
            if roi1.saturating_sub(roi0) < 3 {
                samples.push(invalid_sample(scan_i, Some(predicted)));
                gap_len += 1;
                continue;
            }

            let roi_line = gather_col_segment_f32(img, scan_i, roi0, roi1, col_buf);
            let cand = detect_pair_f32(detector, roi_line, predicted, tracking, cfg, roi0, scan_i);

            match cand {
                Some(s) => {
                    last_valid_center = Some(s.center);
                    gap_len = 0;
                    samples.push(s);
                }
                None => {
                    samples.push(invalid_sample(scan_i, Some(predicted)));
                    gap_len += 1;
                }
            }
        }

        samples
    }
}

fn detect_pair_u8(
    detector: &mut Edge1DDetector,
    line: &[u8],
    predicted: f32,
    tracking: bool,
    cfg: &LaserExtractConfig,
    x_offset: usize,
    scan_i: usize,
) -> Option<LaserSample> {
    let peaks = detector.detect_in_u8_ref(line, &cfg.edge_cfg);
    let pair = best_pair_with_prior_offset(
        peaks,
        x_offset as f32,
        cfg.min_width,
        cfg.max_width,
        predicted,
        cfg.prior_weight,
    )?;

    accept_pair(pair, predicted, tracking, cfg, scan_i)
}

fn detect_pair_u16(
    detector: &mut Edge1DDetector,
    line: &[u16],
    predicted: f32,
    tracking: bool,
    cfg: &LaserExtractConfig,
    x_offset: usize,
    scan_i: usize,
) -> Option<LaserSample> {
    let peaks = detector.detect_in_u16_ref(line, &cfg.edge_cfg);
    let pair = best_pair_with_prior_offset(
        peaks,
        x_offset as f32,
        cfg.min_width,
        cfg.max_width,
        predicted,
        cfg.prior_weight,
    )?;

    accept_pair(pair, predicted, tracking, cfg, scan_i)
}

fn detect_pair_f32(
    detector: &mut Edge1DDetector,
    line: &[f32],
    predicted: f32,
    tracking: bool,
    cfg: &LaserExtractConfig,
    x_offset: usize,
    scan_i: usize,
) -> Option<LaserSample> {
    let peaks = detector.detect_in_f32_ref(line, &cfg.edge_cfg);
    let pair = best_pair_with_prior_offset(
        peaks,
        x_offset as f32,
        cfg.min_width,
        cfg.max_width,
        predicted,
        cfg.prior_weight,
    )?;

    accept_pair(pair, predicted, tracking, cfg, scan_i)
}

fn accept_pair(
    pair: EdgePair1D,
    predicted: f32,
    tracking: bool,
    cfg: &LaserExtractConfig,
    scan_i: usize,
) -> Option<LaserSample> {
    if pair.score < cfg.min_score {
        return None;
    }

    if tracking && (pair.center_x - predicted).abs() > cfg.max_jump_px {
        return None;
    }

    Some(LaserSample {
        scan_i,
        center: pair.center_x,
        width: pair.width,
        score: pair.score,
        left: pair.left.x,
        right: pair.right.x,
        valid: true,
    })
}

fn invalid_sample(scan_i: usize, predicted: Option<f32>) -> LaserSample {
    LaserSample {
        scan_i,
        center: predicted.unwrap_or(f32::NAN),
        width: 0.0,
        score: 0.0,
        left: f32::NAN,
        right: f32::NAN,
        valid: false,
    }
}

fn build_points(samples: &[LaserSample], axis: ScanAxis) -> Vec<Point2f> {
    let mut points = Vec::with_capacity(samples.len());
    for s in samples {
        if !s.valid {
            continue;
        }

        let p = match axis {
            ScanAxis::Rows => Point2f {
                x: s.center,
                y: s.scan_i as f32,
            },
            ScanAxis::Cols { .. } => Point2f {
                x: s.scan_i as f32,
                y: s.center,
            },
        };
        points.push(p);
    }
    points
}

fn smooth_valid_centers(samples: &mut [LaserSample]) {
    let mut i = 0usize;
    while i < samples.len() {
        if !samples[i].valid {
            i += 1;
            continue;
        }

        let start = i;
        while i < samples.len() && samples[i].valid {
            i += 1;
        }
        let end = i;

        let run_len = end - start;
        let mut orig = Vec::with_capacity(run_len);
        for s in samples.iter().take(end).skip(start) {
            orig.push(s.center);
        }

        for j in 0..run_len {
            let j0 = j.saturating_sub(2);
            let j1 = (j + 2).min(run_len - 1);
            let mut vals = [0.0f32; 5];
            let mut count = 0usize;
            for &v in orig.iter().take(j1 + 1).skip(j0) {
                vals[count] = v;
                count += 1;
            }
            vals[..count].sort_by(|a, b| a.partial_cmp(b).expect("finite compare"));
            samples[start + j].center = vals[count / 2];
        }
    }
}

/// Find the best bright-on-dark edge pair given a predicted centre position.
///
/// Scores each pair as `(left.strength + right.strength) - prior_weight * |centre - predicted|`.
/// Returns `None` when no valid pair exists within `[min_width, max_width]`.
pub fn best_pair_with_prior(
    peaks: &[EdgePeak],
    min_width: f32,
    max_width: f32,
    predicted_center: f32,
    prior_weight: f32,
) -> Option<EdgePair1D> {
    best_pair_with_prior_offset(
        peaks,
        0.0,
        min_width,
        max_width,
        predicted_center,
        prior_weight,
    )
}

fn best_pair_with_prior_offset(
    peaks: &[EdgePeak],
    x_offset: f32,
    min_width: f32,
    max_width: f32,
    predicted_center: f32,
    prior_weight: f32,
) -> Option<EdgePair1D> {
    let mut best: Option<EdgePair1D> = None;
    let xoff_idx = x_offset.round() as usize;

    for left in peaks.iter().filter(|p| p.polarity == EdgePolarity::Rising) {
        for right in peaks.iter().filter(|p| p.polarity == EdgePolarity::Falling) {
            let lx = left.x + x_offset;
            let rx = right.x + x_offset;
            if rx <= lx {
                continue;
            }

            let width = rx - lx;
            if width < min_width || width > max_width {
                continue;
            }

            let center = 0.5 * (lx + rx);
            let base_score = left.strength + right.strength;
            let total_score = base_score - prior_weight * (center - predicted_center).abs();

            let cand = EdgePair1D {
                left: EdgePeak {
                    x: lx,
                    idx: left.idx + xoff_idx,
                    value: left.value,
                    strength: left.strength,
                    polarity: left.polarity,
                },
                right: EdgePeak {
                    x: rx,
                    idx: right.idx + xoff_idx,
                    value: right.value,
                    strength: right.strength,
                    polarity: right.polarity,
                },
                center_x: center,
                width,
                score: total_score,
                bright_on_dark: true,
            };

            if best.as_ref().is_none_or(|b| cand.score > b.score) {
                best = Some(cand);
            }
        }
    }

    best
}

/// Estimate the coarse stripe centre in a `u8` scan-line using the given method.
///
/// Returns the estimated subpixel centre position, or `None` if the signal is
/// too weak to locate a centre.
pub fn coarse_center_u8(line: &[u8], coarse: &CoarseMethod) -> Option<f32> {
    coarse_center_u8_in_range(line, coarse, 0, line.len())
}

/// Estimate the coarse stripe centre in a `u16` scan-line using the given method.
///
/// See [`coarse_center_u8`] for details.
pub fn coarse_center_u16(line: &[u16], coarse: &CoarseMethod) -> Option<f32> {
    coarse_center_u16_in_range(line, coarse, 0, line.len())
}

/// Estimate the coarse stripe centre in an `f32` scan-line using the given method.
///
/// See [`coarse_center_u8`] for details.
pub fn coarse_center_f32(line: &[f32], coarse: &CoarseMethod) -> Option<f32> {
    coarse_center_f32_in_range(line, coarse, 0, line.len())
}

fn coarse_center_u8_in_range(
    line: &[u8],
    coarse: &CoarseMethod,
    start: usize,
    end: usize,
) -> Option<f32> {
    if start >= end || end > line.len() {
        return None;
    }

    let (max_idx, max_v) = argmax_u8(&line[start..end]).map(|(i, v)| (start + i, v as f32))?;
    match *coarse {
        CoarseMethod::Max => Some(max_idx as f32),
        CoarseMethod::CenterOfMass {
            half_width,
            threshold_frac,
        } => {
            let thr = threshold_frac.max(0.0) * max_v;
            let w0 = max_idx.saturating_sub(half_width).max(start);
            let w1 = (max_idx + half_width + 1).min(end);
            let mut sum_w = 0.0f32;
            let mut sum_xw = 0.0f32;
            for (i, &v) in line.iter().enumerate().take(w1).skip(w0) {
                let vf = v as f32;
                if vf >= thr {
                    sum_w += vf;
                    sum_xw += (i as f32) * vf;
                }
            }
            if sum_w <= f32::EPSILON {
                Some(max_idx as f32)
            } else {
                Some(sum_xw / sum_w)
            }
        }
    }
}

fn coarse_center_u16_in_range(
    line: &[u16],
    coarse: &CoarseMethod,
    start: usize,
    end: usize,
) -> Option<f32> {
    if start >= end || end > line.len() {
        return None;
    }

    let (max_idx, max_v) = argmax_u16(&line[start..end]).map(|(i, v)| (start + i, v as f32))?;
    match *coarse {
        CoarseMethod::Max => Some(max_idx as f32),
        CoarseMethod::CenterOfMass {
            half_width,
            threshold_frac,
        } => {
            let thr = threshold_frac.max(0.0) * max_v;
            let w0 = max_idx.saturating_sub(half_width).max(start);
            let w1 = (max_idx + half_width + 1).min(end);
            let mut sum_w = 0.0f32;
            let mut sum_xw = 0.0f32;
            for (i, &v) in line.iter().enumerate().take(w1).skip(w0) {
                let vf = v as f32;
                if vf >= thr {
                    sum_w += vf;
                    sum_xw += (i as f32) * vf;
                }
            }
            if sum_w <= f32::EPSILON {
                Some(max_idx as f32)
            } else {
                Some(sum_xw / sum_w)
            }
        }
    }
}

fn coarse_center_f32_in_range(
    line: &[f32],
    coarse: &CoarseMethod,
    start: usize,
    end: usize,
) -> Option<f32> {
    if start >= end || end > line.len() {
        return None;
    }

    let (max_idx, max_v) = argmax_f32(&line[start..end]).map(|(i, v)| (start + i, v))?;
    match *coarse {
        CoarseMethod::Max => Some(max_idx as f32),
        CoarseMethod::CenterOfMass {
            half_width,
            threshold_frac,
        } => {
            let thr = threshold_frac.max(0.0) * max_v;
            let w0 = max_idx.saturating_sub(half_width).max(start);
            let w1 = (max_idx + half_width + 1).min(end);
            let mut sum_w = 0.0f32;
            let mut sum_xw = 0.0f32;
            for (i, &v) in line.iter().enumerate().take(w1).skip(w0) {
                if v >= thr {
                    sum_w += v;
                    sum_xw += (i as f32) * v;
                }
            }
            if sum_w <= f32::EPSILON {
                Some(max_idx as f32)
            } else {
                Some(sum_xw / sum_w)
            }
        }
    }
}

fn argmax_u8(line: &[u8]) -> Option<(usize, u8)> {
    let mut it = line.iter().copied().enumerate();
    let mut best = it.next()?;
    for (i, v) in it {
        if v > best.1 {
            best = (i, v);
        }
    }
    Some(best)
}

fn argmax_u16(line: &[u16]) -> Option<(usize, u16)> {
    let mut it = line.iter().copied().enumerate();
    let mut best = it.next()?;
    for (i, v) in it {
        if v > best.1 {
            best = (i, v);
        }
    }
    Some(best)
}

fn argmax_f32(line: &[f32]) -> Option<(usize, f32)> {
    let mut it = line.iter().copied().enumerate();
    let mut best = it.next()?;
    for (i, v) in it {
        if v > best.1 {
            best = (i, v);
        }
    }
    Some(best)
}

#[inline]
fn roi_bounds(center: f32, half_width: usize, len: usize) -> (usize, usize) {
    if len == 0 {
        return (0, 0);
    }
    let c = center.round() as isize;
    let start = (c - half_width as isize).max(0) as usize;
    let end = (c + half_width as isize + 1).min(len as isize) as usize;
    (start.min(len), end.min(len))
}

fn gather_col_segment_u8<'a>(
    img: &ImageView<'_, u8>,
    x: usize,
    y0: usize,
    y1: usize,
    out: &'a mut Vec<u8>,
) -> &'a [u8] {
    assert!(x < img.width(), "x out of bounds");
    assert!(y0 <= y1 && y1 <= img.height(), "invalid y-range");

    let len = y1 - y0;
    out.resize(len, 0);

    if img.is_contiguous()
        && let Some(data) = img.as_contiguous_slice()
    {
        let w = img.width();
        // SAFETY:
        // - `x < w`; `y in [y0, y1)` and `y < img.height()`.
        // - index `y*w + x` is in-bounds for contiguous image backing.
        unsafe {
            for (i, y) in (y0..y1).enumerate() {
                *out.get_unchecked_mut(i) = *data.get_unchecked(y * w + x);
            }
        }
        return &out[..len];
    }

    for (i, y) in (y0..y1).enumerate() {
        // SAFETY: bounded by asserts above.
        unsafe {
            *out.get_unchecked_mut(i) = *img.get_unchecked(x, y);
        }
    }
    &out[..len]
}

fn gather_col_segment_u16<'a>(
    img: &ImageView<'_, u16>,
    x: usize,
    y0: usize,
    y1: usize,
    out: &'a mut Vec<u16>,
) -> &'a [u16] {
    assert!(x < img.width(), "x out of bounds");
    assert!(y0 <= y1 && y1 <= img.height(), "invalid y-range");

    let len = y1 - y0;
    out.resize(len, 0);

    if img.is_contiguous()
        && let Some(data) = img.as_contiguous_slice()
    {
        let w = img.width();
        // SAFETY: same indexing argument as u8 variant.
        unsafe {
            for (i, y) in (y0..y1).enumerate() {
                *out.get_unchecked_mut(i) = *data.get_unchecked(y * w + x);
            }
        }
        return &out[..len];
    }

    for (i, y) in (y0..y1).enumerate() {
        // SAFETY: bounded by asserts above.
        unsafe {
            *out.get_unchecked_mut(i) = *img.get_unchecked(x, y);
        }
    }
    &out[..len]
}

fn gather_col_segment_f32<'a>(
    img: &ImageView<'_, f32>,
    x: usize,
    y0: usize,
    y1: usize,
    out: &'a mut Vec<f32>,
) -> &'a [f32] {
    assert!(x < img.width(), "x out of bounds");
    assert!(y0 <= y1 && y1 <= img.height(), "invalid y-range");

    let len = y1 - y0;
    out.resize(len, 0.0);

    if img.is_contiguous()
        && let Some(data) = img.as_contiguous_slice()
    {
        let w = img.width();
        // SAFETY: same indexing argument as u8 variant.
        unsafe {
            for (i, y) in (y0..y1).enumerate() {
                *out.get_unchecked_mut(i) = *data.get_unchecked(y * w + x);
            }
        }
        return &out[..len];
    }

    for (i, y) in (y0..y1).enumerate() {
        // SAFETY: bounded by asserts above.
        unsafe {
            *out.get_unchecked_mut(i) = *img.get_unchecked(x, y);
        }
    }
    &out[..len]
}

#[cfg(test)]
mod tests {
    use vm_primitives::Image;
    use vm_primitives::{DoGKernel1D, edge::conv1d::convolve_f32};

    use crate::{ColAccess, Error, LaserExtractConfig, LaserExtractor, ScanAxis};

    fn frac_overlap(i: usize, left: f32, right: f32) -> f32 {
        let x0 = i as f32 - 0.5;
        let x1 = i as f32 + 0.5;
        (x1.min(right) - x0.max(left)).clamp(0.0, 1.0)
    }

    fn blur_rows(img_f: &mut [f32], width: usize, height: usize, sigma: f32) {
        let k = DoGKernel1D::new(sigma);
        let mut tmp = vec![0.0f32; width];
        for y in 0..height {
            let row = &img_f[y * width..(y + 1) * width];
            convolve_f32(
                row,
                &k.g,
                k.radius,
                vm_primitives::BorderMode::Clamp,
                &mut tmp,
            );
            img_f[y * width..(y + 1) * width].copy_from_slice(&tmp);
        }
    }

    fn blur_cols(img_f: &mut [f32], width: usize, height: usize, sigma: f32) {
        let k = DoGKernel1D::new(sigma);
        let mut col = vec![0.0f32; height];
        let mut out = vec![0.0f32; height];
        for x in 0..width {
            for y in 0..height {
                col[y] = img_f[y * width + x];
            }
            convolve_f32(
                &col,
                &k.g,
                k.radius,
                vm_primitives::BorderMode::Clamp,
                &mut out,
            );
            for y in 0..height {
                img_f[y * width + x] = out[y];
            }
        }
    }

    fn transpose_u8(img: &Image<u8>) -> Image<u8> {
        let w = img.width();
        let h = img.height();
        let mut out = vec![0u8; w * h];
        let src = img.data();
        for y in 0..h {
            for x in 0..w {
                out[x * h + y] = src[y * w + x];
            }
        }
        Image::from_vec(h, w, out).expect("valid transposed image")
    }

    #[test]
    fn rows_mode_basic() {
        let (w, h) = (64usize, 40usize);
        let (x_l, x_r) = (20.3f32, 25.7f32);

        let mut img_f = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                img_f[y * w + x] = 255.0 * frac_overlap(x, x_l, x_r);
            }
        }
        blur_rows(&mut img_f, w, h, 0.8);

        let img_u8: Vec<u8> = img_f
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        let img = Image::from_vec(w, h, img_u8).expect("valid image");

        let mut ext = LaserExtractor::new(1.2);
        let cfg = LaserExtractConfig {
            axis: ScanAxis::Rows,
            ..LaserExtractConfig::default()
        };

        let line = ext
            .extract_line_u8(&img.as_view(), 0..h, &cfg, None)
            .expect("valid extractor arguments");

        let exp_c = 0.5 * (x_l + x_r);
        let exp_w = x_r - x_l;
        let mut valid = 0usize;
        let mut sum_err = 0.0f32;
        let mut sum_werr = 0.0f32;
        for s in &line.samples {
            if s.valid {
                valid += 1;
                sum_err += (s.center - exp_c).abs();
                sum_werr += (s.width - exp_w).abs();
            }
        }

        assert!(valid >= h - 2);
        let mean_err = sum_err / valid as f32;
        let mean_werr = sum_werr / valid as f32;
        assert!(mean_err <= 0.15);
        assert!(mean_werr <= 0.4);
    }

    #[test]
    fn cols_mode_basic_gather() {
        let (w, h) = (64usize, 40usize);
        let (y_l, y_r) = (10.2f32, 15.6f32);

        let mut img_f = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                img_f[y * w + x] = 255.0 * frac_overlap(y, y_l, y_r);
            }
        }
        blur_cols(&mut img_f, w, h, 0.8);

        let img_u8: Vec<u8> = img_f
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        let img = Image::from_vec(w, h, img_u8).expect("valid image");

        let mut ext = LaserExtractor::new(1.2);
        let cfg = LaserExtractConfig {
            axis: ScanAxis::Cols {
                access: ColAccess::Gather,
            },
            ..LaserExtractConfig::default()
        };

        let line = ext
            .extract_line_u8(&img.as_view(), 0..w, &cfg, None)
            .expect("valid extractor arguments");

        let exp_c = 0.5 * (y_l + y_r);
        let mut valid = 0usize;
        let mut sum_err = 0.0f32;
        for s in &line.samples {
            if s.valid {
                valid += 1;
                sum_err += (s.center - exp_c).abs();
            }
        }

        assert!(valid >= w - 2);
        assert!(sum_err / valid as f32 <= 0.15);
    }

    #[test]
    fn sloped_gaps_reflections_rows_and_cols() {
        let (w, h) = (96usize, 72usize);
        let stripe_w = 5.2f32;

        let mut img_rows = vec![0.0f32; w * h];
        for y in 0..h {
            let main_center = 18.0 + 0.22 * (y as f32);
            let x_l = main_center - stripe_w * 0.5;
            let x_r = main_center + stripe_w * 0.5;

            if !(28..=34).contains(&y) {
                for x in 0..w {
                    img_rows[y * w + x] += 255.0 * frac_overlap(x, x_l, x_r);
                }
            }

            if y % 3 == 0 {
                let refl_c = main_center + 13.0;
                let rl = refl_c - stripe_w * 0.4;
                let rr = refl_c + stripe_w * 0.4;
                for x in 0..w {
                    img_rows[y * w + x] += 90.0 * frac_overlap(x, rl, rr);
                }
            }
        }
        blur_rows(&mut img_rows, w, h, 0.8);
        let img_rows_u8: Vec<u8> = img_rows
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        let img_r = Image::from_vec(w, h, img_rows_u8).expect("valid image");

        let mut ext = LaserExtractor::new(1.2);
        let cfg_rows = LaserExtractConfig {
            axis: ScanAxis::Rows,
            max_gap_scans: 4,
            max_jump_px: 8.0,
            ..LaserExtractConfig::default()
        };
        let line_r = ext
            .extract_line_u8(&img_r.as_view(), 0..h, &cfg_rows, None)
            .expect("valid extractor arguments");

        let gap_valid = line_r.samples[28..=34].iter().filter(|s| s.valid).count();
        assert!(gap_valid <= 2);
        assert!(line_r.samples[40..].iter().filter(|s| s.valid).count() >= 20);

        let mut err_sum = 0.0f32;
        let mut err_count = 0usize;
        for s in &line_r.samples {
            if s.valid && !(28..=34).contains(&s.scan_i) {
                let true_c = 18.0 + 0.22 * (s.scan_i as f32);
                err_sum += (s.center - true_c).abs();
                err_count += 1;
            }
        }
        assert!(err_count > 20);
        assert!(err_sum / err_count as f32 <= 0.5);

        // Build the analogous horizontal/sloped case for Cols.
        let mut img_cols = vec![0.0f32; w * h];
        for x in 0..w {
            let main_center = 12.0 + 0.16 * (x as f32);
            let y_l = main_center - stripe_w * 0.5;
            let y_r = main_center + stripe_w * 0.5;

            if !(22..=28).contains(&x) {
                for y in 0..h {
                    img_cols[y * w + x] += 255.0 * frac_overlap(y, y_l, y_r);
                }
            }

            if x % 4 == 0 {
                let refl_c = main_center + 11.5;
                let rl = refl_c - stripe_w * 0.4;
                let rr = refl_c + stripe_w * 0.4;
                for y in 0..h {
                    img_cols[y * w + x] += 90.0 * frac_overlap(y, rl, rr);
                }
            }
        }
        blur_cols(&mut img_cols, w, h, 0.8);
        let img_cols_u8: Vec<u8> = img_cols
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        let img_c = Image::from_vec(w, h, img_cols_u8).expect("valid image");

        let cfg_cols = LaserExtractConfig {
            axis: ScanAxis::Cols {
                access: ColAccess::Gather,
            },
            max_gap_scans: 4,
            max_jump_px: 8.0,
            ..LaserExtractConfig::default()
        };
        let line_c = ext
            .extract_line_u8(&img_c.as_view(), 0..w, &cfg_cols, None)
            .expect("valid extractor arguments");

        let gap_valid_c = line_c.samples[22..=28].iter().filter(|s| s.valid).count();
        assert!(gap_valid_c <= 2);
        assert!(line_c.samples[34..].iter().filter(|s| s.valid).count() >= 30);
    }

    #[test]
    fn cols_transposed_matches_gather() {
        let (w, h) = (64usize, 40usize);
        let (y_l, y_r) = (10.2f32, 15.6f32);

        let mut img_f = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                img_f[y * w + x] = 255.0 * frac_overlap(y, y_l, y_r);
            }
        }
        blur_cols(&mut img_f, w, h, 0.8);

        let img_u8: Vec<u8> = img_f
            .iter()
            .map(|&v| v.round().clamp(0.0, 255.0) as u8)
            .collect();
        let img = Image::from_vec(w, h, img_u8).expect("valid image");
        let img_t = transpose_u8(&img);

        let mut ext = LaserExtractor::new(1.2);
        let cfg_g = LaserExtractConfig {
            axis: ScanAxis::Cols {
                access: ColAccess::Gather,
            },
            ..LaserExtractConfig::default()
        };
        let out_g = ext
            .extract_line_u8(&img.as_view(), 0..w, &cfg_g, None)
            .expect("valid extractor arguments");

        let cfg_t = LaserExtractConfig {
            axis: ScanAxis::Cols {
                access: ColAccess::Transposed,
            },
            ..LaserExtractConfig::default()
        };
        let out_t = ext
            .extract_line_u8(&img.as_view(), 0..w, &cfg_t, Some(&img_t.as_view()))
            .expect("valid extractor arguments");

        assert_eq!(out_g.samples.len(), out_t.samples.len());
        for (a, b) in out_g.samples.iter().zip(out_t.samples.iter()) {
            assert_eq!(a.valid, b.valid);
            if a.valid {
                assert!((a.center - b.center).abs() <= 0.05);
            }
        }
    }

    #[test]
    fn transposed_access_without_a_transposed_image_is_an_error() {
        // Forgetting the argument is ordinary API misuse, not a broken internal
        // invariant, so it is reported rather than asserted.
        let img = Image::from_vec(8, 4, vec![0u8; 8 * 4]).expect("valid image");
        let mut ext = LaserExtractor::new(1.0);
        let cfg = LaserExtractConfig {
            axis: ScanAxis::Cols {
                access: ColAccess::Transposed,
            },
            ..Default::default()
        };
        assert!(matches!(
            ext.extract_line_u8(&img.as_view(), 0..8, &cfg, None),
            Err(Error::InvalidConfig(_))
        ));
    }

    #[test]
    fn transposed_image_with_wrong_dimensions_is_an_error() {
        let img = Image::from_vec(8, 4, vec![0u8; 8 * 4]).expect("valid image");
        // Correct transpose would be 4x8; this is not.
        let bad = Image::from_vec(8, 4, vec![0u8; 8 * 4]).expect("valid image");
        let mut ext = LaserExtractor::new(1.0);
        let cfg = LaserExtractConfig {
            axis: ScanAxis::Cols {
                access: ColAccess::Transposed,
            },
            ..Default::default()
        };
        assert!(matches!(
            ext.extract_line_u8(&img.as_view(), 0..8, &cfg, Some(&bad.as_view())),
            Err(Error::InvalidConfig(_))
        ));
    }
}
