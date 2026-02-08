use std::ops::Range;

use vm_core::{BorderMode, ImageView, Point2f};
use vm_edge::{Edge1DConfig, Edge1DDetector, EdgePair1D, EdgePeak, EdgePolarity, SubpixRefine};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanAxis {
    Rows,
    Cols { access: ColAccess },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColAccess {
    Gather,
    Transposed,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LaserSample {
    pub scan_i: usize,
    pub center: f32,
    pub width: f32,
    pub score: f32,
    pub left: f32,
    pub right: f32,
    pub valid: bool,
}

#[derive(Debug, Clone)]
pub struct LaserLine {
    pub axis: ScanAxis,
    pub samples: Vec<LaserSample>,
    pub points: Vec<Point2f>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CoarseMethod {
    Max,
    CenterOfMass {
        half_width: usize,
        threshold_frac: f32,
    },
}

#[derive(Debug, Clone)]
pub struct LaserExtractConfig {
    pub axis: ScanAxis,
    pub coarse: CoarseMethod,
    pub roi_half_width: usize,
    pub max_jump_px: f32,
    pub max_gap_scans: usize,
    pub min_score: f32,
    pub min_width: f32,
    pub max_width: f32,
    pub edge_cfg: Edge1DConfig,
    pub prior_weight: f32,
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

#[derive(Debug, Clone)]
pub struct LaserExtractor {
    detector: Edge1DDetector,
    col_u8: Vec<u8>,
    col_u16: Vec<u16>,
    col_f32: Vec<f32>,
}

impl LaserExtractor {
    pub fn new(sigma: f32) -> Self {
        Self {
            detector: Edge1DDetector::new(sigma),
            col_u8: Vec::new(),
            col_u16: Vec::new(),
            col_f32: Vec::new(),
        }
    }

    pub fn set_sigma(&mut self, sigma: f32) {
        self.detector.set_sigma(sigma);
    }

    pub fn extract_line_u8(
        &mut self,
        img: &ImageView<'_, u8>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
        transposed: Option<&ImageView<'_, u8>>,
    ) -> LaserLine {
        self.detector.set_sigma(cfg.edge_cfg.sigma);
        let mut samples = match cfg.axis {
            ScanAxis::Rows => self.extract_rows_u8_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Gather,
            } => self.extract_cols_gather_u8_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Transposed,
            } => {
                let img_t =
                    transposed.expect("transposed image is required for ColAccess::Transposed");
                assert_eq!(
                    img_t.width(),
                    img.height(),
                    "transposed width must match original height"
                );
                assert_eq!(
                    img_t.height(),
                    img.width(),
                    "transposed height must match original width"
                );
                self.extract_rows_u8_samples(img_t, scan_range.clone(), cfg)
            }
        };

        if cfg.enable_smoothing {
            smooth_valid_centers(&mut samples);
        }

        let points = build_points(&samples, cfg.axis);
        LaserLine {
            axis: cfg.axis,
            samples,
            points,
        }
    }

    pub fn extract_line_u16(
        &mut self,
        img: &ImageView<'_, u16>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
        transposed: Option<&ImageView<'_, u16>>,
    ) -> LaserLine {
        self.detector.set_sigma(cfg.edge_cfg.sigma);
        let mut samples = match cfg.axis {
            ScanAxis::Rows => self.extract_rows_u16_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Gather,
            } => self.extract_cols_gather_u16_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Transposed,
            } => {
                let img_t =
                    transposed.expect("transposed image is required for ColAccess::Transposed");
                assert_eq!(
                    img_t.width(),
                    img.height(),
                    "transposed width must match original height"
                );
                assert_eq!(
                    img_t.height(),
                    img.width(),
                    "transposed height must match original width"
                );
                self.extract_rows_u16_samples(img_t, scan_range.clone(), cfg)
            }
        };

        if cfg.enable_smoothing {
            smooth_valid_centers(&mut samples);
        }

        let points = build_points(&samples, cfg.axis);
        LaserLine {
            axis: cfg.axis,
            samples,
            points,
        }
    }

    pub fn extract_line_f32(
        &mut self,
        img: &ImageView<'_, f32>,
        scan_range: Range<usize>,
        cfg: &LaserExtractConfig,
        transposed: Option<&ImageView<'_, f32>>,
    ) -> LaserLine {
        self.detector.set_sigma(cfg.edge_cfg.sigma);
        let mut samples = match cfg.axis {
            ScanAxis::Rows => self.extract_rows_f32_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Gather,
            } => self.extract_cols_gather_f32_samples(img, scan_range.clone(), cfg),
            ScanAxis::Cols {
                access: ColAccess::Transposed,
            } => {
                let img_t =
                    transposed.expect("transposed image is required for ColAccess::Transposed");
                assert_eq!(
                    img_t.width(),
                    img.height(),
                    "transposed width must match original height"
                );
                assert_eq!(
                    img_t.height(),
                    img.width(),
                    "transposed height must match original width"
                );
                self.extract_rows_f32_samples(img_t, scan_range.clone(), cfg)
            }
        };

        if cfg.enable_smoothing {
            smooth_valid_centers(&mut samples);
        }

        let points = build_points(&samples, cfg.axis);
        LaserLine {
            axis: cfg.axis,
            samples,
            points,
        }
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

pub fn coarse_center_u8(line: &[u8], coarse: &CoarseMethod) -> Option<f32> {
    coarse_center_u8_in_range(line, coarse, 0, line.len())
}

pub fn coarse_center_u16(line: &[u16], coarse: &CoarseMethod) -> Option<f32> {
    coarse_center_u16_in_range(line, coarse, 0, line.len())
}

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
    use vm_core::Image;
    use vm_edge::{DoGKernel1D, conv1d::convolve_f32};

    use crate::{ColAccess, LaserExtractConfig, LaserExtractor, ScanAxis};

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
            convolve_f32(row, &k.g, k.radius, vm_core::BorderMode::Clamp, &mut tmp);
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
            convolve_f32(&col, &k.g, k.radius, vm_core::BorderMode::Clamp, &mut out);
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

        let line = ext.extract_line_u8(&img.as_view(), 0..h, &cfg, None);

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

        let line = ext.extract_line_u8(&img.as_view(), 0..w, &cfg, None);

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
        let line_r = ext.extract_line_u8(&img_r.as_view(), 0..h, &cfg_rows, None);

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
        let line_c = ext.extract_line_u8(&img_c.as_view(), 0..w, &cfg_cols, None);

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
        let out_g = ext.extract_line_u8(&img.as_view(), 0..w, &cfg_g, None);

        let cfg_t = LaserExtractConfig {
            axis: ScanAxis::Cols {
                access: ColAccess::Transposed,
            },
            ..LaserExtractConfig::default()
        };
        let out_t = ext.extract_line_u8(&img.as_view(), 0..w, &cfg_t, Some(&img_t.as_view()));

        assert_eq!(out_g.samples.len(), out_t.samples.len());
        for (a, b) in out_g.samples.iter().zip(out_t.samples.iter()) {
            assert_eq!(a.valid, b.valid);
            if a.valid {
                assert!((a.center - b.center).abs() <= 0.05);
            }
        }
    }
}
