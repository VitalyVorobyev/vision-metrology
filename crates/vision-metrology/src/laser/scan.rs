//! The scan loops, shared by every pixel type.
//!
//! `u8`, `u16` and `f32` extraction used to be six hand-copied ~60-line
//! functions (rows × 3 types, gathered columns × 3 types). They are now two
//! generic loops over [`Pixel`]; the compiler emits the same specialised code
//! the hand-written versions did.

use std::ops::Range;

use vm_primitives::{Edge1DDetector, ImageView, Pixel};

use super::coarse::{coarse_center_in_range, roi_bounds};
use super::gather::gather_col_segment;
use super::pairing::detect_pair;
use super::postprocess::invalid_sample;
use super::types::{LaserExtractConfig, LaserSample};

/// The row and column loops share this bookkeeping verbatim; only line
/// acquisition differs between them.
struct Tracker {
    last_valid_center: Option<f32>,
    gap_len: usize,
}

impl Tracker {
    fn new() -> Self {
        Self {
            last_valid_center: None,
            gap_len: 0,
        }
    }

    fn tracking(&self, cfg: &LaserExtractConfig) -> bool {
        self.last_valid_center.is_some() && self.gap_len <= cfg.max_gap_scans
    }

    fn accept(&mut self, samples: &mut Vec<LaserSample>, s: LaserSample) {
        self.last_valid_center = Some(s.center);
        self.gap_len = 0;
        samples.push(s);
    }

    fn miss(&mut self, samples: &mut Vec<LaserSample>, scan_i: usize, predicted: Option<f32>) {
        samples.push(invalid_sample(scan_i, predicted));
        self.gap_len += 1;
    }
}

/// Row-direction scan: full-row detection with a coarse prior.
///
/// Note the deliberate asymmetry with the column path: rows hand the *entire*
/// row slice to the detector (contiguous memory, the fastest mode), while the
/// column path detects only inside the gathered ROI segment.
pub(super) fn extract_rows_samples<P: Pixel>(
    detector: &mut Edge1DDetector,
    img: &ImageView<'_, P>,
    scan_range: Range<usize>,
    cfg: &LaserExtractConfig,
) -> Vec<LaserSample> {
    assert!(
        scan_range.end <= img.height(),
        "scan range out of row bounds"
    );
    let mut samples = Vec::with_capacity(scan_range.end.saturating_sub(scan_range.start));
    let mut tr = Tracker::new();

    for scan_i in scan_range {
        let line = img.row(scan_i);
        let n = line.len();

        let tracking = tr.tracking(cfg);
        let predicted = if tracking {
            let last = tr.last_valid_center.expect("checked");
            let (b0, b1) = roi_bounds(last, cfg.roi_half_width, n);
            coarse_center_in_range(line, &cfg.coarse, b0, b1).unwrap_or(last)
        } else {
            match coarse_center_in_range(line, &cfg.coarse, 0, n) {
                Some(v) => v,
                None => {
                    tr.miss(&mut samples, scan_i, None);
                    continue;
                }
            }
        };

        match detect_pair(detector, line, predicted, tracking, cfg, 0, scan_i) {
            Some(s) => tr.accept(&mut samples, s),
            None => tr.miss(&mut samples, scan_i, Some(predicted)),
        }
    }

    samples
}

/// Column-direction scan with per-line gathering into a reusable buffer.
pub(super) fn extract_cols_gather_samples<P: Pixel>(
    detector: &mut Edge1DDetector,
    img: &ImageView<'_, P>,
    scan_range: Range<usize>,
    cfg: &LaserExtractConfig,
) -> Vec<LaserSample> {
    assert!(
        scan_range.end <= img.width(),
        "scan range out of col bounds"
    );
    let mut samples = Vec::with_capacity(scan_range.end.saturating_sub(scan_range.start));
    // One allocation per call, alongside the output vector — not per scan line.
    // A typed scratch cannot live in the extractor without making the pixel
    // type part of its signature.
    let mut col_buf: Vec<P> = Vec::with_capacity(img.height());
    let col_buf = &mut col_buf;
    let mut tr = Tracker::new();

    for scan_i in scan_range {
        let n = img.height();
        let tracking = tr.tracking(cfg);

        let predicted = if tracking {
            let last = tr.last_valid_center.expect("checked");
            let (b0, b1) = roi_bounds(last, cfg.roi_half_width, n);
            let line = gather_col_segment(img, scan_i, b0, b1, col_buf);
            coarse_center_in_range(line, &cfg.coarse, 0, line.len())
                .map(|v| v + b0 as f32)
                .unwrap_or(last)
        } else {
            let line = gather_col_segment(img, scan_i, 0, n, col_buf);
            match coarse_center_in_range(line, &cfg.coarse, 0, line.len()) {
                Some(v) => v,
                None => {
                    tr.miss(&mut samples, scan_i, None);
                    continue;
                }
            }
        };

        let (roi0, roi1) = roi_bounds(predicted, cfg.roi_half_width, n);
        if roi1.saturating_sub(roi0) < 3 {
            tr.miss(&mut samples, scan_i, Some(predicted));
            continue;
        }

        let roi_line = gather_col_segment(img, scan_i, roi0, roi1, col_buf);
        match detect_pair(detector, roi_line, predicted, tracking, cfg, roi0, scan_i) {
            Some(s) => tr.accept(&mut samples, s),
            None => tr.miss(&mut samples, scan_i, Some(predicted)),
        }
    }

    samples
}
