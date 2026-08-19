//! Sample post-processing shared by every scan mode.

use vm_primitives::Point2f;

use super::types::{LaserSample, ScanAxis};

pub(super) fn invalid_sample(scan_i: usize, predicted: Option<f32>) -> LaserSample {
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

pub(super) fn build_points(samples: &[LaserSample], axis: ScanAxis) -> Vec<Point2f> {
    let mut points = Vec::with_capacity(samples.len());
    for s in samples {
        if !s.valid {
            continue;
        }

        let p = match axis {
            ScanAxis::Rows => Point2f::new(s.center, s.scan_i as f32),
            ScanAxis::Cols { .. } => Point2f::new(s.scan_i as f32, s.center),
        };
        points.push(p);
    }
    points
}

/// Running-median smoothing over each contiguous run of valid samples.
///
/// The window is `2 · half_window + 1`, clipped at the ends of the run. Reads
/// the original centres out of `orig` so the filter is not fed its own output.
pub(super) fn smooth_valid_centers(samples: &mut [LaserSample], half_window: usize) {
    if half_window == 0 {
        return;
    }

    let mut window: Vec<f32> = Vec::with_capacity(2 * half_window + 1);
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
            let j0 = j.saturating_sub(half_window);
            let j1 = (j + half_window).min(run_len - 1);
            window.clear();
            window.extend_from_slice(&orig[j0..=j1]);
            window.sort_by(|a, b| a.partial_cmp(b).expect("finite compare"));
            samples[start + j].center = window[window.len() / 2];
        }
    }
}
