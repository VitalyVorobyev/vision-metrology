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

/// Median-of-5 smoothing over each contiguous run of valid samples.
pub(super) fn smooth_valid_centers(samples: &mut [LaserSample]) {
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
