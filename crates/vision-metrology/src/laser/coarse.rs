//! Coarse stripe-centre estimation used to seed the precise DoG search.

use super::scan::ScanPixel;
use super::types::CoarseMethod;

/// Estimate the coarse stripe centre in a `u8` scan-line using the given method.
///
/// Returns the estimated subpixel centre position, or `None` if the signal is
/// too weak to locate a centre.
pub fn coarse_center_u8(line: &[u8], coarse: &CoarseMethod) -> Option<f32> {
    coarse_center_in_range(line, coarse, 0, line.len())
}

/// Estimate the coarse stripe centre in a `u16` scan-line using the given method.
///
/// See [`coarse_center_u8`] for details.
pub fn coarse_center_u16(line: &[u16], coarse: &CoarseMethod) -> Option<f32> {
    coarse_center_in_range(line, coarse, 0, line.len())
}

/// Estimate the coarse stripe centre in an `f32` scan-line using the given method.
///
/// See [`coarse_center_u8`] for details.
pub fn coarse_center_f32(line: &[f32], coarse: &CoarseMethod) -> Option<f32> {
    coarse_center_in_range(line, coarse, 0, line.len())
}

pub(super) fn coarse_center_in_range<T: ScanPixel>(
    line: &[T],
    coarse: &CoarseMethod,
    start: usize,
    end: usize,
) -> Option<f32> {
    if start >= end || end > line.len() {
        return None;
    }

    let (max_idx, max_v) = argmax(&line[start..end]).map(|(i, v)| (start + i, v.to_f32()))?;
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
                let vf = v.to_f32();
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

fn argmax<T: ScanPixel>(line: &[T]) -> Option<(usize, T)> {
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
pub(super) fn roi_bounds(center: f32, half_width: usize, len: usize) -> (usize, usize) {
    if len == 0 {
        return (0, 0);
    }
    let c = center.round() as isize;
    let start = (c - half_width as isize).max(0) as usize;
    let end = (c + half_width as isize + 1).min(len as isize) as usize;
    (start.min(len), end.min(len))
}
