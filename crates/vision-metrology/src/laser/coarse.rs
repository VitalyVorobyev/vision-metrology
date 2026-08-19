//! Coarse stripe-centre estimation used to seed the precise DoG search.

use super::types::CoarseMethod;
use vm_primitives::Pixel;

/// Estimate the coarse stripe centre inside `[lo, hi)` of a scan line.
///
/// One generic entry point (invariant 19): the `u8`/`u16`/`f32` wrappers that
/// used to sit on top of it were public, allocation-free duplicates with no
/// caller inside the workspace.
pub(super) fn coarse_center_in_range<P: Pixel>(
    line: &[P],
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

fn argmax<P: Pixel>(line: &[P]) -> Option<(usize, P)> {
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
