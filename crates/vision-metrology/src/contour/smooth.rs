//! 1-D Gaussian smoothing of polylines along arc length.
//!
//! The kernel is applied to each coordinate independently as a weighted sum
//! of neighbours within a `3 * sigma` arc-length window. Boundary behaviour
//! is **clamp**: points near the ends use the endpoint value (i.e. the
//! Gaussian is truncated and re-normalised, not padded with zeros).
//!
//! ## Coordinate convention
//! All positions are in pixel units. The arc-length parameter `s[i]` is the
//! cumulative chord length in pixels, so `sigma` is also expressed in pixels.
//!
//! ## Allocation strategy
//! The function allocates one `Vec<Point2f>` for the output (size = input
//! length) and a fixed-size weight scratch buffer on the stack (capped at
//! [`MAX_KERNEL_PTS`] entries). For polylines longer than `MAX_KERNEL_PTS`
//! points the kernel window simply clamps to that size; this is generous
//! enough for any realistic sigma (3σ window = 768 px at σ = 128 px).

use vm_primitives::Point2f;

/// Maximum number of neighbour points considered per output vertex.
///
/// The Gaussian window spans `±3*sigma` arc-length, which for a typical
/// 1-px sampling and σ = 128 px gives 769 neighbours — safely within this
/// limit. Larger windows are silently capped.
pub const MAX_KERNEL_PTS: usize = 2048;

/// Smooth a polyline with 1-D Gaussian convolution along arc length.
///
/// For each output vertex `i`, the result is the weighted average of all
/// input vertices `j` whose arc-length distance `|s[i] - s[j]|` is within
/// `3 * sigma`, using weight `exp(-0.5 * (s[i]-s[j])^2 / sigma^2)`.
///
/// The kernel is renormalised at each vertex so boundary effects do not
/// create a bias; boundary behaviour is therefore equivalent to **clamp**
/// (the nearest endpoint contributes its full weight even when the Gaussian
/// window extends beyond the polyline).
///
/// # Arguments
/// - `points`: input polyline vertices in order.
/// - `sigma`: Gaussian standard deviation in pixels of arc length. Must be
///   positive; values `<= 0.0` return a copy of the input unchanged.
///
/// # Returns
/// A new `Vec<Point2f>` of the same length as `points`. Returns an empty
/// `Vec` when `points` is empty. Returns a clone when `points` has fewer
/// than 2 vertices or `sigma <= 0.0`.
pub fn smooth_polyline(points: &[Point2f], sigma: f32) -> Vec<Point2f> {
    let n = points.len();
    if n < 2 || sigma <= 0.0 {
        return points.to_vec();
    }

    // --- Cumulative arc-length parameters (s[i] in pixels) ---
    let mut s = Vec::with_capacity(n);
    s.push(0.0f32);
    for i in 1..n {
        let dx = points[i].x - points[i - 1].x;
        let dy = points[i].y - points[i - 1].y;
        s.push(s[i - 1] + (dx * dx + dy * dy).sqrt());
    }

    let radius = 3.0 * sigma; // search radius in arc-length
    let inv_2sigma2 = 0.5 / (sigma * sigma);

    let mut out = Vec::with_capacity(n);

    // Stack-allocated weight scratch buffer (capped at MAX_KERNEL_PTS).
    let mut weights = [0.0f32; MAX_KERNEL_PTS];

    for i in 0..n {
        let si = s[i];

        // Binary search for the window [j_lo, j_hi] such that
        // s[j] is within [si - radius, si + radius].
        let j_lo = {
            let target = si - radius;
            let mut lo = 0usize;
            let mut hi = i;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if s[mid] < target {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let j_hi = {
            let target = si + radius;
            let mut lo = i;
            let mut hi = n;
            while lo < hi {
                let mid = (lo + hi) / 2;
                if s[mid] <= target {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            lo.min(n)
        };

        // Number of neighbours (capped at MAX_KERNEL_PTS).
        let count = (j_hi - j_lo).min(MAX_KERNEL_PTS);

        let mut sum_w = 0.0f32;
        let mut sum_x = 0.0f32;
        let mut sum_y = 0.0f32;

        for (k, w_slot) in weights.iter_mut().enumerate().take(count) {
            let j = j_lo + k;
            let ds = s[j] - si;
            let w = (-inv_2sigma2 * ds * ds).exp();
            *w_slot = w;
            sum_w += w;
            sum_x += w * points[j].x;
            sum_y += w * points[j].y;
        }

        // Zero weights should not happen (the centre point always has w=1),
        // but guard against degenerate input with duplicate positions.
        let (ox, oy) = if sum_w > 1e-12 {
            (sum_x / sum_w, sum_y / sum_w)
        } else {
            (points[i].x, points[i].y)
        };

        // Clear used weight slots (zero-allocation cleanup).
        for w_slot in weights.iter_mut().take(count) {
            *w_slot = 0.0;
        }

        out.push(Point2f { x: ox, y: oy });
    }

    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use vm_primitives::Point2f;

    use super::smooth_polyline;

    /// Build a horizontal straight line of `n` points at y = 0.
    fn hline(n: usize) -> Vec<Point2f> {
        (0..n)
            .map(|i| Point2f {
                x: i as f32,
                y: 0.0,
            })
            .collect()
    }

    #[test]
    fn straight_line_stays_straight_after_smoothing() {
        // A horizontal straight line y=0 should remain at y=0 after smoothing
        // because the Gaussian is symmetric and y is constant. The x-coordinates
        // of interior points (far enough from both ends) are also preserved
        // because the Gaussian-weighted average of an arithmetic sequence
        // centred at index i equals i. Boundary points are pulled inward
        // (clamped boundary: the window is one-sided), which is expected.
        let n = 100usize;
        let sigma = 5.0f32;
        let margin = (3.0 * sigma).ceil() as usize + 2; // interior safety margin
        let pts = hline(n);
        let smoothed = smooth_polyline(&pts, sigma);
        assert_eq!(smoothed.len(), n);

        // All y-values should stay at 0 (symmetric kernel, constant y).
        for sm in &smoothed {
            assert!(sm.y.abs() < 1e-4, "y changed: sm.y={:.4}", sm.y);
        }

        // Interior x-values should be preserved.
        for i in margin..(n - margin) {
            assert!(
                (smoothed[i].x - pts[i].x).abs() < 0.1,
                "interior x changed at i={i}: orig={:.4} sm={:.4}",
                pts[i].x,
                smoothed[i].x
            );
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let result = smooth_polyline(&[], 2.0);
        assert!(result.is_empty());
    }

    #[test]
    fn single_point_returns_unchanged() {
        let pts = vec![Point2f { x: 3.0, y: 7.0 }];
        let result = smooth_polyline(&pts, 1.0);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], pts[0]);
    }

    #[test]
    fn zero_sigma_returns_clone() {
        let pts = hline(20);
        let result = smooth_polyline(&pts, 0.0);
        assert_eq!(result.len(), pts.len());
        for (a, b) in pts.iter().zip(result.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn noisy_sine_rms_decreases_after_smoothing() {
        // A noisy sine wave: after smoothing with sigma=3 the RMS error
        // versus the clean sine should decrease compared to the noisy version.
        //
        // Geometry: 200 points on y = sin(x * 2π / 40) + noise,
        // where noise is a deterministic per-pixel splitmix64 hash value in
        // [-0.3, 0.3].
        let n = 200usize;
        let period = 40.0f32;
        let noise_amp = 0.3f32;

        let noisy: Vec<Point2f> = (0..n)
            .map(|i| {
                let x = i as f32;
                let clean_y = (x * 2.0 * core::f32::consts::PI / period).sin();
                // Splitmix64-based deterministic hash.
                let h = splitmix64(i as u64);
                let noise = (h as f32 / u64::MAX as f32) * 2.0 * noise_amp - noise_amp;
                Point2f {
                    x,
                    y: clean_y + noise,
                }
            })
            .collect();

        let clean: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 2.0 * core::f32::consts::PI / period).sin())
            .collect();

        // RMS of noisy signal vs clean.
        let rms_noisy: f32 = (noisy
            .iter()
            .zip(clean.iter())
            .map(|(p, &cy)| (p.y - cy).powi(2))
            .sum::<f32>()
            / n as f32)
            .sqrt();

        let smoothed = smooth_polyline(&noisy, 3.0);

        let rms_smooth: f32 = (smoothed
            .iter()
            .zip(clean.iter())
            .map(|(p, &cy)| (p.y - cy).powi(2))
            .sum::<f32>()
            / n as f32)
            .sqrt();

        assert!(
            rms_smooth < rms_noisy,
            "smoothed RMS ({rms_smooth:.4}) should be less than noisy RMS ({rms_noisy:.4})"
        );
    }

    /// Splitmix64 hash for deterministic noise generation.
    fn splitmix64(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
}
