//! Forward (analytic) and inverse (iterative) Brown-Conrady distortion, and
//! the linear pinhole intrinsics map between normalized and pixel
//! coordinates.
//!
//! **Forward is cheap, inverse is not.** [`distort_pixel`] is a handful of
//! FMAs — it is what builds a `dst → src` [`warp::Map`](crate::warp::Map)
//! (the map needs the source pixel for a given ideal/undistorted
//! destination point, i.e. the forward model). [`undistort_pixel`] is a
//! fixed-iteration-count Newton/fixed-point solve — reserve it for
//! point-wise measurements (a caliper hit, a detected feature), never for
//! building a per-pixel map.

use vm_primitives::Point2f;

use super::types::{BrownConrady5, CameraModel, PinholeIntrinsics};

/// Iterations [`undistort_pixel`] runs. Brown-Conrady's fixed-point
/// undistortion iteration is a contraction for any distortion magnitude a
/// real lens produces; empirically (see `metric` tests) it is converged to
/// `< 1e-6` in normalized-coordinate units well before 10 iterations even at
/// strong distortion, so 20 leaves a wide margin at negligible cost (a few
/// FMAs each).
const UNDISTORT_ITERS: u32 = 20;

/// Map normalized camera coordinates to pixel coordinates through the
/// pinhole intrinsics: `u = fx*x + skew*y + cx`, `v = fy*y + cy`.
#[inline]
pub fn normalized_to_pixel(intrinsics: &PinholeIntrinsics, normalized: Point2f) -> Point2f {
    Point2f::new(
        intrinsics.fx * normalized.x + intrinsics.skew * normalized.y + intrinsics.cx,
        intrinsics.fy * normalized.y + intrinsics.cy,
    )
}

/// Invert [`normalized_to_pixel`] — the **linear** part only, no
/// distortion. Closed-form (no iteration): the intrinsics matrix is upper
/// triangular.
#[inline]
pub fn pixel_to_normalized_linear(intrinsics: &PinholeIntrinsics, pixel: Point2f) -> Point2f {
    let y = (pixel.y - intrinsics.cy) / intrinsics.fy;
    let x = (pixel.x - intrinsics.cx - intrinsics.skew * y) / intrinsics.fx;
    Point2f::new(x, y)
}

/// Apply Brown-Conrady distortion to normalized coordinates, returning the
/// distorted normalized coordinates (still pre-intrinsics).
#[inline]
fn distort_normalized(dist: &BrownConrady5, x: f32, y: f32) -> (f32, f32) {
    let r2 = x * x + y * y;
    let radial = 1.0 + r2 * (dist.k1 + r2 * (dist.k2 + r2 * dist.k3));
    let xd = x * radial + 2.0 * dist.p1 * x * y + dist.p2 * (r2 + 2.0 * x * x);
    let yd = y * radial + dist.p1 * (r2 + 2.0 * y * y) + 2.0 * dist.p2 * x * y;
    (xd, yd)
}

/// Forward distortion model: normalized (undistorted) camera coordinates →
/// distorted **pixel** coordinates.
///
/// Analytic and cheap — this is the function to build a `dst → src`
/// undistortion [`warp::Map`](crate::warp::Map) with (see
/// [`undistort_map`](super::undistort_map)): a destination (ideal) pixel's
/// normalized ray, run forward through distortion, gives the pixel to
/// sample in the raw (distorted) source image.
///
/// ```
/// use vision_metrology::metric::{CameraModel, distort_pixel};
/// use vision_metrology::Point2f;
///
/// let camera = CameraModel::default(); // fx=fy=1, no distortion
/// let p = distort_pixel(&camera, Point2f::new(0.1, -0.2));
/// assert!((p.x - 0.1).abs() < 1e-6 && (p.y - (-0.2)).abs() < 1e-6);
/// ```
#[inline]
pub fn distort_pixel(camera: &CameraModel, normalized: Point2f) -> Point2f {
    let (xd, yd) = distort_normalized(&camera.distortion, normalized.x, normalized.y);
    normalized_to_pixel(&camera.intrinsics, Point2f::new(xd, yd))
}

/// Inverse distortion model: distorted **pixel** coordinates → normalized
/// (undistorted) camera coordinates.
///
/// Runs a fixed 20-step fixed-point iteration
/// (`x_{n+1} = (xd - tangential(x_n, y_n)) / radial(x_n, y_n)`, the standard
/// OpenCV-style Brown-Conrady inverse) seeded at the distorted point itself.
/// There is no early-exit and no configurable iteration count — the model
/// carries no such field (see [`BrownConrady5`]'s docs) precisely so this
/// function's cost and convergence behaviour are the same for every camera.
///
/// # Convergence
/// This is a fixed-point iteration on a contraction for any distortion a
/// real lens produces; it is not a general Newton solve and can diverge for
/// pathological (non-physical) coefficients. Convergence is verified on the
/// golden-fixture round-trip test (`< 1e-6` in normalized-coordinate units
/// over a grid, well within 20 iterations).
#[inline]
pub fn undistort_pixel(camera: &CameraModel, pixel: Point2f) -> Point2f {
    let distorted = pixel_to_normalized_linear(&camera.intrinsics, pixel);
    let (xd, yd) = (distorted.x, distorted.y);
    let mut x = xd;
    let mut y = yd;
    for _ in 0..UNDISTORT_ITERS {
        let r2 = x * x + y * y;
        let radial = 1.0
            + r2 * (camera.distortion.k1 + r2 * (camera.distortion.k2 + r2 * camera.distortion.k3));
        let dx = 2.0 * camera.distortion.p1 * x * y + camera.distortion.p2 * (r2 + 2.0 * x * x);
        let dy = camera.distortion.p1 * (r2 + 2.0 * y * y) + 2.0 * camera.distortion.p2 * x * y;
        x = (xd - dx) / radial;
        y = (yd - dy) / radial;
    }
    Point2f::new(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::types::PinholeIntrinsics;

    fn camera_with_distortion() -> CameraModel {
        CameraModel {
            intrinsics: PinholeIntrinsics {
                fx: 800.0,
                fy: 810.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            distortion: BrownConrady5 {
                k1: -0.12,
                k2: 0.02,
                k3: 0.0,
                p1: 0.001,
                p2: -0.0005,
            },
        }
    }

    #[test]
    fn undistort_inverts_distort_over_a_grid() {
        let camera = camera_with_distortion();
        let mut max_err = 0.0f32;
        for iy in -5..=5 {
            for ix in -5..=5 {
                let x = ix as f32 * 0.06;
                let y = iy as f32 * 0.06;
                let pixel = distort_pixel(&camera, Point2f::new(x, y));
                let back = undistort_pixel(&camera, pixel);
                let err = ((back.x - x).powi(2) + (back.y - y).powi(2)).sqrt();
                max_err = max_err.max(err);
            }
        }
        assert!(max_err < 1e-6, "round-trip max error {max_err}");
    }

    #[test]
    fn zero_distortion_is_the_identity_map() {
        let camera = CameraModel {
            intrinsics: PinholeIntrinsics {
                fx: 500.0,
                fy: 500.0,
                cx: 100.0,
                cy: 100.0,
                skew: 0.0,
            },
            distortion: BrownConrady5::default(),
        };
        let n = Point2f::new(0.05, -0.03);
        let pixel = distort_pixel(&camera, n);
        assert_eq!(pixel, normalized_to_pixel(&camera.intrinsics, n));
        let back = undistort_pixel(&camera, pixel);
        assert!((back.x - n.x).abs() < 1e-6);
        assert!((back.y - n.y).abs() < 1e-6);
    }

    #[test]
    fn pixel_to_normalized_linear_inverts_normalized_to_pixel_with_skew() {
        let intrinsics = PinholeIntrinsics {
            fx: 700.0,
            fy: 690.0,
            cx: 400.0,
            cy: 300.0,
            skew: 5.0,
        };
        let n = Point2f::new(0.2, -0.15);
        let pixel = normalized_to_pixel(&intrinsics, n);
        let back = pixel_to_normalized_linear(&intrinsics, pixel);
        assert!((back.x - n.x).abs() < 1e-6);
        assert!((back.y - n.y).abs() < 1e-6);
    }
}
