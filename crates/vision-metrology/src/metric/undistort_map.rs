//! Build a whole-image undistortion [`warp::Map`](crate::warp::Map).

use vm_primitives::Point2f;

use super::distortion::{distort_pixel, pixel_to_normalized_linear};
use super::types::CameraModel;
use crate::warp::Map;

/// Build a `dst → src` [`Map`] that undistorts a `w × h` image shot by
/// `camera`.
///
/// **Direction trap (read before touching this function).** A `dst → src`
/// map answers "for this destination (undistorted) pixel, where in the
/// source (raw, distorted) image do I look?" A destination pixel `(x, y)`
/// is first turned into a normalized ray under `camera`'s **linear**
/// intrinsics only ([`pixel_to_normalized_linear`] — no distortion, since
/// the destination image is defined to be the undistorted one), then that
/// same ray is run through the **forward** distortion model
/// ([`distort_pixel`](super::distort_pixel)) to find where it actually
/// landed in the raw sensor image. Using the *inverse* distortion model
/// here instead — the natural-sounding but wrong choice — would answer a
/// different question ("where does this distorted pixel undistort to?") and
/// produce a map that looks plausible (right shape, roughly right pixels)
/// while being subtly, silently wrong. See `crate::warp`'s own module docs
/// for the same `dst → src` convention used by every other map builder in
/// this workspace.
///
/// Destination and source share `camera`'s intrinsics/pixel grid — this
/// builds the "same field of view, distortion removed" map, not a resize.
pub fn undistort_map(camera: &CameraModel, w: usize, h: usize) -> Map {
    let camera = *camera;
    Map::from_fn(w, h, move |x, y| {
        let normalized = pixel_to_normalized_linear(&camera.intrinsics, Point2f::new(x, y));
        let distorted = distort_pixel(&camera, normalized);
        (distorted.x, distorted.y)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::types::{BrownConrady5, PinholeIntrinsics};
    use crate::warp::Interp;
    use vm_primitives::{BorderMode, Image};

    #[test]
    fn zero_distortion_undistort_map_is_the_identity() {
        let camera = CameraModel {
            intrinsics: PinholeIntrinsics {
                fx: 500.0,
                fy: 500.0,
                cx: 32.0,
                cy: 24.0,
                skew: 0.0,
            },
            distortion: BrownConrady5::default(),
        };
        let (w, h) = (64, 48);
        let map = undistort_map(&camera, w, h);
        let src: Image<u8> =
            Image::from_vec(w, h, (0..(w * h)).map(|i| i as u8).collect()).expect("valid image");
        let mut dst = vec![0u8; w * h];
        map.apply(&src.as_view(), &mut dst, Interp::Nearest, BorderMode::Clamp)
            .expect("apply succeeds");
        assert_eq!(dst, src.data());
    }

    #[test]
    fn barrel_distortion_moves_the_corner_source_coordinate_measurably() {
        // A wide-normalized-angle corner (small fx/fy relative to the pixel
        // offset) under strong barrel (negative k1) distortion: the raw
        // source coordinate the map reads for the destination corner must
        // differ from the corner's own coordinate by a clearly nonzero
        // amount — a basic sanity check that the map isn't accidentally an
        // identity/no-op (which an inverted or mis-wired distortion
        // direction could produce for a weak enough case).
        let intrinsics = PinholeIntrinsics {
            fx: 200.0,
            fy: 200.0,
            cx: 32.0,
            cy: 32.0,
            skew: 0.0,
        };
        let camera = CameraModel {
            intrinsics,
            distortion: BrownConrady5 {
                k1: -0.5,
                ..Default::default()
            },
        };
        let corner_normalized = pixel_to_normalized_linear(&intrinsics, Point2f::new(0.0, 0.0));
        let corner_src = distort_pixel(&camera, corner_normalized);
        let diff = ((corner_src.x - 0.0).powi(2) + (corner_src.y - 0.0).powi(2)).sqrt();
        assert!(
            diff > 0.3,
            "corner moved only {diff} px under strong distortion"
        );

        // And the map built from the same camera reproduces that exact
        // per-pixel source coordinate (checked via `Map::apply` on a ramp,
        // since `Map`'s coordinate table itself is not public outside
        // `warp`).
        let map = undistort_map(&camera, 64, 64);
        let src: Image<f32> = Image::from_vec(
            64,
            64,
            (0..64 * 64)
                .map(|i| (i % 64) as f32 + 1000.0 * (i / 64) as f32)
                .collect(),
        )
        .unwrap();
        let mut dst = vec![0.0f32; 64 * 64];
        map.apply(
            &src.as_view(),
            &mut dst,
            Interp::Bilinear,
            BorderMode::Clamp,
        )
        .expect("apply succeeds");
        let expected = vm_primitives::sample_bilinear_f32(
            &src.as_view(),
            corner_src.x,
            corner_src.y,
            BorderMode::Clamp,
        );
        assert!((dst[0] - expected).abs() < 1e-3);
    }
}
