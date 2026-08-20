//! The runtime bird's-eye / rectify path: a [`Plane3::xy`] raster, its
//! ideal (undistorted) homography, and the `dst → src` [`warp::Map`] that
//! composes it with forward Brown-Conrady distortion.
//!
//! **A pure homography cannot carry Brown-Conrady distortion** — a
//! homography is linear in homogeneous coordinates, distortion is not.
//! [`homography_plane_to_image`] therefore returns the *ideal* (undistorted)
//! mapping only; [`plane_grid_map`] is the one to actually resample a raw
//! camera frame with, since it composes the homography with
//! [`distort_pixel`](super::distort_pixel) per destination pixel.
//!
//! **Measurements should use [`pixel_to_plane`](super::pixel_to_plane), not
//! this path.** `pixel_to_plane` is exact per-point (ray/plane
//! intersection); this module exists for the whole-image case — a live
//! bird's-eye preview or mosaic tile — where resampling every pixel through
//! the exact per-pixel ray would cost far more than a homography-plus-map
//! and buys nothing extra for a display surface.

use nalgebra::Matrix3;

use vm_primitives::{Point2f, Projective2f};

use super::distortion::distort_pixel;
use super::types::{CameraModel, PlaneGrid, Pose3};
use crate::warp::Map;

/// `M`: the (camera-intrinsics-free) `dst grid → camera-frame direction`
/// homogeneous map for `grid` under `pose`.
///
/// `grid` lies on the reference frame's `z = 0` plane by construction (see
/// [`PlaneGrid`]'s docs), so a grid pixel `(col, row)` corresponds to the
/// 3-D point `(origin_mm.x + col*mm_per_px, origin_mm.y + row*mm_per_px, 0)`
/// in the reference frame. Substituting `z = 0` into `pose`'s affine map
/// leaves a homogeneous *linear* map of `(col, row, 1)` — this is exactly
/// what makes the plane-to-image relationship a homography rather than a
/// full 3-D projection.
///
/// Computed in `f64` (invariant 20): the matrix multiply that builds this
/// is one-time setup cost, not a per-pixel one, so there is no reason to
/// pay for `f32`'s precision loss here.
fn plane_to_camera_matrix(pose: &Pose3, grid: &PlaneGrid) -> Matrix3<f64> {
    let iso: nalgebra::Isometry3<f64> = nalgebra::convert(*pose);
    let r = iso.rotation.to_rotation_matrix();
    let col0 = r.matrix().column(0).into_owned();
    let col1 = r.matrix().column(1).into_owned();
    let t = iso.translation.vector;

    let mm_per_px = grid.mm_per_px as f64;
    let origin = nalgebra::Vector2::new(grid.origin_mm.x as f64, grid.origin_mm.y as f64);

    let m0 = col0 * mm_per_px;
    let m1 = col1 * mm_per_px;
    let m2 = col0 * origin.x + col1 * origin.y + t;

    Matrix3::from_columns(&[m0, m1, m2])
}

/// Apply a homogeneous 3x3 map to `(x, y, 1)` and perspective-divide,
/// entirely in `f64`.
#[inline]
fn apply_homogeneous(m: &Matrix3<f64>, x: f32, y: f32) -> Point2f {
    let p = m * nalgebra::Vector3::new(x as f64, y as f64, 1.0);
    Point2f::new((p.x / p.z) as f32, (p.y / p.z) as f32)
}

/// The **ideal** (undistorted) homography mapping [`PlaneGrid`] pixel
/// coordinates `(col, row)` to undistorted image pixel coordinates.
///
/// This cannot include Brown-Conrady distortion (see this module's own docs) —
/// use [`plane_grid_map`] to get the actual `dst → src` map over the raw,
/// distorted camera image.
pub fn homography_plane_to_image(
    camera: &CameraModel,
    pose: &Pose3,
    grid: &PlaneGrid,
) -> Projective2f {
    let m = plane_to_camera_matrix(pose, grid);
    let k = Matrix3::new(
        camera.intrinsics.fx as f64,
        camera.intrinsics.skew as f64,
        camera.intrinsics.cx as f64,
        0.0,
        camera.intrinsics.fy as f64,
        camera.intrinsics.cy as f64,
        0.0,
        0.0,
        1.0,
    );
    let h = k * m;
    let h32 = h.cast::<f32>();
    Projective2f::from_matrix_unchecked(h32)
}

/// The runtime bird's-eye `dst → src` [`Map`]: destination is [`PlaneGrid`]
/// pixel coordinates, source is the **raw** (distorted) camera image.
///
/// Composes [`homography_plane_to_image`]'s geometry with forward
/// distortion ([`distort_pixel`](super::distort_pixel)) per destination
/// pixel — the map [`Map::apply`]/[`Map::apply_with_mask`] actually needs to
/// resample a real camera frame. Apply with `BorderMode::Constant` +
/// [`Map::apply_with_mask`] (same reasoning as
/// [`crate::matching::CropSpec`]'s rectify path): a grid pixel outside the
/// camera's field of view has no real content, and `Clamp` would fabricate
/// texture there.
///
/// For point-wise measurements, use [`pixel_to_plane`](super::pixel_to_plane)
/// instead — it is exact, this is a per-pixel resample built for a
/// whole-image preview/mosaic.
pub fn plane_grid_map(camera: &CameraModel, pose: &Pose3, grid: &PlaneGrid) -> Map {
    let m = plane_to_camera_matrix(pose, grid);
    let camera = *camera;
    Map::from_fn(grid.w, grid.h, move |x, y| {
        let normalized = apply_homogeneous(&m, x, y);
        let distorted = distort_pixel(&camera, normalized);
        (distorted.x, distorted.y)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::types::PinholeIntrinsics;
    use crate::warp::Interp;
    use vm_primitives::{BorderMode, Image, sample_bilinear_f32};

    fn identity_camera() -> CameraModel {
        CameraModel {
            intrinsics: PinholeIntrinsics {
                fx: 1000.0,
                fy: 1000.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            distortion: Default::default(),
        }
    }

    /// With zero distortion, `plane_grid_map`'s per-pixel source coordinate
    /// must equal `homography_plane_to_image`'s — `distort_pixel` reduces to
    /// the plain intrinsics map (pinned by `distortion::zero_distortion_is_the_identity_map`),
    /// so the two paths differ only in *how* they reach that coordinate
    /// (one matrix multiply vs. a homogeneous-map-then-distort closure).
    /// Verified indirectly through `Map::apply`: sample a source ramp with
    /// each and compare, plus an independent hand-computed reference at
    /// each homography-predicted coordinate.
    #[test]
    fn plane_grid_map_matches_homography_with_no_distortion() {
        let camera = identity_camera();
        let pose = Pose3::translation(0.0, 0.0, 1000.0);
        let grid = PlaneGrid {
            origin_mm: Point2f::new(-100.0, -100.0),
            mm_per_px: 1.0,
            w: 8,
            h: 8,
        };

        let h = homography_plane_to_image(&camera, &pose, &grid);
        let map = plane_grid_map(&camera, &pose, &grid);

        // A ramp source large enough to cover the whole projected footprint.
        let (sw, sh) = (640usize, 480usize);
        let src: Image<f32> = Image::from_vec(
            sw,
            sh,
            (0..sw * sh)
                .map(|i| (i % sw) as f32 + 1000.0 * (i / sw) as f32)
                .collect(),
        )
        .unwrap();

        let mut dst = vec![0.0f32; grid.w * grid.h];
        map.apply(
            &src.as_view(),
            &mut dst,
            Interp::Bilinear,
            BorderMode::Constant(0.0),
        )
        .unwrap();

        for y in 0..grid.h {
            for x in 0..grid.w {
                let via_h = h * Point2f::new(x as f32, y as f32);
                let expected = sample_bilinear_f32(
                    &src.as_view(),
                    via_h.x,
                    via_h.y,
                    BorderMode::Constant(0.0),
                );
                let got = dst[y * grid.w + x];
                assert!(
                    (got - expected).abs() < 1e-2,
                    "at ({x},{y}): map gave {got}, homography-derived reference {expected}"
                );
            }
        }
    }

    #[test]
    fn plane_grid_map_center_maps_to_principal_point_when_centered() {
        let camera = identity_camera();
        let pose = Pose3::translation(0.0, 0.0, 1000.0);
        // Grid centered under the camera: pixel (w/2, h/2)'s plane point is
        // (0, 0), which the identity intrinsics maps to (cx, cy).
        let grid = PlaneGrid {
            origin_mm: Point2f::new(-50.0, -50.0),
            mm_per_px: 1.0,
            w: 100,
            h: 100,
        };
        let h = homography_plane_to_image(&camera, &pose, &grid);
        let center = h * Point2f::new(50.0, 50.0);
        assert!((center.x - camera.intrinsics.cx).abs() < 1e-2);
        assert!((center.y - camera.intrinsics.cy).abs() < 1e-2);
    }
}
