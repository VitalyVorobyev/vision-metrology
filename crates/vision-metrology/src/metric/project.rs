//! Pixel → 3-D ray → plane intersection: the exact (non-homography) metric
//! path. Prefer this for point-wise measurements; see [`super::homography_plane_to_image`]
//! / [`super::plane_grid_map`] for the runtime bird's-eye/rectify path.

use vm_primitives::{Point2f, Point3f, Vec3f};

use super::distortion::undistort_pixel;
use super::types::{CameraModel, Plane3, Pose3};

/// Unit ray direction in the **camera frame**, for a distorted pixel.
///
/// Runs [`undistort_pixel`] to recover the normalized (undistorted) ray
/// direction `(x, y, 1)`, then normalizes. The ray originates at the camera
/// center (the camera frame's origin).
#[inline]
pub fn pixel_to_ray(camera: &CameraModel, pixel: Point2f) -> Vec3f {
    let n = undistort_pixel(camera, pixel);
    Vec3f::new(n.x, n.y, 1.0).normalize()
}

/// Intersect the ray `origin + t * dir` (`t >= 0`) with `plane`.
///
/// Returns `None` when the ray is parallel to the plane (`|n · dir|` below
/// a small epsilon) or when the intersection lies **behind** the ray origin
/// (`t < 0`) — a pixel whose back-projected ray cannot physically see the
/// plane it was asked to intersect.
///
/// Runs in `f64` internally (invariant 20: accumulation f64) since a
/// near-glancing ray/plane angle can lose precision in `f32` well before the
/// division.
pub fn ray_plane_intersect(origin: Point3f, dir: Vec3f, plane: &Plane3) -> Option<Point3f> {
    let n = plane.n.cast::<f64>();
    let o = origin.coords.cast::<f64>();
    let d = dir.cast::<f64>();
    let denom = n.dot(&d);
    if denom.abs() < 1e-9 {
        return None;
    }
    let t = -(n.dot(&o) + plane.d as f64) / denom;
    if !t.is_finite() || t < 0.0 {
        return None;
    }
    let p = o + d * t;
    Some(Point3f::new(p.x as f32, p.y as f32, p.z as f32))
}

/// Deterministic in-plane orthonormal basis `(u, v)` for `plane`, such that
/// `(u, v, plane.n)` is right-handed.
///
/// Picks the reference frame's `x` axis as the seed unless it is nearly
/// parallel to `plane.n` (dot product magnitude `> 0.9`), in which case `y`
/// is used instead — this keeps the basis well-conditioned (never
/// projecting a near-degenerate seed) and, since the seed choice is a pure
/// function of `plane.n`, fully deterministic and reproducible across calls.
fn plane_basis(plane: &Plane3) -> (Vec3f, Vec3f) {
    let n = plane.n.cast::<f64>();
    let seed_x = nalgebra::Vector3::new(1.0f64, 0.0, 0.0);
    let seed_y = nalgebra::Vector3::new(0.0f64, 1.0, 0.0);
    let seed = if seed_x.dot(&n).abs() > 0.9 {
        seed_y
    } else {
        seed_x
    };
    let u = (seed - n * seed.dot(&n)).normalize();
    let v = n.cross(&u);
    (
        Vec3f::new(u.x as f32, u.y as f32, u.z as f32),
        Vec3f::new(v.x as f32, v.y as f32, v.z as f32),
    )
}

/// Project a pixel onto `plane` (via [`pixel_to_ray`] +
/// [`ray_plane_intersect`]) and express the intersection in the plane's own
/// 2-D basis.
///
/// **Basis.** The plane's origin is its point closest to the reference
/// frame's origin, `p0 = -d · n` (valid since `n` is unit); its axes are
/// this module's own deterministic in-plane `(u, v)` basis. The returned `Point2f` is
/// `((P - p0)·u, (P - p0)·v)` for the intersection point `P`, in the
/// reference frame's own length units (mm, per this module's convention).
///
/// Returns `None` when [`ray_plane_intersect`] does (ray parallel to the
/// plane, or the intersection is behind the camera).
///
/// This is the **exact** metric path — prefer it for point-wise
/// measurements (a caliper hit, a fitted circle center). It is not what
/// [`super::plane_grid_map`] uses: that path is restricted to
/// [`Plane3::xy`] and expressed as a homography purely so it can compose
/// with [`crate::warp::Map`] for a whole-image rectify, not because the
/// underlying geometry differs.
pub fn pixel_to_plane(
    camera: &CameraModel,
    pose: &Pose3,
    plane: &Plane3,
    pixel: Point2f,
) -> Option<Point2f> {
    let dir_cam = pixel_to_ray(camera, pixel);
    let cam_from_ref = pose;
    let ref_from_cam = cam_from_ref.inverse();
    let origin_ref = ref_from_cam * Point3f::origin();
    let dir_ref = ref_from_cam * dir_cam;

    let hit = ray_plane_intersect(origin_ref, dir_ref, plane)?;

    let (u, v) = plane_basis(plane);
    let p0 = plane.n * (-plane.d);
    let local = hit - Point3f::from(p0);
    Some(Point2f::new(local.dot(&u), local.dot(&v)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::types::PinholeIntrinsics;

    fn identity_camera() -> CameraModel {
        CameraModel {
            intrinsics: PinholeIntrinsics {
                fx: 1.0,
                fy: 1.0,
                cx: 0.0,
                cy: 0.0,
                skew: 0.0,
            },
            distortion: Default::default(),
        }
    }

    #[test]
    fn pixel_to_ray_of_principal_point_is_the_optical_axis() {
        let camera = identity_camera();
        let ray = pixel_to_ray(&camera, Point2f::new(0.0, 0.0));
        assert!((ray - Vec3f::new(0.0, 0.0, 1.0)).norm() < 1e-6);
    }

    #[test]
    fn ray_plane_intersect_axis_aligned() {
        // Camera looking down +z, plane z=500 (n=(0,0,1), d=-500).
        let origin = Point3f::origin();
        let dir = Vec3f::new(0.0, 0.0, 1.0);
        let plane = Plane3 {
            n: Vec3f::new(0.0, 0.0, 1.0),
            d: -500.0,
        };
        let hit = ray_plane_intersect(origin, dir, &plane).expect("hits the plane");
        assert!((hit.z - 500.0).abs() < 1e-4);
        assert!(hit.x.abs() < 1e-4 && hit.y.abs() < 1e-4);
    }

    #[test]
    fn ray_plane_intersect_behind_camera_is_none() {
        let origin = Point3f::origin();
        let dir = Vec3f::new(0.0, 0.0, 1.0);
        let plane = Plane3 {
            n: Vec3f::new(0.0, 0.0, 1.0),
            d: 500.0, // plane at z = -500, behind a +z-looking ray
        };
        assert!(ray_plane_intersect(origin, dir, &plane).is_none());
    }

    #[test]
    fn ray_plane_intersect_parallel_is_none() {
        let origin = Point3f::new(0.0, 0.0, 0.0);
        let dir = Vec3f::new(1.0, 0.0, 0.0); // along the plane
        let plane = Plane3 {
            n: Vec3f::new(0.0, 0.0, 1.0),
            d: -500.0,
        };
        assert!(ray_plane_intersect(origin, dir, &plane).is_none());
    }

    #[test]
    fn pixel_to_plane_identity_pose_xy_plane() {
        // Camera at origin looking down +z, 1000mm above z=0 (n·X + d = 0
        // with n=(0,0,1) means d=0 for the z=0 plane).
        let camera = identity_camera();
        let pose = Pose3::translation(0.0, 0.0, 1000.0); // camera-from-reference
        let plane = Plane3::xy();

        // Pixel (0,0) -> normalized (0,0) -> ray straight down +z from the
        // camera center. In reference frame the camera sits at z=-1000
        // (inverse of translation by +1000 along z), looking toward z=0.
        let p =
            pixel_to_plane(&camera, &pose, &plane, Point2f::new(0.0, 0.0)).expect("hits the plane");
        assert!(p.x.abs() < 1e-4 && p.y.abs() < 1e-4);

        // An off-axis pixel: normalized (0.1, 0.2) at z=1000mm distance ->
        // plane offset (100, 200) mm (pinhole similar triangles).
        let p2 =
            pixel_to_plane(&camera, &pose, &plane, Point2f::new(0.1, 0.2)).expect("hits the plane");
        assert!((p2.x - 100.0).abs() < 1e-2, "x = {}", p2.x);
        assert!((p2.y - 200.0).abs() < 1e-2, "y = {}", p2.y);
    }
}
