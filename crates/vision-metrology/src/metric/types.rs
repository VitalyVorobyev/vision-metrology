//! Canonical calibration types: pinhole intrinsics, Brown-Conrady distortion,
//! a camera model, poses, and the plane types used to turn a pixel into a
//! metric position.

use vm_primitives::{Isometry3f, Point2f, Vec3f};

/// Pinhole camera intrinsics, in pixels.
///
/// `fx`/`fy` are the focal lengths in pixel units, `cx`/`cy` the principal
/// point, and `skew` the (usually zero) pixel-axis non-orthogonality term.
/// The intrinsics matrix this represents is
/// `K = [[fx, skew, cx], [0, fy, cy], [0, 0, 1]]`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PinholeIntrinsics {
    /// Focal length along image x, in pixels.
    pub fx: f32,
    /// Focal length along image y, in pixels.
    pub fy: f32,
    /// Principal point x, in pixels.
    pub cx: f32,
    /// Principal point y, in pixels.
    pub cy: f32,
    /// Pixel-axis skew (usually `0.0`).
    pub skew: f32,
}

/// Brown-Conrady 5-parameter radial-tangential distortion model.
///
/// Applied to **normalized** camera coordinates `(x, y) = (X/Z, Y/Z)`:
///
/// ```text
/// r2 = x*x + y*y
/// radial = 1 + k1*r2 + k2*r2^2 + k3*r2^3
/// xd = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
/// yd = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
/// ```
///
/// There is no stored iteration count: [`undistort_pixel`](super::undistort_pixel)
/// runs a fixed, documented number of Newton iterations rather than reading
/// one from the model — see that function's docs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BrownConrady5 {
    /// Radial coefficient k1.
    pub k1: f32,
    /// Radial coefficient k2.
    pub k2: f32,
    /// Radial coefficient k3.
    pub k3: f32,
    /// Tangential coefficient p1.
    pub p1: f32,
    /// Tangential coefficient p2.
    pub p2: f32,
}

/// A calibrated camera: intrinsics + distortion.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CameraModel {
    /// Pinhole intrinsics (pixels).
    pub intrinsics: PinholeIntrinsics,
    /// Brown-Conrady distortion (normalized-coordinate domain).
    pub distortion: BrownConrady5,
}

impl Default for PinholeIntrinsics {
    fn default() -> Self {
        Self {
            fx: 1.0,
            fy: 1.0,
            cx: 0.0,
            cy: 0.0,
            skew: 0.0,
        }
    }
}

/// A camera pose: **camera-from-reference**.
///
/// `Pose3 = Isometry3f`. For a point `p_ref` expressed in the reference
/// frame (the rig/table frame calibration was solved in), `pose * p_ref`
/// gives the same point in the **camera's** frame — i.e. `pose` is what
/// calibration-rs calls `cam_se3_rig` / `T_C_R`. This is the direction every
/// function in this module expects and every importer in
/// [`io`](super::io) produces; getting it backwards silently mirrors every
/// downstream projection through the origin.
///
/// Units are documented per invariant: this module works in **millimetres**.
/// The reference frame is whatever the calibration solved for (typically a
/// rig or table frame with a physical target on it); `io` importers convert
/// meters (`RigExtrinsicsExport`) to mm on import.
pub type Pose3 = Isometry3f;

/// An oriented plane in the reference frame: `n · X + d = 0`.
///
/// `n` is expected to be a **unit** vector; `d` is the signed distance from
/// the reference-frame origin to the plane divided by `|n|` (i.e. exactly
/// the offset term of the plane equation, not a physical distance in any
/// other sense once `n` is normalized).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Plane3 {
    /// Unit plane normal, in the reference frame.
    pub n: Vec3f,
    /// Signed offset: the plane is `{X : n·X + d = 0}`.
    pub d: f32,
}

impl Plane3 {
    /// The reference frame's `z = 0` plane (`n = (0, 0, 1)`, `d = 0`) — the
    /// default plane assumed by [`homography_plane_to_image`](super::homography_plane_to_image)
    /// and [`plane_grid_map`](super::plane_grid_map).
    #[inline]
    pub fn xy() -> Self {
        Self {
            n: Vec3f::new(0.0, 0.0, 1.0),
            d: 0.0,
        }
    }
}

/// A metric raster laid out on the reference frame's `z = 0` plane.
///
/// `PlaneGrid` describes a `w × h` grid of pixels covering plane
/// coordinates `[origin_mm.x, origin_mm.x + w*mm_per_px) ×
/// [origin_mm.y, origin_mm.y + h*mm_per_px)`, where the plane's own `(x, y)`
/// coordinates are the reference frame's `x`/`y` directly (the grid always
/// sits on [`Plane3::xy`] — see [`homography_plane_to_image`](super::homography_plane_to_image)'s
/// docs for why a general [`Plane3`] is not threaded through the
/// homography/warp path).
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlaneGrid {
    /// Plane-frame coordinate of grid pixel `(0, 0)`'s corner, in mm.
    pub origin_mm: Point2f,
    /// Millimetres per grid pixel (uniform in x and y).
    pub mm_per_px: f32,
    /// Grid width in pixels.
    pub w: usize,
    /// Grid height in pixels.
    pub h: usize,
}
