//! The calibration bridge: pixel → millimetre.
//!
//! `metric` mirrors the parameter types a real calibration produces
//! ([`PinholeIntrinsics`], [`BrownConrady5`], [`CameraModel`], [`Pose3`],
//! [`Plane3`], [`PlaneGrid`]) on `nalgebra` 0.35, and turns them into pixel
//! ↔ millimetre conversions. `calibration-rs` remains the **offline**
//! calibration system (it is pinned to nalgebra 0.34 and cannot be a direct
//! dependency here — see `docs/system-design.md`'s "vision-calibration:
//! offline/runtime split"); [`io`] imports its JSON exports.
//!
//! ## Units: this module works in millimetres
//! Every 3-D/plane quantity here (`Pose3` translations, `Plane3::d`,
//! `PlaneGrid::origin_mm`/`mm_per_px`, [`pixel_to_plane`]'s returned
//! `Point2f`) is in **millimetres**. Pixel-domain quantities
//! (`PinholeIntrinsics`, pixel coordinates passed to/from the functions
//! here) stay in pixels, same as the rest of the crate.
//! [`io::import_rig_extrinsics`] converts its source format's meters to mm
//! on import; [`io::import_table_calibration`] is already in mm (documented
//! in that importer, since the source format states no unit).
//!
//! ## `Pose3`: camera-from-reference
//! `Pose3 = Isometry3f`. For a point `p_ref` in the reference frame (the
//! frame the calibration solved for — typically a rig or table frame with a
//! physical target on it), `pose * p_ref` is the same point in the
//! **camera's** frame. This is `cam_se3_rig` / `T_C_R` in calibration-rs's
//! own naming, and every function and importer here follows it — read
//! [`Pose3`]'s own docs before writing a new call site, since the opposite
//! direction is a plausible-looking, silent bug (see the `warp` module's
//! own "direction trap" note for the same shape of mistake).
//!
//! ## Two ways to turn a pixel into a plane position
//! | Path | Function | Exact? | Cost | Use for |
//! |---|---|---|---|---|
//! | Ray/plane | [`pixel_to_plane`] | yes (per-point ray/plane intersection) | one point at a time | point-wise measurements: a caliper hit, a fitted circle center |
//! | Homography + warp | [`homography_plane_to_image`] / [`plane_grid_map`] | approximate off the ideal homography, restricted to [`Plane3::xy`] | O(1) setup, then a per-pixel gather via [`crate::warp::Map`] | whole-image bird's-eye preview/mosaic — not a substitute for the exact path in a measurement |
//!
//! A pure homography cannot carry Brown-Conrady distortion (it is linear in
//! homogeneous coordinates, distortion is not) — see
//! [`homography_plane_to_image`]'s own docs for how the two are composed.
//!
//! ## `f32` storage, `f64` accumulation (invariant 20)
//! Every type here stores `f32`. The iterative undistortion solve, the
//! ray/plane intersection, and the homography construction all accumulate
//! in `f64` internally — plane-adjacent quantities (millimetres, near-zero
//! ray/plane denominators) are exactly the kind of computation invariant 20
//! exists for.

mod distortion;
mod homography;
pub mod io;
mod project;
mod types;
mod undistort_map;

pub use distortion::{
    distort_pixel, normalized_to_pixel, pixel_to_normalized_linear, undistort_pixel,
};
pub use homography::{homography_plane_to_image, plane_grid_map};
pub use project::{pixel_to_plane, pixel_to_ray, ray_plane_intersect};
pub use types::{BrownConrady5, CameraModel, PinholeIntrinsics, Plane3, PlaneGrid, Pose3};
pub use undistort_map::undistort_map;
