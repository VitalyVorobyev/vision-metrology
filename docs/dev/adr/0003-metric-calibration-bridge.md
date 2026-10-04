# ADR-0003: `metric`, the calibration bridge from pixels to millimetres

- Status: Accepted
- Date: 2026-08-20

## Context

Measurements have to be reported in millimetres on a physical plane, through a camera
calibration. Calibration itself (estimating intrinsics, distortion and extrinsics) lives in
`calibration-rs`. That crate is structurally pinned to nalgebra 0.34 through its solver
chain, and its nalgebra types cross its public API. Its MSRV is also above this workspace's.

A wrong convention here (units, pose direction, plane basis) produces plausible-looking
numbers with the right shape and wrong values, not a crash.

## Decision

- **Offline/runtime split.** `calibration-rs` stays the offline calibration system. `metric`
  mirrors only the parameter types (`CameraModel`, `Pose3`, `Plane3`, `PlaneGrid`) on
  nalgebra 0.35 and loads calibration-rs JSON exports. A golden-file test pins the import.
- **Millimetres throughout.** Every 3-D and plane quantity is in mm, and pixel quantities
  stay in pixels. Importers convert at the boundary. A format that documents no unit is
  interpreted at its importer, with the reasoning written next to the code.
- **`Pose3` is camera-from-reference (`T_C_R`)**, matching calibration-rs: `pose * p_ref` is
  `p_ref` in the camera frame.
- **Two plane-to-image paths, deliberately different.**
  - `pixel_to_plane` is exact (undistort, back-project, intersect with the plane) for any
    `Plane3`. Use it for point-wise measurement.
  - `plane_grid_map` / `undistort_map` are the whole-image runtime path: a homography
    composed with forward distortion per destination pixel, built as a `warp::Map`.
  - A homography cannot carry Brown–Conrady distortion, and its linearity needs `z = 0`. The
    whole-image path is therefore restricted to a `PlaneGrid` on the reference frame's own
    `xy` plane, and both entry points say so.
- **The in-plane basis of `pixel_to_plane` is deterministic.** The origin is the plane's
  closest point to the reference origin. The axes come from projecting the reference `x`
  axis onto the plane (or `y` when `x` is nearly parallel to the normal), completed
  right-handed. At `n = (0, 0, 1)` it reduces to `PlaneGrid`'s own axes.
- **Planar 3-D is handled by rectifying first**, then matching in the rectified view.
  `tests/metric_rectify.rs` checks it on a synthetic tilt sweep of 0–40°: the model is found
  at every tilt, at the same position to a small fraction of a pixel.

## Alternatives

- **A direct `calibration-rs` dependency.** Blocked on its nalgebra version. It will replace
  the mirror when upstream moves.
- **Metres, the wire format's unit.** Rejected because every downstream tolerance in this
  domain is stated in millimetres.
- **One homography path for everything.** Rejected because it cannot represent lens
  distortion, and it is singular for planes through the camera centre.
- **Homography refinement of the matched pose** (Hofhauser–Steger). Rectify-first leaves
  little for it to gain on planar targets. It is kept for non-planar cases.

## Consequences

- The mirrored types must track calibration-rs's export schema until the direct dependency
  lands.
- A bird's-eye view of a tilted or offset plane has no shortcut. The caller re-expresses the
  plane in a frame where it is `z = 0`.
