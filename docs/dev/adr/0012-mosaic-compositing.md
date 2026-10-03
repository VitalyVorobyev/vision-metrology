# ADR-0012: Mosaics pick one camera per pixel and never blend

- Status: Accepted
- Date: 2026-08-20

## Context

A bird's-eye composite of several calibrated cameras is assembled from
`metric::plane_grid_map` per camera plus `warp::Map::apply_with_mask`. The only new logic is
the compositing rule. The composite exists to be measured on.

## Decision

- **Not a library module.** The rule is about 40 lines, and no library code consumes it. It
  lives in `examples/birdseye_mosaic.rs` and the lab.
- **Nearest-camera-centre priority.** Among the cameras whose mask covers a destination
  pixel, the composite takes the camera whose reprojection of that plane point lands closest
  to its own principal point. Ties go to the lower camera index. This is deterministic and
  favours each camera where its optics are best.
- **No blending.** A `source_id` map (camera index, 255 where nothing covers the pixel) is
  first-class output, so every composite pixel traces to exactly one calibration. Feathering
  exists only as an opt-in, display-only mode.

## Alternatives

- **Averaging or feathering by default.** At a seam, an averaged pixel cannot be attributed to
  one camera's distortion model and extrinsics, exactly where independently calibrated views
  disagree most.
- **A `mosaic` module.** It would be more API surface than the logic it wraps.

## Consequences

- Seams show exposure differences between cameras. Gain compensation is in the backlog.
- The composite is only as good as the plane it is built on. A calibration whose reference
  plane is not the target needs the plane estimated first, as the example does.
