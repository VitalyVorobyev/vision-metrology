# vision-metrology

High-level algorithms for industrial machine-vision metrology: shape-based object
detection, calipers and robust primitive fitting, contour topology, laser stripe
extraction, segmentation, image warping, and the pixel → millimetre calibration
bridge. Pure Rust — no OpenCV, no FFI.

This crate re-exports [`vm-primitives`](../vm-primitives) in full, so it is the only
dependency you need.

```toml
[dependencies]
vision-metrology = "0.1"
```

## Modules

| Module | Content |
|---|---|
| `contour` | `ContourGraph` — junction-aware topology (T/Y junctions, loops) built from edgels, with per-edge tangent, curvature, arc-length parameterization, and Gaussian polyline smoothing |
| `corr` | Cross-correlation matching over `corrmatch` (`CorrTemplate`, `find`, `find_topk`) plus inter-frame `displacement` with optional Lucas-Kanade refinement |
| `fit` | `fit_line` / `fit_circle` / `fit_ellipse` — algebraic-init then geometric refine, optional `RobustLoss` (Huber/Tukey) and `RansacConfig`, every `Fit<M>` reports `rms` / `max_dev` / `n_used` |
| `laser` | `LaserExtractor` — laser stripe centerlines from opposite-polarity 1-D edge pairs, scanning rows or columns, with ROI and prior tracking |
| `lsd` | `LsdDetector` — line-segment detection with NFA validation |
| `matching` | `ShapeModel` + `ShapeMatcher` — gradient-orientation similarity, coarse-to-fine search over translation / rotation / uniform scale, occlusion-proportional scoring, subpixel pose refinement, masked teaching, and canonical-pose crops (`matching::crop`) |
| `measure` | `Caliper` (rect / arc / radial placements) + `MetrologyModel` — measure a located part and fit the result, typed `RejectReason` on a caliper that finds nothing, `diagnostics::layout` for caliper placement |
| `metric` | The calibration bridge: `CameraModel` / `Pose3` / `Plane3` / `PlaneGrid`, exact `pixel_to_plane`, runtime `plane_grid_map` / `undistort_map`, importers for calibration-rs and `table_calibration` JSON |
| `scale` | Scale estimation for `matching` (moments / log-polar) and `find_scale_invariant` — estimate once, resample the model, verify in a narrow band |
| `segment` | Otsu and adaptive thresholding, connected-component labeling with per-component stats, watershed, edgel region growing |
| `warp` | `Map` — a precomputed `dst → src` coordinate table (affine / projective / polar / log-polar / `from_fn`) with `apply` / `apply_with_mask` and a first-class validity mask |

Every module is a default-on Cargo feature. The full architectural map, including
`vm-primitives`, is [`docs/system-design.md`](../../docs/system-design.md#layering).

The `vm_primitives` crate most callers need by name — `Image`, `Edge2DDetector`,
`Pyramid`, morphology, geometry — is re-exported at this crate's root as an
explicit curated list; the full lower crate is always reachable as
`vision_metrology::vm_primitives`. Every other name lives at its module path
only (`vision_metrology::contour::ContourGraph`, not a second flattened path) —
`use vision_metrology::prelude::*;` is the convenience for the common set.

## Example

```rust
use vision_metrology::Image;
use vision_metrology::contour::Connectivity;
use vision_metrology::segment::{component_stats, label_connected_components_u8, otsu_threshold_u8};

// Two 32×32 bright squares on a dark background.
let mut data = vec![20u8; 128 * 128];
for (y0, x0) in [(16usize, 16usize), (72, 80)] {
    for y in y0..y0 + 32 {
        for x in x0..x0 + 32 {
            data[y * 128 + x] = 200;
        }
    }
}
let img = Image::from_vec(128, 128, data).expect("valid image");

// `otsu_threshold_u8` returns the threshold value, not a mask.
let t = otsu_threshold_u8(&img.as_view());
let mask: Vec<u8> = img.data().iter().map(|&v| if v > t { 255 } else { 0 }).collect();
let mask = Image::from_vec(128, 128, mask).expect("valid image");

let labels = label_connected_components_u8(&mask.as_view(), Connectivity::C8);
for c in component_stats(&labels, 16) {
    println!("component {}: {} px, centroid ({:.1}, {:.1})",
             c.label, c.pixel_count, c.centroid.x, c.centroid.y);
}
// component 1: 1024 px, centroid (31.5, 31.5)
// component 2: 1024 px, centroid (95.5, 87.5)
```

## Examples

Runnable end-to-end programs in [`examples/`](examples):

| Example | Shows |
|---|---|
| `pyramid` | Building and inspecting an image pyramid |
| `edge_1d` / `edge_2d` | Subpixel 1-D and 2-D edge detection |
| `contour_graph` | Contour topology, junctions, curvature |
| `morphology` | Erode / dilate / open / close, chamfer distance |
| `line_segments` | LSD line-segment detection |
| `inspect_canend` | Locate → fixture → measure → pass/fail on real frames (needs a dataset) |
| `measure_circles` | End-to-end circle metrology: calipers, robust circle fit, `rms` / `max_dev` gating |
| `segmentation` | Thresholding, labeling, component statistics |
| `shape_matching` | Building a shape model and locating it, rotated, in a scene |
| `laserline` | Laser stripe extraction from a multi-snap image (takes `--input`) |
| `align_crops` | Teach → find → rectify into canonical model-frame crops (needs a dataset) |
| `birdseye_mosaic` | Bird's-eye composite of two calibrated cameras (needs a dataset) |
| `pose_audit` | Independent ZNCC cross-check of recovered poses, diagnostic overlays |
| `gen_illustrations` | Regenerates the deterministic PNGs under `docs/assets/` |

```bash
cargo run -p vision-metrology --example measure_circles
cargo run -p vision-metrology --example laserline -- --help
```

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
