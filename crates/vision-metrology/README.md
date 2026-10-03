# vision-metrology

Algorithms for industrial machine-vision metrology:
- shape-based object detection;
- calipers and robust primitive fitting;
- the pixel → millimetre calibration bridge;
- image warping, cross-correlation and displacement;
- contour topology, laser stripe extraction, segmentation and line segments.

Pure Rust: no OpenCV, no FFI.

The crate re-exports [`vm-primitives`](https://github.com/VitalyVorobyev/vision-metrology/tree/main/crates/vm-primitives),
so it is the only dependency you need.

```toml
[dependencies]
vision-metrology = "0.1"
```

## Modules

| Module | Content |
|---|---|
| `contour` | `ContourGraph`: junction-aware topology (T/Y junctions, loops) built from edgels, with per-edge tangent, curvature, arc-length parameterization, and Gaussian polyline smoothing |
| `corr` | Cross-correlation matching over `corrmatch` (`CorrTemplate`, `find`, `find_topk`) plus inter-frame `displacement` with optional Lucas–Kanade refinement |
| `fit` | `fit_line` / `fit_circle` / `fit_ellipse`: algebraic start then geometric refinement, optional `RobustLoss` (Huber/Tukey) and `RansacConfig`; every `Fit<M>` reports `rms` / `max_dev` / `n_used` |
| `laser` | `LaserExtractor`: laser stripe centrelines from opposite-polarity 1-D edge pairs, scanning rows or columns, with ROI and prior tracking |
| `lsd` | `LsdDetector`: line-segment detection with NFA validation |
| `matching` | `ShapeModel` + `ShapeMatcher`: gradient-orientation similarity, coarse-to-fine search over translation / rotation / uniform scale, subpixel pose refinement, masked teaching, and canonical-pose crops (`matching::crop`) |
| `measure` | `Caliper` (rect / arc / radial / strip placements; gradient-peak, midpoint or half-contrast edge location) and `MetrologyModel`: measure a located part and fit the result, with a typed `RejectReason` when a caliper finds nothing, `diagnostics::explain` to trace one measurement, and `diagnostics::layout` for caliper placement |
| `metric` | `CameraModel` / `Pose3` / `Plane3` / `PlaneGrid`, exact `pixel_to_plane`, `plane_grid_map` / `undistort_map` for whole images, importers for calibration-rs and `table_calibration` JSON |
| `scale` | Scale estimation for `matching` (moments / log-polar) and `find_scale_invariant`: estimate once, resample the model, verify in a narrow band |
| `segment` | Otsu and adaptive thresholding, connected-component labeling with per-component stats, watershed, edgel region growing |
| `warp` | `Map`: a precomputed `dst → src` coordinate table (affine / projective / polar / log-polar / `from_fn`) with `apply` / `apply_with_mask` and a validity mask |

## Features

Each module is a Cargo feature, and all are on by default. Build only what you use:

```toml
vision-metrology = { version = "0.1", default-features = false, features = ["matching", "measure"] }
```

| Feature | Enables | Implies |
|---|---|---|
| `contour` | `contour` | — |
| `corr` | `corr` | — |
| `fit` | `fit` | — |
| `laser` | `laser` | — |
| `lsd` | `lsd` | — |
| `matching` | `matching` | `warp` (rectified crops build a `warp::Map`) |
| `measure` | `measure` | `fit` (measured points are fitted) |
| `metric` | `metric` | `warp` (whole-image maps are `warp::Map`s); pulls in `serde_json` for the importers |
| `scale` | `scale` | `corr`, `matching`, `segment`, `warp` |
| `segment` | `segment` | `contour` (region growing consumes a `ContourGraph`) |
| `warp` | `warp` | — |
| `serde` | `ShapeModel` save/load | `matching` |

## Importing

`use vision_metrology::prelude::*;` brings in the working set, including `vm-primitives`'
own prelude. Every name also lives at its module path, for example
`vision_metrology::contour::ContourGraph`. The most-used `vm-primitives` names (`Image`,
`Edge2DDetector`, `Pyramid`, geometry) are re-exported at this crate's root. The whole lower
crate is reachable as `vision_metrology::vm_primitives`.

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
let stats = component_stats(&labels, 16);
assert_eq!(stats.len(), 2);
for c in stats {
    println!("component {}: {} px, centroid ({:.1}, {:.1})",
             c.label, c.pixel_count, c.centroid.x, c.centroid.y);
}
// component 1: 1024 px, centroid (31.5, 31.5)
// component 2: 1024 px, centroid (95.5, 87.5)
```

## Guides and examples

- [Shape-based object detection](https://github.com/VitalyVorobyev/vision-metrology/blob/main/docs/shape-matching.md)
- [Measuring a located part](https://github.com/VitalyVorobyev/vision-metrology/blob/main/docs/measure.md)
- [Performance and accuracy](https://github.com/VitalyVorobyev/vision-metrology/blob/main/docs/performance.md)

Runnable programs are in
[`examples/`](https://github.com/VitalyVorobyev/vision-metrology/tree/main/crates/vision-metrology/examples):

| Example | Shows |
|---|---|
| `pyramid` | Building and inspecting an image pyramid |
| `edge_1d` / `edge_2d` | Subpixel 1-D and 2-D edge detection |
| `contour_graph` | Contour topology, junctions, curvature |
| `morphology` | Erode / dilate / open / close, chamfer distance |
| `line_segments` | LSD line-segment detection |
| `segmentation` | Thresholding, labeling, component statistics |
| `shape_matching` | Building a shape model and locating it, rotated, in a scene |
| `measure_circles` | Circle metrology: calipers, robust circle fit, `rms` / `max_dev` gating |
| `laserline` | Laser stripe extraction from a multi-snap image (`--input`) |
| `inspect_canend` | Locate → fixture → measure → pass/fail on a directory of frames |
| `align_crops` | Teach → find → rectify into canonical model-frame crops |
| `pose_audit` | Independent ZNCC cross-check of recovered poses, diagnostic overlays |
| `birdseye_mosaic` | Bird's-eye composite of two calibrated cameras |
| `caliperbench_run` | Strip calipers over a CaliperBench requests file (its JSONL protocol) |

```bash
cargo run -p vision-metrology --example measure_circles
cargo run -p vision-metrology --example laserline -- --help
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
