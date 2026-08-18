# vision-metrology

High-level algorithms for industrial machine-vision metrology: contour topology,
laser stripe extraction, multi-scale edges, shape fitting, segmentation, and
shape-based object detection. Pure Rust — no OpenCV, no FFI.

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
| `laser` | `LaserExtractor` — laser stripe centerlines from opposite-polarity 1-D edge pairs, scanning rows or columns, with ROI and prior tracking |
| `matching` | `ShapeModel` + `ShapeMatcher` — gradient-orientation similarity, coarse-to-fine search over translation / rotation / uniform scale, occlusion-proportional scoring, subpixel pose refinement |
| `multiscale` | `MultiScaleEdgeDetector` — 2-D edge detection at every pyramid level, merged back to level-0 coordinates |
| `segment` | Otsu and adaptive thresholding, connected-component labeling with per-component stats, watershed, edgel region growing |
| `shape` | `LsdDetector` (line-segment detection with NFA validation), `ConicFitter` (Bookstein / Fitzgibbon), RANSAC ellipse fitting |

Everything from `vm_primitives` — `Image`, `Edge2DDetector`, `PyramidF32`, morphology,
geometry — is re-exported at this crate's root as well, and each module's own types
are re-exported flat. So `vision_metrology::ContourGraph` and
`vision_metrology::contour::ContourGraph` are the same type.

## Example

```rust
use vision_metrology::{
    Connectivity, Image, component_stats, label_connected_components_u8, otsu_threshold_u8,
};

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
| `multiscale_edges` | Merging detections across pyramid levels |
| `contour_graph` | Contour topology, junctions, curvature |
| `morphology` | Erode / dilate / open / close, chamfer distance |
| `line_segments` | LSD line-segment detection |
| `measure_circles` | End-to-end circle metrology with ellipse fitting |
| `segmentation` | Thresholding, labeling, component statistics |
| `shape_matching` | Building a shape model and locating it, rotated, in a scene |
| `laserline` | Laser stripe extraction from a multi-snap image (takes `--input`) |

```bash
cargo run -p vision-metrology --example measure_circles
cargo run -p vision-metrology --example laserline -- --help
```

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
