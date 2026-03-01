Multi-scale 2-D edge detection across a Gaussian pyramid.

`vm-multiscale` pairs `vm-pyr`'s `PyramidF32` with `vm-edge`'s `Edge2DDetector` at each pyramid level. After detection, all edgel positions are mapped back to level-0 (full-resolution) pixel coordinates by multiplying by the corresponding scale factor, so consumers receive a single unified edgel list regardless of the number of pyramid levels used. Gradient thresholds are automatically scaled per level so that coarse-level detections are not dominated by noise. An optional duplicate-suppression pass removes redundant detections that appear at multiple scales.

## Quick start

```rust
use vm_multiscale::{MultiScaleEdgeDetector, MultiScaleConfig};
use vm_core::Image;

let img: Image<u8> = Image::new(512, 512);
// ... fill with your image data ...

let config = MultiScaleConfig { num_levels: 3, ..Default::default() };
let mut det = MultiScaleEdgeDetector::new(config);
let edgels = det.detect_u8(&img);
println!("detected {} edgels across 3 scales", edgels.len());
```

## Key public types

| Type | Description |
|---|---|
| `MultiScaleEdgeDetector` | Stateful detector; owns pyramid and per-level edge detectors |
| `MultiScaleConfig` | Number of levels, per-level thresholds, duplicate suppression |
| `ScaleAnnotatedEdgel` | Edgel with level-0 coordinates and originating scale index |
