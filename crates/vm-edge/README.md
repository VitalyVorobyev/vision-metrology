Subpixel 1-D and 2-D edge detection using derivative-of-Gaussian (DoG) convolution.

`vm-edge` provides two detectors. `Edge1DDetector` applies a 1-D DoG kernel along a signal vector and returns subpixel peak positions — the building block for laser stripe extraction. `Edge2DDetector` runs Scharr gradient convolution on a full image, applies non-maximum suppression (NMS), hysteresis thresholding, and refines each surviving edgel to subpixel accuracy along the gradient normal. `GradientBuffers` are reusable scratch allocations that eliminate per-frame heap pressure in the 2-D hot path. Edge-pair primitives (opposite-polarity bright/dark pairs) feed `vm-laser` directly.

## Quick start

```rust
use vm_edge::{Edge2DDetector, Edge2DConfig};
use vm_core::Image;

// 64x64 image: left half dark, right half bright — vertical step edge
let mut img: Image<u8> = Image::new(64, 64);
for y in 0..64_u32 {
    for x in 32..64_u32 {
        img[(y, x)] = 200;
    }
}

let config = Edge2DConfig::default();
let mut det = Edge2DDetector::new(config);
let edgels = det.detect_u8(&img);
println!("detected {} edgels", edgels.len());
if let Some(e) = edgels.first() {
    println!("first edgel: ({:.2}, {:.2}), strength={:.2}", e.x, e.y, e.strength);
}
```

## Key public types

| Type | Description |
|---|---|
| `Edge1DDetector` | 1-D DoG edge detector for signal vectors |
| `Edge2DDetector` | Full-image Scharr + NMS + subpixel 2-D edgel detector |
| `Edgel` | Subpixel edge point with position, orientation, and strength |
| `EdgePeak` | 1-D subpixel peak with polarity |
| `GradientBuffers` | Reusable Gx/Gy/magnitude scratch buffers |
