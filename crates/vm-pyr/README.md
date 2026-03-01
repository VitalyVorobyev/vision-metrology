Ultra-fast 2x2 mean image pyramid for coarse-to-fine processing.

`vm-pyr` implements a fixed 2x2 mean (box-filter) downsample — no Gaussian blur, no fractional weights. Each level is exactly half the width and half the height of the previous level. The hot downsampling path is allocation-free: `PyramidF32` pre-allocates all level buffers at construction time and reuses them across `build_from_u8` calls. The pyramid is the foundation for `vm-multiscale` and any other coarse-to-fine algorithm in the workspace.

## Quick start

```rust
use vm_pyr::{PyramidF32, downsample2x2_mean_u8};
use vm_core::Image;

// Build a 4-level pyramid from a u8 image
let src: Image<u8> = Image::new(1280, 1024);  // fill with your data
let mut pyr = PyramidF32::new(1280, 1024, 4);
pyr.build_from_u8(&src);

// Access level 2 (320x256 f32 image)
let level2 = pyr.level(2);
println!("level 2 size: {}x{}", level2.width(), level2.height());
```

## Key public types

| Type / Function | Description |
|---|---|
| `PyramidF32` | Reusable multi-level f32 image pyramid |
| `downsample2x2_mean_u8` | Single-step 2x2 mean downsample from u8 to f32 |
