Binary morphology, chamfer distance, and skeletonization.

`vm-morph` provides parameterized binary erode and dilate operations with `Square` and `Disk` structuring elements of arbitrary radius, plus the derived `opening` and `closing` compound operations. The 3-4-5 chamfer distance transform approximates Euclidean distance on binary images efficiently without floating-point square roots. Zhang-Suen iterative thinning reduces foreground regions to single-pixel-wide skeletons suitable for centerline extraction.

## Quick start

```rust
use vm_morph::{erode_binary_u8, StructuringElement};
use vm_core::Image;

// Synthetic cross pattern
let mut img: Image<u8> = Image::new(32, 32);
for i in 0..32_u32 { img[(i, 16)] = 255; img[(16, i)] = 255; }

let se = StructuringElement::Square(2);
let eroded = erode_binary_u8(&img, &se);
println!("eroded foreground pixels: {}", eroded.data().iter().filter(|&&v| v > 0).count());
```

## Key public types

| Type / Function | Description |
|---|---|
| `erode_binary_u8` | Binary erosion with parameterized structuring element |
| `dilate_binary_u8` | Binary dilation with parameterized structuring element |
| `chamfer_distance_u8` | 3-4-5 chamfer distance transform on a binary image |
| `thin_binary_u8` | Zhang-Suen iterative skeletonization |
| `StructuringElement` | `Square(r)` or `Disk(r)` SE descriptor |
