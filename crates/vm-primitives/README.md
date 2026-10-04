# vm-primitives

Low-level building blocks for industrial machine-vision metrology: image views and
sampling, geometry types, an image pyramid, subpixel edge detection, and binary
morphology. Pure Rust: no OpenCV, no FFI.

Most users should depend on
[`vision-metrology`](https://github.com/VitalyVorobyev/vision-metrology/tree/main/crates/vision-metrology)
instead, which re-exports this crate alongside the domain algorithms.

## Modules

| Module | Content |
|---|---|
| `core` | `Image` / `ImageView` / `ImageViewMut` generic over the `Pixel` trait (u8 / u16 / f32), nearest and bilinear sampling, `BorderMode`, nalgebra geometry aliases (`Point2f`, `Vec2f`, `Isometry2f`, `Similarity2f`, `Affine2f`, `Projective2f`), `Rect2f`, `Circle2f`, `Ellipse2f`, and the shared `Error` type |
| `pyr` | `Pyramid`: 2×2 mean downsample generic over pixel type, drop-odd policy, optional binomial pre-smooth, buffers reused across calls; `level_to_base` / `base_to_level` coordinate mapping |
| `edge` | `Edge1DDetector` with a `Derivative1D` (derivative of Gaussian, or Gaussian then central differences) and a `SubpixRefine` (parabolic, log-parabolic `Gaussian3`, centroid), `Edge2DDetector` (Scharr, non-maximum suppression, hysteresis) producing subpixel `Edgel`s with unit gradient normals, `DirectionField` (dense gated gradient directions, optionally filled lazily in tiles), opposite-polarity `EdgePair1D` for laser stripes, and `LevelCrossing1D` (end levels, interpolated level crossings, local half-contrast edges) |
| `morph` | Erode / dilate / open / close over a parameterized `StructuringElement`, Borgefors 3-4-5 chamfer distance, Zhang–Suen thinning |

Names are reachable at their module path and at the crate root; `vm_primitives::prelude`
holds the working set.

The `serde` feature adds serde derives to the geometry and config types.

## Conventions

- **Pixel centres.** Integer `i` means coordinate `i as f32`.
- **Element stride**, not byte stride: pixel `(x, y)` is at `y * stride + x`.
- **`Edgel::n`** is a unit normal pointing dark → bright.
- **Binary images** use `0` for background and `255` for foreground.
- The default border mode is `Clamp`.
- Detectors own their scratch buffers; reuse one instance across frames.

## Example

```rust
use vm_primitives::{Edge2DConfig, Edge2DDetector, Image};

// A vertical step edge between x = 31 and x = 32.
let data: Vec<u8> = (0..64 * 64).map(|i| if i % 64 >= 32 { 200 } else { 0 }).collect();
let img = Image::from_vec(64, 64, data).expect("valid image");

let mut det = Edge2DDetector::new();
let edgels = det.detect(&img.as_view(), &Edge2DConfig::default());
assert!(!edgels.is_empty());
assert!(edgels.iter().all(|e| (e.p.x - 31.5).abs() < 1.0));
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
