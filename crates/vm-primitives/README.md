# vm-primitives

Low-level building blocks for industrial machine-vision metrology: image views and
sampling, geometry types, an image pyramid, subpixel edge detection, and binary
morphology. Pure Rust — no OpenCV, no FFI.

Most users should depend on [`vision-metrology`](../vision-metrology) instead, which
re-exports this crate in full alongside the high-level algorithms.

## Modules

| Module | Content |
|---|---|
| `core` | `Image` / `ImageView` / `ImageViewMut`, nearest and bilinear sampling, `BorderMode`, `Point2f` / `Vec2f` / `Rect2f` / `Angle`, nalgebra type aliases (`Isometry2f`, `Similarity2f`, `Affine2f`, `Projective2f`), and the shared `Error` type |
| `pyr` | `Pyramid` — 2×2 mean downsample generic over pixel type, drop-odd policy, optional binomial pre-smooth, buffers reused across calls |
| `edge` | `Edge1DDetector` (DoG), `Edge2DDetector` (Scharr + NMS + hysteresis) producing subpixel `Edgel`s with unit gradient normals, `GradientBuffers`, and opposite-polarity `EdgePair1D` for laser stripes |
| `morph` | Erode / dilate / open / close over a parameterized `StructuringElement`, Borgefors 3-4-5 chamfer distance, Zhang–Suen thinning |

All names are re-exported flat at the crate root, so `vm_primitives::Edge2DDetector`
and `vm_primitives::edge::Edge2DDetector` both work.

## Conventions

- **Pixel centers.** Integer `i` means coordinate `i as f32`.
- **Element stride**, not byte stride: pixel `(x, y)` is at `y * stride + x`.
- **`Edgel::n`** is a unit normal pointing dark→bright (increasing intensity).
- **Binary images** use `0` for background and `255` for foreground.
- Default border mode is `Clamp`.

## Example

```rust
use vm_primitives::{Edge2DConfig, Edge2DDetector, Image};

// A vertical step edge at x = 32.
let data: Vec<u8> = (0..64 * 64).map(|i| if i % 64 >= 32 { 200 } else { 0 }).collect();
let img = Image::from_vec(64, 64, data).expect("valid image");

let mut det = Edge2DDetector::new();
let edgels = det.detect_u8(&img.as_view(), &Edge2DConfig::default());
assert!(edgels.iter().all(|e| (e.p.x - 31.5).abs() < 1.0));
```

The detector owns its scratch buffers; reuse one instance across frames.

## License

Licensed under either of [Apache License, Version 2.0](../../LICENSE-APACHE) or
[MIT license](../../LICENSE-MIT) at your option.
