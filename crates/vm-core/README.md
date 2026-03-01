Core image, geometry, and sampling primitives for the vision-metrology workspace.

`vm-core` provides the foundational types used across every crate in the workspace. `Image<T>` owns a flat pixel buffer and exposes read-only `ImageView<T>` slices with pixel-center coordinate semantics (pixel `i` has center coordinate `i as f32`). Border modes — `Clamp`, `Zero`, and `Wrap` — govern out-of-bounds sampling throughout the pipeline. Point, vector, and rect types are thin nalgebra aliases so that downstream crates never need to re-import nalgebra directly. The shared `Error` type propagates across all workspace crates.

## Quick start

```rust
use vm_core::{Image, BorderMode, Point2f};

// Create a 64x64 u8 image
let mut img: Image<u8> = Image::new(64, 64);
img[(10, 20)] = 255;

// Get a read-only view and sample with Clamp border
let view = img.view();
let val = view.sample_clamp(10.0, 20.0);
println!("val = {val}");

// Geometry helpers
let p = Point2f::new(10.0, 20.0);
println!("point: ({}, {})", p.x, p.y);
```

## Key public types

| Type | Description |
|---|---|
| `Image<T>` | Owned 2-D pixel buffer |
| `ImageView<T>` | Borrowed read-only view into an `Image<T>` |
| `Point2f` | 2-D point alias (`nalgebra::Point2<f32>`) |
| `Vec2f` | 2-D vector alias (`nalgebra::Vector2<f32>`) |
| `Rect2f` | Axis-aligned rectangle with f32 coordinates |
| `BorderMode` | `Clamp` / `Zero` / `Wrap` out-of-bounds policy |
| `Error` | Workspace-wide error type |
