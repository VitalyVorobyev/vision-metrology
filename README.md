[![CI](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml)
[![Security Audit](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml)
[![Publish Rust Docs](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml)

# vision-metrology

High-precision, high-performance image processing for industrial machine-vision
metrology, in pure Rust. Subpixel edges, laser stripe extraction, contour topology,
shape fitting, segmentation, and shape-based object detection — with Python bindings.

No OpenCV, no FFI. All coordinates follow the **pixel-center** convention: integer
`i` means coordinate `i as f32`.

## Crates

| Crate | Description |
|---|---|
| [`vm-primitives`](crates/vm-primitives) | Low-level building blocks: image views and sampling, geometry types, image pyramid, subpixel 1-D/2-D edge detection, binary morphology |
| [`vision-metrology`](crates/vision-metrology) | High-level algorithms built on `vm-primitives`, which it re-exports in full — one dependency is enough |
| [`vm-python`](crates/vm-python) | PyO3 bindings; distributed as `vision-metrology`, imported as `vision_metrology` |

### `vm-primitives` modules

| Module | Content |
|---|---|
| `core` | `Image` / `ImageView` / `ImageViewMut`, sampling and interpolation, border modes, geometry primitives and nalgebra type aliases, the shared `Error` type |
| `pyr` | 2×2 mean image pyramid, generic over pixel type, with optional anti-alias pre-smoothing |
| `edge` | Subpixel 1-D/2-D edge detection (DoG, Scharr), edgels with gradient normals, opposite-polarity edge pairs, dense gradient direction fields |
| `morph` | Binary morphology with parameterized structuring elements, chamfer distance transform, Zhang–Suen thinning |

### `vision-metrology` modules

| Module | Content |
|---|---|
| `contour` | Junction-aware contour graph (T/Y junctions, loops), per-edge tangent and curvature, polyline smoothing |
| `laser` | Laser stripe extraction using opposite-polarity edge pairs, with ROI and prior tracking |
| `matching` | Shape-based object detection: gradient-orientation model, coarse-to-fine search over translation / rotation / scale, subpixel pose refinement — see the [guide](docs/shape-matching.md) |
| `segment` | Otsu and adaptive thresholding, connected-component labeling, watershed, edgel region growing |
| `shape` | LSD line-segment detection, Bookstein/Fitzgibbon conic fitting, RANSAC ellipse fitting |

## Pipeline

```
Image
  │
  ▼
pyr           2×2 mean pyramid — coarse-to-fine levels
  │
  ▼
edge          subpixel 1-D/2-D edge detection, gradient buffers
  │
  ▼
contour       topology graph — T/Y junctions, loops, polyline smoothing
  │
  ├─────►  shape      LSD line segments, conic/ellipse fitting + RANSAC
  │
  ├─────►  segment    thresholding, CCL, watershed, per-component stats
  │
  └─────►  matching   shape model, coarse-to-fine search, pose refinement
```

`laser` consumes `edge` directly — it scans rows or columns for opposite-polarity
1-D edge pairs rather than going through the 2-D pipeline.

## Quick start

```toml
[dependencies]
vision-metrology = "0.1"
```

Requires Rust 1.91 or newer.

```rust
use vision_metrology::{Edge2DConfig, Edge2DDetector, Image};

let img = Image::<u8>::new_fill(64, 64, 0);
let mut det = Edge2DDetector::new();
let edgels = det.detect_u8(&img.as_view(), &Edge2DConfig::default());
```

Runnable examples live in [`crates/vision-metrology/examples/`](crates/vision-metrology/examples):

```bash
cargo run -p vision-metrology --example measure_circles
cargo run -p vision-metrology --example contour_graph
cargo run -p vision-metrology --example shape_matching
cargo run -p vision-metrology --example laserline -- --help
```

## Guides

- [Shape-based object detection](docs/shape-matching.md) — building a shape
  model, choosing a polarity, tuning contrast, reading the score.

Project direction and internals: [system design](docs/system-design.md),
[roadmap](docs/roadmap.md), [backlog](docs/backlog.md).

## Python

Build and install the extension (requires [maturin](https://www.maturin.rs/)):

```bash
cd crates/vm-python
maturin develop   # editable install into the active Python environment
```

```python
import numpy as np
import vision_metrology as vm

det = vm.EdgeDetector(vm.EdgeConfig())
img = np.zeros((64, 64), dtype=np.uint8)
img[:, 32:] = 200
edgels_obj = det.detect_u8(img)
edgels_fn = vm.detect_edges_u8(img, vm.EdgeConfig())
print(edgels_obj[0], edgels_fn[0])
```

- distribution: `vision-metrology`
- import: `vision_metrology`
- wheels are ABI3 (`abi3-py310`) and require Python `>= 3.10`

See [`examples/python/`](examples/python) for runnable end-to-end scripts.

## Contributing

Build, test, lint, and benchmark instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
