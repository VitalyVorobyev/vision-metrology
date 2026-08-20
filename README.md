[![CI](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml)
[![Security Audit](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml)
[![Publish Rust Docs](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml)

# vision-metrology

High-precision, high-performance image processing for industrial machine-vision
metrology, in pure Rust. Subpixel edges, laser stripe extraction, contour topology,
shape fitting, segmentation, and shape-based object detection — with Python bindings.

No OpenCV, no FFI. All coordinates follow the **pixel-center** convention: integer
`i` means coordinate `i as f32`.

<table>
<tr>
<td width="33%"><img src="docs/assets/shape-matching.png" alt="Shape matching"><br>Shape-based matching: model contour found at two poses, scored</td>
<td width="33%"><img src="docs/assets/caliper-anatomy.png" alt="Caliper anatomy"><br>A caliper: the placed box and its cross-averaged 1-D profile</td>
<td width="33%"><img src="docs/assets/laser-stripe.png" alt="Laser stripe extraction"><br>Laser stripe extraction: subpixel centerline over the stripe</td>
</tr>
<tr>
<td width="33%"><img src="docs/assets/circle-fit.png" alt="Robust circle fit"><br>Robust circle fit: noisy points, outliers rejected, residual whiskers</td>
<td width="33%"><img src="docs/assets/contour-graph.png" alt="Contour graph"><br>Contour graph: a T-junction traced into three colored edges</td>
<td width="33%"><img src="docs/assets/pyramid-levels.png" alt="Pyramid levels"><br>A 5-level image pyramid, coarse-to-fine</td>
</tr>
</table>

All rendered deterministically from synthetic fixtures by
[`gen_illustrations`](crates/vision-metrology/examples/gen_illustrations.rs) — see
[CONTRIBUTING.md](CONTRIBUTING.md#documentation-illustrations) to regenerate.

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
| `fit` | Robust line / circle / ellipse fitting, algebraic-init then geometric refine, every fit reports `rms` / `max_dev` / `n_used` |
| `laser` | Laser stripe extraction using opposite-polarity edge pairs, with ROI and prior tracking |
| `matching` | Shape-based object detection: gradient-orientation model, coarse-to-fine search over translation / rotation / scale, subpixel pose refinement — see the [guide](docs/shape-matching.md) |
| `measure` | Calipers (rect / arc / radial) and metrology models: measure a located part and fit the result — see the [guide](docs/measure.md) |
| `segment` | Otsu and adaptive thresholding, connected-component labeling, watershed, edgel region growing |
| `lsd` | LSD line-segment detection |
| `warp` | Image warping: build a `dst → src` `Map` once (affine / projective / polar / arbitrary `from_fn`), apply it per frame with a first-class validity mask |

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
  ├─────►  lsd        LSD line segments
  │
  ├─────►  segment    thresholding, CCL, watershed, per-component stats
  │
  ├─────►  fit        robust line / circle / ellipse fitting, residuals
  │
  └─────►  matching   shape model, coarse-to-fine search, pose refinement
                 │
                 ▼
            measure    calipers at the found pose, fit primitives, pass/fail
```

`laser` consumes `edge` directly — it scans rows or columns for opposite-polarity
1-D edge pairs rather than going through the 2-D pipeline. `fit` also runs
directly off `edge`/`contour` output; `measure` is the module that closes the
loop, applying calipers at a `matching` pose and fitting the result with `fit`.

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
let edgels = det.detect(&img.as_view(), &Edge2DConfig::default());
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
- [Measuring a located part](docs/measure.md) — calipers, rect vs. arc vs.
  radial placement, the metrology model, reading `RejectReason`.

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

## Lab

[`lab/`](lab/README.md) is a local interactive workbench (FastAPI + React) over the
`vision_metrology` package — upload an image, drag a box to teach a shape model, find it
elsewhere in the frame, measure circles and lines against the found pose, and see the
per-caliper hit/reject reasons and intensity profiles behind the fit. It is the
library's own teach → find → measure → judge chain, made visible; pixels only, no
persistence beyond disk. See `lab/README.md` for the API and what is deliberately out of
scope.

```bash
cd lab/backend && uv sync && uv run uvicorn vm_lab.app:app --reload   # :8000
cd lab/frontend && bun install && bun run dev                         # :5174
```

## Contributing

Build, test, lint, and benchmark instructions are in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
