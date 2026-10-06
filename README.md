[![CI](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml)
[![Security Audit](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml)
[![Publish Rust Docs](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml)

# vision-metrology

High-precision image processing for industrial machine-vision metrology, in pure Rust with
Python bindings. Locate a part, measure it with subpixel calipers, fit primitives robustly,
and report the result in millimetres through a camera calibration.

No OpenCV, no FFI. Coordinates follow the **pixel-centre** convention: integer `i` means
coordinate `i as f32`.

<table>
<tr>
<td width="33%"><img src="docs/assets/shape-matching.png" alt="Shape matching"><br>Shape-based matching: a model contour found at two poses</td>
<td width="33%"><img src="docs/assets/caliper-anatomy.png" alt="Caliper anatomy"><br>A caliper and its cross-averaged 1-D profile</td>
<td width="33%"><img src="docs/assets/laser-stripe.png" alt="Laser stripe extraction"><br>Laser stripe extraction: subpixel centreline</td>
</tr>
<tr>
<td width="33%"><img src="docs/assets/circle-fit.png" alt="Robust circle fit"><br>Robust circle fit with outliers rejected</td>
<td width="33%"><img src="docs/assets/contour-graph.png" alt="Contour graph"><br>Contour graph: a T-junction traced into three edges</td>
<td width="33%"><img src="docs/assets/birdseye-mosaic.png" alt="Bird's-eye mosaic"><br>Two calibrated cameras composited onto their shared plane</td>
</tr>
</table>

## What it does

```
undistort / rectify → locate the part → use its pose as a fixture → calipers
                    → robust fit with residuals → millimetres → pass / fail
```

Alongside that chain: 1-D/2-D subpixel edges, junction-aware contour graphs, laser stripe
extraction, LSD line segments, thresholding and connected components, cross-correlation
and inter-frame displacement, image warping, and binary morphology.

| Crate | What it is |
|---|---|
| [`vision-metrology`](crates/vision-metrology) | The domain algorithms. It re-exports `vm-primitives`, so it is the only dependency you need. Module and feature tables are in its README. |
| [`vm-primitives`](crates/vm-primitives) | Building blocks: images and sampling, geometry, pyramids, edges, morphology. |
| [`vm-python`](crates/vm-python) | Python bindings, installed as `vision-metrology` and imported as `vision_metrology`. |

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

Runnable programs are in [`crates/vision-metrology/examples/`](crates/vision-metrology/examples):

```bash
cargo run -p vision-metrology --example measure_circles
cargo run -p vision-metrology --example shape_matching -- --help
```

### Python

Build and install the extension with [maturin](https://www.maturin.rs/) (Python 3.10+):

```bash
cd crates/vm-python && maturin develop --release
```

```python
import numpy as np
import vision_metrology as vm

img = np.zeros((64, 64), dtype=np.uint8)
img[:, 32:] = 200

edgels = vm.EdgeDetector(vm.EdgeConfig()).detect(img)
print(len(edgels), edgels[0])
```

More scripts are in [`examples/python/`](examples/python), and the API overview is in the
[vm-python README](crates/vm-python/README.md).

## Guides

- [Shape-based object detection](docs/shape-matching.md): building a model, polarity,
  contrast tuning, reading the score, scale invariance, saving models.
- [Measuring a located part](docs/measure.md): calipers, rect, arc, radial and strip
  placement, where an edge is located (gradient peak, midpoint or half-contrast), the
  metrology model, reading `RejectReason`, and tracing a failure with `explain`.
- [Tracking a bead](docs/bead.md): a prior curve refined by strip calipers, the pair gates,
  the regulariser, reading the quality statistics, and moving a prior from frame to frame.
- [Performance and accuracy](docs/performance.md): speed, accuracy envelopes on
  synthetic ground truth, and real-data results.
- API reference: `cargo doc --open`, or the
  [published rustdoc](https://vitalyvorobyev.github.io/vision-metrology/).

## Lab

[`lab/`](lab/README.md) is an interactive workbench over the library. You can open
captures, teach a shape model from picked contours, find it across a set, rectify, measure
with calipers in pixels or millimetres, and see every caliper's hit or rejection reason.
It runs in a browser (over the Python bindings) or as a desktop app (calling the Rust
library directly).

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.
