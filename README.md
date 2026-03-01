[![CI](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml)
[![Security Audit](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml)
[![Publish Rust Docs](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml)

# vision-metrology

Rust workspace for industrial machine-vision metrology.

## Crates

| Crate | Description |
|---|---|
| `vm-core` | Image views, sampling/interpolation, border modes, geometry primitives (nalgebra aliases) |
| `vm-pyr` | Ultra-fast 2x2 mean image pyramid |
| `vm-edge` | Subpixel 1-D/2-D edge detection (DoG), edgels, gradient buffers |
| `vm-laser` | Industrial laser stripe extraction using opposite-polarity edge pairs |
| `vm-contour` | Junction-aware contour graph extraction (T/Y junctions, loops, polyline smoothing) |
| `vm-morph` | Binary morphology, chamfer distance, Zhang-Suen thinning |
| `vm-multiscale` | Multi-scale 2-D edge detection across a Gaussian pyramid |
| `vm-shape` | LSD line-segment detection, conic/ellipse fitting with RANSAC |
| `vm-segment` | Otsu/adaptive thresholding, connected-component labeling, watershed |
| `vm-match` | Chamfer-based rigid/similarity edge-model matching with ICP refinement |
| `vm-python` | PyO3 Python bindings package (`vision-metrology` / `vision_metrology`) |
| `vision-metrology` | Umbrella re-export crate |

## Pipeline

```
Image
  |
  v
vm-pyr          (2x2 mean pyramid — coarse-to-fine levels)
  |
  v
vm-edge         (subpixel 1-D/2-D DoG edge detection, gradient buffers)
  |
  v
vm-multiscale   (merge detections across pyramid levels → level-0 coords)
  |
  v
vm-contour      (topology graph — T/Y junctions, loops, polyline smoothing)
  |
  +-----> vm-shape    (LSD line segments, conic/ellipse fitting + RANSAC)
  |
  +-----> vm-segment  (thresholding, CCL, watershed, per-component stats)
  |
  +-----> vm-match    (chamfer edge-model matching, ICP, IoU NMS)
```

## Quick start

```bash
# Run all tests
cargo test

# Run the measure_circles end-to-end example
cargo run -p vision-metrology --example measure_circles
```

## Python

Build and install the Python extension (requires [maturin](https://www.maturin.rs/)):

```bash
cd crates/vm-python
maturin develop   # editable install into the active Python environment
```

Then use from Python:

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

Python bindings are a hard-break rename:
- distribution: `vision-metrology`
- import: `vision_metrology`
- legacy `vm_python` / `Py*` names are removed.

Wheels are built with ABI3 (`abi3-py310`) and require Python `>=3.10`.

See `examples/python/` for runnable end-to-end scripts.

## Benchmarks

Run all workspace benchmarks:
```bash
cargo bench --workspace
```

Run all `vm-pyr` benchmarks:
```bash
cargo bench -p vm-pyr
```

Run only the downsample benchmark target:
```bash
cargo bench -p vm-pyr --bench downsample
```

Run only the specific downsample benchmark function:
```bash
cargo bench -p vm-pyr --bench downsample -- downsample2x2_mean_u8_to_f32_1280x1024
```

Run `vm-laser` benchmarks:
```bash
cargo bench -p vm-laser
```

Run `vm-edge` benchmarks:
```bash
cargo bench -p vm-edge --bench edge2d
```

Run `vm-contour` benchmarks:
```bash
cargo bench -p vm-contour --bench build_graph
```

### Benchmark snapshot

Measured via `cargo bench --workspace` on 2026-02-08 (local machine, Criterion defaults):

| Benchmark | Time (approx) |
|---|---:|
| `vm_pyr::downsample2x2_mean_u8_to_f32_1280x1024` | `38.20 us` |
| `vm_pyr::pyramid_build_u8_6_levels_1280x1024` | `122.6 us` |
| `vm_laser::rows_1280x512` | `2.10 ms` |
| `vm_laser::cols_gather_512x1280` | `146.1 us` |
| `vm_edge::edge2d_detect_u8_1280x1024` | `5.52 ms` |
| `vm_contour::vm_contour_build_graph_50k` | `3.70 ms` |

Numbers vary by CPU, toolchain, thermal state, and background load.
