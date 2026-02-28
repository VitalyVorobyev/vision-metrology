# Benchmark Baselines

Performance baselines for the `vision-metrology` workspace.
All measurements are single-core (no Rayon) on a developer laptop unless noted.
Target platform: x86-64, macOS/Linux, `cargo bench --release`.

## How to regenerate

```bash
cargo bench --workspace --exclude vm-python 2>&1 | tee bench_baselines.txt
```

To benchmark a single crate:

```bash
cargo bench -p vm-pyr
cargo bench -p vm-edge
cargo bench -p vm-laser
cargo bench -p vm-contour
cargo bench -p vm-segment
cargo bench -p vm-shape
cargo bench -p vm-match
cargo bench -p vm-multiscale
```

---

## vm-pyr  (`crates/vm-pyr/benches/downsample.rs`)

| Benchmark | Description | Baseline |
|-----------|-------------|----------|
| `downsample2x2_mean_u8_to_f32_1280x1024` | Single 2×2 mean downsample, u8→f32, 1280×1024 | TBD |
| `pyramid_build_u8_6_levels_1280x1024` | Full 6-level f32 pyramid from u8 source, 1280×1024 | TBD |

---

## vm-edge  (`crates/vm-edge/benches/edge2d.rs`)

| Benchmark | Description | Baseline |
|-----------|-------------|----------|
| `edge2d_detect_u8_1280x1024` | 2-D Scharr + NMS + hysteresis + parabolic subpixel, 1280×1024 | TBD |

---

## vm-laser  (`crates/vm-laser/benches/extract.rs`)

| Benchmark | Description | Baseline |
|-----------|-------------|----------|
| `vm_laser_rows_1280x512` | Full row-scan laser extraction, 1280×512 | TBD |
| `vm_laser_cols_gather_512x1280` | Column-scan with gather buffer, 512×1280 | TBD |

---

## vm-contour  (`crates/vm-contour/benches/build_graph.rs`)

| Benchmark | Description | Baseline |
|-----------|-------------|----------|
| `vm_contour_build_graph_50k` | Contour graph construction from 50 k synthetic edgels | TBD |

---

## vm-segment  (`crates/vm-segment/benches/segment.rs`)

| Benchmark | Description | Baseline |
|-----------|-------------|----------|
| `otsu_1280x1024` | Otsu global threshold, 1280×1024 | TBD |
| `adaptive_threshold_1280x1024` | Adaptive (integral-image) threshold, 1280×1024 | TBD |
| `ccl_1280x1024` | Connected-component labeling (C8), 1280×1024 | TBD |
| `watershed_1280x1024_4seeds` | Watershed segmentation with 4 seeds, 1280×1024 | TBD |

---

## vm-shape  (`crates/vm-shape/benches/detect.rs`)

| Benchmark | Description | Baseline |
|-----------|-------------|----------|
| `lsd_detect_u8_1280x1024` | LSD line detection, 1280×1024 | TBD |
| `lsd_detect_u8_512x512` | LSD line detection, 512×512 | TBD |
| `conic_ransac_1000pts` | RANSAC ellipse fit, 1000 points, 500 iterations | TBD |
| `conic_direct_bookstein_100pts` | Direct Bookstein conic fit, 100 points | TBD |

---

## vm-match  (`crates/vm-match/benches/match.rs`)

| Benchmark | Description | Baseline |
|-----------|-------------|----------|
| `rigid_match_1280x1024_20edgel_model` | Rigid chamfer grid search + ICP, 20-edgel model, 1280×1024 scene | TBD |

---

## vm-multiscale  (`crates/vm-multiscale/benches/detect.rs`)

| Benchmark | Description | Baseline |
|-----------|-------------|----------|
| `multiscale_detect_u8_1280x1024_3levels` | 3-level multi-scale edge detection, 1280×1024 | TBD |
| `multiscale_detect_u8_1280x1024_1level` | Single-level edge detection via multiscale API, 1280×1024 | TBD |

---

## Performance target

Full pipeline (pyramid + multi-scale edges + contour graph): **< 100 ms** per 1280×1024 frame on a single x86-64 core.
Laser row scan alone: **< 2 ms** on 1280×512.

---

## Notes

- All `TBD` values should be filled in after running `cargo bench --workspace --exclude vm-python`.
- Criterion reports median time with confidence interval; record the median.
- Re-measure after any change to a hot-path function (convolution, downsample, NMS, CCL).
- SIMD auto-vectorisation is assumed from `rustc`; no explicit SIMD intrinsics are currently used.
