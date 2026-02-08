# vision-metrology

Rust workspace for industrial machine-vision metrology.

[![CI](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/ci.yml)
[![Security Audit](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml/badge.svg)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/audit.yml)
[![Publish Rust Docs](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml/badge.svg?branch=main)](https://github.com/VitalyVorobyev/vision-metrology/actions/workflows/publish-docs.yml)

## Current status
- `vm-core` implemented and tested (image views, sampling, geometry).
- `vm-pyr` implemented and benchmarked (fast 2x2 mean downsample, reusable f32 pyramid).
- `vm-edge` implemented and tested (1D DoG edges, subpixel refinement, edge-pair selection for laser rows).
- `vm-laser` implemented and tested (row/column laser line extraction with ROI + prior tracking).
- CI, audit, and docs workflows are configured in GitHub Actions.

## Crates
- `vm-core`: image views, sampling/interpolation, geometry primitives.
- `vm-edge`: robust 1D edge extraction + opposite-polarity edge-pair primitive.
- `vm-laser`: industrial laser line extraction using edge-pair tracking (rows/cols/transposed).
- `vm-pyr`: fast 2x2 mean downsample and reusable f32 image pyramid.
- `vision-metrology`: umbrella crate re-exporting workspace crates.

## Quick start
```bash
cargo test
```

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

### Benchmark snapshot
Measured via `cargo bench --workspace` on 2026-02-08 (local machine, Criterion defaults):

| Benchmark | Time (approx) |
|---|---:|
| `vm_pyr::downsample2x2_mean_u8_to_f32_1280x1024` | `38.20 us` |
| `vm_pyr::pyramid_build_u8_6_levels_1280x1024` | `122.6 us` |
| `vm_laser::rows_1280x512` | `2.10 ms` |
| `vm_laser::cols_gather_512x1280` | `146.1 us` |

Numbers vary by CPU, toolchain, thermal state, and background load.
