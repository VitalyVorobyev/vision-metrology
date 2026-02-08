# vision-metrology

Rust workspace for industrial machine-vision metrology.

## Crates
- `vm-core`: image views, sampling/interpolation, geometry primitives.
- `vm-pyr`: fast 2x2 mean downsample and reusable f32 image pyramid.
- `vision-metrology`: umbrella crate re-exporting workspace crates.

## Quick start
```bash
cargo test
```

## Benchmarks
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
