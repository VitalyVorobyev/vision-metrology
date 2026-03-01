Umbrella crate re-exporting the entire vision-metrology workspace.

`vision-metrology` is a single convenience entry point: add it as your only dependency and all workspace crates become available under their respective module paths. It is the right starting point for integration examples and end-to-end pipelines where you want access to the full stack without listing each crate individually.

## Quick start

```bash
# Run the built-in measure_circles end-to-end example
cargo run -p vision-metrology --example measure_circles
```

Or from your own crate's `Cargo.toml`:

```toml
[dependencies]
vision-metrology = { path = "../vision-metrology" }
```

## Key crates re-exported

| Module | Source crate | Provides |
|---|---|---|
| `vm_core` | `vm-core` | Image, geometry, border modes, Error |
| `vm_pyr` | `vm-pyr` | PyramidF32, 2x2 mean downsample |
| `vm_edge` | `vm-edge` | Edge1DDetector, Edge2DDetector, Edgel, GradientBuffers |
| `vm_laser` | `vm-laser` | LaserLineDetector, LaserSample, LaserLine |
| `vm_contour` | `vm-contour` | ContourGraph, build_graph_from_edgels, smooth_polyline |
| `vm_morph` | `vm-morph` | erode/dilate, chamfer distance, thinning |
| `vm_multiscale` | `vm-multiscale` | MultiScaleEdgeDetector, ScaleAnnotatedEdgel |
| `vm_shape` | `vm-shape` | LsdDetector, ConicFitter, LineSegment2f, Ellipse2f |
| `vm_segment` | `vm-segment` | otsu_threshold_u8, label_connected_components_u8, ComponentStats |
| `vm_match` | `vm-match` | EdgeModel, RigidEdgeMatcher, MatchResult |
