Junction-aware contour graph extraction from 2-D edgels.

`vm-contour` converts a flat list of `Edgel`s into a topological graph of polylines. Integer-grid adjacency rules determine connectivity (C4 or C8), while each graph edge retains full subpixel coordinates, tangent angles, and curvature values. The builder handles T-junctions, Y-junctions, and closed loops without special-casing. A separate `smooth_polyline` utility applies iterative Gaussian-weighted averaging to reduce quantization noise while preserving junction positions.

## Quick start

```rust
use vm_contour::{build_graph_from_edgels, ContourBuildConfig, smooth_polyline, Connectivity};
use vm_edge::Edgel;

let edgels: Vec<Edgel> = vec![/* ... */];
let config = ContourBuildConfig {
    connectivity: Connectivity::C8,
    ..Default::default()
};
let graph = build_graph_from_edgels(&edgels, &config);
println!("nodes: {}, edges: {}", graph.node_count(), graph.edge_count());

// Smooth the first polyline
if let Some(edge) = graph.edges().next() {
    let smoothed = smooth_polyline(edge.points(), 1.0, 3);
    println!("smoothed {} points", smoothed.len());
}
```

## Key public types

| Type | Description |
|---|---|
| `ContourGraph` | Directed graph of polyline edges and junction nodes |
| `GraphEdge` | Polyline segment with subpixel points, tangents, curvatures |
| `Node` | Junction or endpoint node in the graph |
| `NodeId` | Lightweight node handle |
| `Connectivity` | `C4` or `C8` adjacency rule |
| `ContourBuildConfig` | Builder parameters (connectivity, min-length, etc.) |
