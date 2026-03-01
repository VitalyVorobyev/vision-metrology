Chamfer-distance rigid and similarity edge-model matching with ICP refinement.

`vm-match` implements template-based edge matching for industrial inspection. `EdgeModel` stores a set of template edgels in centred coordinates (zero mean) along with their surface normals. The coarse search performs an exhaustive grid over translation x rotation (and optionally scale for similarity matching) using a precomputed chamfer distance map of the scene. The top-K candidate poses are refined with Iterative Closest Point (ICP) to sub-pixel accuracy. A normal-coherence score rejects poses where model normals are anti-aligned with scene gradients, preventing flipped detections. Greedy IoU-NMS removes duplicate detections in multi-instance search.

## Quick start

```rust
use vm_match::{EdgeModel, RigidEdgeMatcher, RigidMatchConfig};
use vm_edge::Edgel;
use vm_core::Image;

let template_edgels: Vec<Edgel> = vec![/* ... */];
let model = EdgeModel::from_edgels(&template_edgels);

let scene: Image<u8> = Image::new(512, 512);
// ... fill with scene image ...

let config = RigidMatchConfig::default();
let matcher = RigidEdgeMatcher::new(config);
let results = matcher.match_model(&model, &scene);
for r in &results {
    println!("match: tx={:.1}, ty={:.1}, angle={:.2}rad, score={:.3}",
        r.tx, r.ty, r.angle, r.score);
}
```

## Key public types

| Type | Description |
|---|---|
| `EdgeModel` | Centred template edgels with normals |
| `RigidEdgeMatcher` | Grid search + ICP + NMS for rigid (translation + rotation) matching |
| `RigidMatchConfig` | Search ranges, step sizes, ICP iterations, NMS IoU threshold |
| `MatchResult` | Refined pose (tx, ty, angle) with chamfer score |
