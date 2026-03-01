Image segmentation: thresholding, connected-component labeling, and watershed.

`vm-segment` provides a complete binary segmentation pipeline. `otsu_threshold_u8` computes the globally optimal Otsu threshold from the image histogram in a single pass. `adaptive_threshold_u8` computes a local-mean threshold in a sliding window for images with uneven illumination. Two-pass connected-component labeling (C4 or C8) assigns integer labels to foreground blobs and returns per-component statistics (area, bounding box, centroid). Marker-based watershed segments touching or overlapping objects that simple thresholding cannot separate.

## Quick start

```rust
use vm_segment::{otsu_threshold_u8, label_connected_components_u8, Connectivity};
use vm_core::Image;

let img: Image<u8> = Image::new(64, 64);
// ... fill with your image ...

let thresh = otsu_threshold_u8(&img);
println!("Otsu threshold: {thresh}");

let binary: Image<u8> = img.threshold(thresh);
let (label_img, n_labels) = label_connected_components_u8(&binary, Connectivity::C8);
println!("found {n_labels} components");
```

## Key public types

| Type / Function | Description |
|---|---|
| `otsu_threshold_u8` | Global Otsu threshold from histogram |
| `adaptive_threshold_u8` | Local-mean adaptive threshold |
| `label_connected_components_u8` | Two-pass CCL returning label image and count |
| `CcLabel` | Integer label type (0 = background) |
| `ComponentStats` | Area, bounding box, and centroid for one component |
