Industrial laser stripe extraction using opposite-polarity edge pairs.

`vm-laser` scans each row (or column) of a region-of-interest and locates the center of a bright laser stripe by finding the inner pair of dark-to-bright / bright-to-dark edges. A prior position from the previous scan line guides the search window, providing continuity across occluded or low-contrast segments. One `LaserSample` is emitted per scan line — including an explicitly invalid sample when no stripe is found — so downstream consumers always receive a fixed-length result. Column scanning uses reusable gather buffers to avoid cache thrashing on transposed access patterns.

## Quick start

```rust
use vm_laser::{LaserLineDetector, LaserLineConfig};
use vm_core::Image;

let mut img: Image<u8> = Image::new(640, 480);
// ... fill with laser image data ...

let config = LaserLineConfig::default();
let mut det = LaserLineDetector::new(config);
let line = det.extract_rows(&img, None);

for sample in &line.samples {
    if sample.is_valid() {
        println!("row {}: x={:.2}", sample.row, sample.x);
    }
}
```

## Key public types

| Type | Description |
|---|---|
| `LaserLineDetector` | Stateful detector; holds scratch buffers and prior state |
| `LaserLineConfig` | Threshold, half-width, ROI, and search-window parameters |
| `LaserSample` | Single scan-line result (position, validity flag, strength) |
| `LaserLine` | Collection of all `LaserSample`s for one extraction pass |
