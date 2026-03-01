Line segment detection (LSD) and conic/ellipse fitting.

`vm-shape` provides two independent shape extraction capabilities. `LsdDetector` implements gradient-coherence line-segment detection with NFA (Number of False Alarms) validation, producing sub-pixel accurate `LineSegment2f` results directly from a grayscale image. `ConicFitter` implements Bookstein/Fitzgibbon algebraic conic fitting on a set of 2-D points and wraps it with RANSAC-based outlier rejection; the output is converted from algebraic conic coefficients to geometric `Ellipse2f` parameters (centre, semi-axes, orientation angle).

## Quick start

```rust
use vm_shape::{LsdDetector, LsdConfig, ConicFitter, ConicFitConfig};
use vm_core::Image;

// Line segment detection
let img: Image<u8> = Image::new(256, 256);
let lsd = LsdDetector::new(LsdConfig::default());
let segments = lsd.detect(&img);
println!("found {} line segments", segments.len());

// Ellipse fitting
use vm_core::Point2f;
let pts: Vec<Point2f> = vec![/* points on ellipse */];
let fitter = ConicFitter::new(ConicFitConfig::default());
if let Some(ellipse) = fitter.fit_ellipse(&pts) {
    println!("ellipse: centre=({:.2}, {:.2}), a={:.2}, b={:.2}",
        ellipse.cx, ellipse.cy, ellipse.a, ellipse.b);
}
```

## Key public types

| Type | Description |
|---|---|
| `LsdDetector` | Gradient-coherence LSD with NFA validation |
| `LsdConfig` | Scale, sigma, threshold, and NFA parameters |
| `LineSegment2f` | Sub-pixel line segment with endpoints and width |
| `ConicFitter` | Bookstein/Fitzgibbon algebraic fit + RANSAC outlier rejection |
| `ConicFitConfig` | RANSAC iterations, inlier tolerance, min-point count |
| `Ellipse2f` | Geometric ellipse: centre, semi-axes (a >= b), orientation angle |
