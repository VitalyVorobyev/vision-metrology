---
name: metrology-invariants
description: Use this when implementing or reviewing anything subpixel (edges, laser, contours). Prevents silent coordinate and convention bugs.
---

# Metrology invariants (pixel-center world)

## Coordinate convention

* Integer index `i` corresponds to **pixel center** at `i as f32`.
* Subpixel values are expressed in the same coordinate system.

## Subpixel outputs

* Always specify what `x`/`y` means in docs (center coordinates).
* Provide tolerances in tests:

  * quick unit tests: ~0.1 px is fine
  * precision: the accuracy suite (`crates/vision-metrology/tests/accuracy.rs`) pins each
    detector's worst bias/sigma as an envelope — add a row for a new subpixel path

## Robustness

* Prefer edge-pair (Rising→Falling) for laser stripes over intensity peak fitting.
* If the algorithm uses an ROI around a predicted center, document:

  * how prediction is formed
  * what happens on gaps / reacquisition
