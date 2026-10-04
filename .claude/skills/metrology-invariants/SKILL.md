---
name: metrology-invariants
description: Use this when implementing or reviewing anything subpixel (edges, laser, contours). Prevents silent coordinate and convention bugs.
---

# Metrology invariants (pixel-center world)

## Coordinate convention

* Invariants 1, 2, 11 and 20 in `docs/dev/system-design.md` bind here: pixel centres, the
  pyramid coordinate mapping, the default border mode, and f32 storage with f64
  accumulation. Read them there rather than from a copy.

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
