---

name: laser-extract
description: Use this when touching vm-laser. Encodes the intended coarse→ROI→DoG→pairing→tracking behavior and defaults.
-------------------------------------------------------------------------------------------------------------------------

# Laser extraction intent (industrial)

## Stripe assumptions

* Bright-on-dark.
* Typical width: 3–6 px (default accept window 2–10 px).

## Pipeline (expected)

1. Coarse estimate per scan (max or COM).
2. Precise pass in an ROI around prediction:

   * 1D DoG response → extrema
   * choose Rising→Falling pair
   * apply width constraints + prior to predicted center
3. Continuity along scan axis:

   * allow short gaps
   * reacquire after max gap exceeded

## Performance expectations

* Rows scan is the fast path (contiguous slices).
* Cols scan:

  * default: gather ROI segment into scratch buffer
  * optional: accept caller-provided transposed view

## “Don’t do”

* Quadratic peak fit on intensity (stripe can be wide/flat/saturated).
