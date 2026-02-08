---

name: tests-synthetic-fixtures
description: Use this to create deterministic test images/signals with known subpixel ground truth (edges, stripes, circles). Helps lock correctness before optimizing.
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Synthetic fixtures (deterministic)

## Principles

* Deterministic by default (no RNG), or seeded RNG only.
* Tests should check:

  * geometry error (px)
  * stability under small noise/blur
  * edge cases (odd sizes, borders, ROI truncation)

## Useful fixtures

* **Half-plane step edge** with known line equation (slanted edge):

  * `I(p)=A` if dot(n, p) < t else `B`
  * Optional blur along `n` (binomial/gaussian-ish) to make it realistic
* **Stripe (laser)** as difference of two steps (bright-on-dark):

  * known left/right subpixel positions → known center + width
* **Circle / ring boundary** (later):

  * implicit distance function `d = sqrt((x-cx)^2+(y-cy)^2)-r`
  * intensity from sign(d) and controlled blur

## Expected-value checks

* Keep early tests loose (~0.1–0.2 px).
* Add tighter tests later (aim 0.05 px) on higher-fidelity fixtures.

## Regression strategy

* When a bug appears, capture it as a small synthetic fixture.
