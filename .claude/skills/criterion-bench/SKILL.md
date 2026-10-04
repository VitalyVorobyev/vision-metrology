---
name: criterion-bench
description: Use this when adding or modifying hot paths. Adds a small benchmark and keeps results comparable over time.
---

# Criterion bench hygiene (minimal)

## Add a bench when

* You add an unsafe fast path
* You change convolution/downsample/scan loops
* You introduce new thresholds or ROI behavior that changes work per scan

## Bench rules

* Use representative sizes (e.g., 1280×1024, 720p, “laser typical”).
* Avoid random input unless seeded; deterministic data makes diffs meaningful.
* Name benches by operation and size: `downsample2x2_to_f32_u8_1280x1024`,
  `laser_extract_rows_1280x512`.

## Report

* If you changed perf behavior, put before/after numbers in the PR description, with one
  sentence on what got faster or slower and why (the rule is in `CONTRIBUTING.md`,
  Benchmarks).
