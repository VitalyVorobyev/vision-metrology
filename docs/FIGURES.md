# Figures Pipeline

This gallery uses an external-fixtures pipeline:

1. Python creates deterministic fixtures (`docs/fig/fixtures/...`).
2. Rust (`vm_gallery`) runs algorithms on those fixtures and writes raw outputs (`docs/fig/raw/...`).
3. Python plots final gallery images (`docs/fig/*.png`) from Rust raw outputs.

Core Rust crates do not depend on Python.

## Regenerate Everything

```bash
python tools/gallery/run_all.py
```

## Layout

- `docs/fig/fixtures/<case>/input.png`
- `docs/fig/fixtures/<case>/truth.json`
- `docs/fig/raw/<case>/...` (Rust outputs)
- `docs/fig/<final_figure>.png` (final plots)

## Case Map

- `morphology_open_close.png`
  - Fixture: `fixtures/morphology/`
  - Rust: `vm_gallery morphology`
  - Raw: `raw/morphology/input.png`, `open.png`, `close.png`, `meta.json`

- `pyramid_levels.png`
  - Fixture: `fixtures/pyramid/`
  - Rust: `vm_gallery pyramid`
  - Raw: `raw/pyramid/level_*.png`, `meta.json`

- `edge1d_dog_stripe.png`
  - Fixture: `fixtures/edge1d/` (1-row signal PNG)
  - Rust: `vm_gallery edge1d`
  - Raw: `raw/edge1d/signal.csv`, `response.csv`, `result.json`, `meta.json`

- `laser_rows.png`
  - Fixture: `fixtures/laser_rows/`
  - Rust: `vm_gallery laser_rows`
  - Raw: `raw/laser_rows/line.json`, `overlay.png`, `meta.json`

- `laser_cols.png`
  - Fixture: `fixtures/laser_cols/`
  - Rust: `vm_gallery laser_cols`
  - Raw: `raw/laser_cols/line.json`, `overlay.png`, `meta.json`

- `edgels_overlay.png`
  - Fixture: `fixtures/edgels2d/`
  - Rust: `vm_gallery edgels2d`
  - Raw: `raw/edgels2d/edgels.json`, `meta.json`

- `contour_graph.png`
  - Fixture: `fixtures/contour_graph/`
  - Rust: `vm_gallery contour_graph`
  - Raw: `raw/contour_graph/graph.json`, `meta.json`
