PyO3 Python bindings for the vision-metrology workspace.

This package is published as `vision-metrology` and imported as `vision_metrology`.
It provides both:
- stateful object APIs (`EdgeDetector`, `ShapeMatcher`, ...), and
- declarative free functions (`detect_edges_u8`, `find_shape_model`, ...).

## Breaking change (hard break)

Legacy names were removed:
- `import vm_python` -> `import vision_metrology as vm`
- `PyEdgeDetector` -> `EdgeDetector`
- `PyLsdDetector` -> `LsdDetector`
- `PyConicFitter` -> `Fitter` (also gained `fit_circle`, matching the Rust `fit` module)
- `PySegmenter` -> `Segmenter`
- `PyEdgel` -> `Edgel`
- `PyLineSegment` -> `LineSegment`
- `PyEllipse` -> `Ellipse`
- `PyComponentStats` -> `ComponentStats`

The chamfer-based `RigidMatcher` / `MatchResult` / `RigidMatchConfig` /
`match_rigid_model` API has been **removed**, not deprecated. Shape-based
object detection replaces it:

```python
model = vm.ShapeModel(reference, (x, y, width, height))
for m in vm.ShapeMatcher(vm.ShapeSearchConfig(min_score=0.6)).find(scene, model):
    print(m.score, m.x, m.y, m.angle, m.scale)
```

## Python version and wheels

- Requires Python `>=3.10`.
- Built as ABI3 (`abi3-py310`) wheels.

## Quick start

```bash
cd crates/vm-python
maturin develop
```

```python
import numpy as np
import vision_metrology as vm

img = np.zeros((64, 64), dtype=np.uint8)
img[:, 32:] = 200

# Object API
edgels_obj = vm.EdgeDetector(vm.EdgeConfig()).detect(img)

# Free-function API
edgels_fn = vm.detect_edges_u8(img, vm.EdgeConfig())
```

## Config classes

- `EdgeConfig`
- `LsdConfig`
- `FitConfig`
- `MeasureConfig`
- `ShapeModelConfig` (nests `EdgeConfig` as `.edge`)
- `ShapeSearchConfig` (nests `ShapeSearchTuning` as `.tuning`; `.roi`,
  `.angle_range`, `.scale_range` narrow the search)
- `Contrast` — `Contrast.raw(v)` / `Contrast.fraction_of_range(f)`, the two
  variants of `min_contrast` on `ShapeModelConfig` and `ShapeSearchConfig`

### Config design notes

- **Sentinels are `None`.** Every "auto" / "unlimited" Rust `Option<T>` is a
  Python `None` — `num_levels=None` picks the pyramid depth automatically,
  `max_matches=None` reports every instance, and so on.
- **`Hysteresis` is a pair of optionals, not its own type.** `EdgeConfig`'s
  `low_thresh` / `high_thresh` are both `None` for automatic thresholding, or
  both set for manual — the familiar `argparse`-style optional-pair idiom,
  rather than a second small enum type for one either/or choice.
- **`Contrast` *is* its own type.** A bare `float` for `min_contrast` would
  have to silently pick a unit, and picking the wrong one changes behaviour
  by up to 257x between `uint8` and `uint16` images — exactly the ambiguity
  the Rust `Contrast` type exists to remove. Construct with the two static
  methods.
- **Search effort is nested.** `ShapeSearchConfig`'s top-level fields say
  *what* to look for (`min_score`, `roi`, `angle_range`, ...); the six fields
  on `ShapeSearchConfig.tuning` say *how hard* the search works
  (`greediness`, `max_candidates`, ...) and are rarely touched.

## Free functions

- `detect_edges_u8(img, config)`
- `detect_line_segments_u8(img, config)`
- `fit_ellipse(pts, config)`
- `find_shape_model(model_image, roi, scene_image, model_config=None, search_config=None)`
- `otsu_threshold(img)`
- `threshold_binary(img, threshold)`
- `label_components(img, connectivity=8)`
- `component_stats(label_img, n_labels, min_area=1)`

See `examples/python/` for runnable scripts.
