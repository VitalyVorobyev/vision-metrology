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
- `PyConicFitter` -> `ConicFitter`
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
- `ConicFitConfig`
- `ShapeModelConfig`
- `ShapeSearchConfig`

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
