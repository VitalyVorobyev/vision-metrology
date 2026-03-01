PyO3 Python bindings for the vision-metrology workspace.

This package is published as `vision-metrology` and imported as `vision_metrology`.
It provides both:
- stateful object APIs (`EdgeDetector`, `RigidMatcher`, ...), and
- declarative free functions (`detect_edges_u8`, `match_rigid_model`, ...).

## Breaking change (hard break)

Legacy names were removed:
- `import vm_python` -> `import vision_metrology as vm`
- `PyEdgeDetector` -> `EdgeDetector`
- `PyMultiScaleDetector` -> `MultiScaleDetector`
- `PyLsdDetector` -> `LsdDetector`
- `PyConicFitter` -> `ConicFitter`
- `PyRigidMatcher` -> `RigidMatcher`
- `PySegmenter` -> `Segmenter`
- `PyEdgel` -> `Edgel`
- `PyLineSegment` -> `LineSegment`
- `PyEllipse` -> `Ellipse`
- `PyMatchResult` -> `MatchResult`
- `PyComponentStats` -> `ComponentStats`

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
edgels_obj = vm.EdgeDetector(vm.EdgeConfig()).detect_u8(img)

# Free-function API
edgels_fn = vm.detect_edges_u8(img, vm.EdgeConfig())
```

## Config classes

- `EdgeConfig`
- `MultiScaleConfig`
- `LsdConfig`
- `ConicFitConfig`
- `RigidMatchConfig`

## Free functions

- `detect_edges_u8(img, config)`
- `detect_multiscale_edges_u8(img, config)`
- `detect_line_segments_u8(img, config)`
- `fit_ellipse(pts, config)`
- `match_rigid_model(model_edgels, scene_img, edge_config, match_config)`
- `otsu_threshold(img)`
- `threshold_binary(img, threshold)`
- `label_components(img, connectivity=8)`
- `component_stats(label_img, n_labels, min_area=1)`

See `examples/python/` for runnable scripts.
