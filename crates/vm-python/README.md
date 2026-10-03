# vision-metrology for Python

Python bindings for [vision-metrology](https://github.com/VitalyVorobyev/vision-metrology):
industrial machine-vision metrology with subpixel edges, shape-based matching, calipers,
robust fitting, the pixel → millimetre calibration bridge, image warping and
cross-correlation. numpy in, numpy out.

- Install name: `vision-metrology`; import name: `vision_metrology`.
- Python 3.10 or newer; ABI3 wheels (`abi3-py310`).
- Ships `__init__.pyi` and `py.typed`, so type checkers and IDEs see the real signatures.

## Install from source

```bash
cd crates/vm-python
maturin develop --release
```

## Quick start

```python
import numpy as np
import vision_metrology as vm

img = np.zeros((64, 64), dtype=np.uint8)
img[:, 32:] = 200

edgels = vm.EdgeDetector(vm.EdgeConfig()).detect(img)   # object API
edgels = vm.detect_edges(img, vm.EdgeConfig())          # free-function API
```

Every entry point that is generic over the pixel type in Rust accepts `uint8`, `uint16`
or `float32` arrays and dispatches on the array's dtype. Any other dtype raises
`ValueError`.

Locate a part and measure it at the found pose:

```python
model = vm.ShapeModel(reference, (x, y, width, height))
matches = vm.ShapeMatcher(vm.ShapeSearchConfig(min_score=0.6)).find(scene, model)

# Nominal geometry in the reference image's coordinates.
metrology = vm.MetrologyModel()
metrology.add(vm.MetrologyObject(vm.MetrologyShape.circle((cx, cy), 40.0)))
for m in matches:
    results = metrology.apply(scene, x=m.x, y=m.y, angle=m.angle, scale=m.scale,
                              origin=model.origin())
    for r in results:  # one result per object, in order
        print(r.kind, r.circle.r, r.rms, len(r.hits))
```

A single `Caliper` that finds nothing raises `vm.MeasureRejected`, and `e.args[0]` names
the reason (`"no_edge"`, `"wrong_polarity"`, `"too_oblique"`, `"off_image"`,
`"profile_too_short"`). `MetrologyModel.apply` reports a failed object as a
`MetrologyError` in its slot instead of raising, so one bad object does not hide the
others.

## Configs

Each Rust config is a Python class with keyword arguments, for example
`vm.ShapeSearchConfig(min_score=0.6)`.
- **Automatic and unlimited values are `None`.** For example, `num_levels=None` picks the
  pyramid depth.
- **Search effort is nested.** `ShapeSearchConfig.tuning` (`ShapeSearchTuning`) holds
  `greediness`, `max_candidates` and the other rarely touched fields.
- **Contrast thresholds carry their unit.** Use `vm.Contrast.raw(v)` (Scharr response on
  the input pixel scale) or `vm.Contrast.fraction_of_range(f)` (transfers between `uint8`
  and `uint16`).
- **Hysteresis is a pair of optionals.** `EdgeConfig.low_thresh` / `high_thresh` are both
  `None` (automatic) or both set.

## Coverage

| Rust module | Python |
|---|---|
| `edge` (2-D) | `EdgeDetector`, `detect_edges`, `Edgel` |
| `lsd` | `LsdDetector`, `detect_line_segments`, `LineSegment` |
| `fit` | `Fitter` (`fit_line`, `fit_circle`, `fit_ellipse`), `fit_line`, `fit_ellipse`, `FitConfig` |
| `matching` | `ShapeModel` (incl. `save`/`load`), `ShapeMatcher`, `ShapeMatch`, `find_shape_model`, `ShapeModelConfig`, `ShapeSearchConfig`, `Contrast`, `CropSpec` (`ShapeMatch.model_frame_map`) |
| `measure` | `Caliper` (`rect`/`arc`/`radial`, `measure`, `measure_pairs`, `profile`), `MeasureConfig`, `MetrologyModel` (`apply`, `layout`), `MetrologyObject`, `MetrologyShape`, `MetrologyResult`, `MetrologyError`, `CaliperPlacement`, `MeasureRejected` |
| `warp` | `Map` (`affine`, `projective`, `polar`, `log_polar`, `apply`, `apply_with_mask`) |
| `metric` | `CameraModel`, `PinholeIntrinsics`, `BrownConrady5`, `Plane3`, `PlaneGrid`, `pixel_to_plane`, `project_plane_points`, `plane_grid_map`, `undistort_map`, `load_rig_extrinsics`, `load_table_calibration` |
| `corr` | `CorrTemplate`, `find`, `find_topk`, `displacement`, `CorrConfig`, `CorrTemplateConfig`, `DisplacementConfig`, `Refine` |
| `scale` | `estimate_scale_moments`, `estimate_scale_logpolar`, `find_scale_invariant_roi`, `find_scale_invariant_center`, `ScaleEstimate` |
| `segment` | `Segmenter`, `otsu_threshold`, `threshold_binary`, `label_components`, `component_stats` |
| `contour` | `build_contour_graph`, `ContourGraph`, `smooth_polyline` |
| `morph` | `erode`, `dilate`, `open`, `close`, `thin`, `chamfer_distance` |

**Not bound:**
- `laser`;
- `segment::watershed` and edgel region growing;
- `contour::build_graph_from_edgels` (the raw-edgel constructor);
- the standalone 1-D edge detector (reachable through `Caliper`);
- pyramids and direction fields.

Runnable scripts are in
[`examples/python/`](https://github.com/VitalyVorobyev/vision-metrology/tree/main/examples/python).

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
