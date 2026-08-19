"""
Smoke tests for the vision_metrology PyO3 extension module.

These tests require the module to be built with maturin:
    maturin develop --manifest-path crates/vm-python/Cargo.toml

Run with:
    python -m pytest crates/vm-python/tests/test_bindings.py -v
"""

import numpy as np
import pytest
import vision_metrology as vm


def make_step_image(w: int = 64, h: int = 64, edge_x: int = 32) -> np.ndarray:
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, edge_x:] = 200
    return img


def test_hard_break_namespace():
    assert hasattr(vm, "EdgeDetector")
    assert hasattr(vm, "detect_edges_u8")
    assert not hasattr(vm, "PyEdgeDetector")
    assert not hasattr(vm, "PyEdgel")
    assert hasattr(vm, "ShapeModel")
    assert hasattr(vm, "ShapeMatcher")
    assert hasattr(vm, "find_shape_model")
    # The chamfer matcher is gone, not deprecated.
    assert not hasattr(vm, "RigidMatcher")
    assert not hasattr(vm, "match_rigid_model")


def test_edge_detector_object_and_function_parity():
    img = make_step_image()
    cfg = vm.EdgeConfig()

    det = vm.EdgeDetector(cfg)
    edgels_obj = det.detect_u8(img)
    edgels_fn = vm.detect_edges_u8(img, cfg)

    assert len(edgels_obj) > 0
    assert len(edgels_obj) == len(edgels_fn)

    e = edgels_obj[0]
    for attr in ("x", "y", "nx", "ny", "strength"):
        assert hasattr(e, attr)


def test_edge_detector_no_edges_on_blank():
    img = np.full((64, 64), 128, dtype=np.uint8)
    edgels = vm.detect_edges_u8(img, vm.EdgeConfig())
    assert len(edgels) == 0


def test_line_detector_object_and_function():
    h, w = 64, 64
    img = np.zeros((h, w), dtype=np.uint8)
    img[32:, :] = 255

    cfg = vm.LsdConfig()
    det = vm.LsdDetector(cfg)
    segs_obj = det.detect_u8(img)
    segs_fn = vm.detect_line_segments_u8(img, cfg)

    assert isinstance(segs_obj, list)
    assert isinstance(segs_fn, list)
    assert len(segs_obj) == len(segs_fn)

    if segs_fn:
        s = segs_fn[0]
        for attr in ("x1", "y1", "x2", "y2", "width", "nfa", "angle", "length"):
            assert hasattr(s, attr)


def test_conic_fitter_object_and_function():
    n = 40
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([32.0 + 10.0 * np.cos(t), 32.0 + 5.0 * np.sin(t)]).astype(
        np.float32
    )

    cfg = vm.FitConfig(ransac_iters=200, inlier_tol=1.5)
    obj_fit = vm.Fitter(cfg).fit_ellipse(pts)
    fn_fit = vm.fit_ellipse(pts, cfg)

    assert obj_fit is not None
    assert fn_fit is not None
    assert abs(obj_fit.cx - fn_fit.cx) < 1e-3
    assert abs(obj_fit.cy - fn_fit.cy) < 1e-3
    # Every fit reports what qualifies it.
    assert obj_fit.rms < 0.05
    assert obj_fit.max_dev < 0.1
    assert obj_fit.n_used == n


def test_conic_fitter_too_few_points_returns_none():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    cfg = vm.FitConfig(ransac_iters=10, inlier_tol=1.0)
    assert vm.fit_ellipse(pts, cfg) is None


def test_circle_fit_reports_residuals():
    """The circle fit the library had no implementation of until now."""
    n = 60
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([70.0 + 25.0 * np.cos(t), 55.0 + 25.0 * np.sin(t)]).astype(
        np.float32
    )

    fit = vm.Fitter().fit_circle(pts)
    assert fit is not None
    assert abs(fit.cx - 70.0) < 1e-2
    assert abs(fit.cy - 55.0) < 1e-2
    assert abs(fit.r - 25.0) < 1e-2
    assert fit.rms < 1e-2
    assert fit.n_used == n


def test_robust_loss_rejects_an_outlier():
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    pts = np.column_stack([50.0 + 20.0 * np.cos(t), 50.0 + 20.0 * np.sin(t)])
    pts = np.vstack([pts, [[120.0, 120.0]]]).astype(np.float32)

    plain = vm.Fitter(vm.FitConfig()).fit_circle(pts)
    robust = vm.Fitter(
        vm.FitConfig(loss="tukey", loss_scale=2.0, ransac_iters=300, inlier_tol=1.0)
    ).fit_circle(pts)
    assert plain is not None and robust is not None
    assert robust.n_used == 40, "the outlier should be dropped"
    assert robust.rms < plain.rms


def test_fit_config_validates_the_loss_name():
    import pytest

    with pytest.raises(ValueError):
        vm.Fitter(vm.FitConfig(loss="nonsense")).fit_circle(
            np.zeros((10, 2), dtype=np.float32)
        )


def test_segmentation_free_functions():
    img = np.full((64, 64), 30, dtype=np.uint8)
    img[10:21, 10:21] = 200
    img[40:56, 40:56] = 200

    threshold = vm.otsu_threshold(img)
    binary = vm.threshold_binary(img, threshold)
    labels, n_labels = vm.label_components(binary, connectivity=8)
    stats = vm.component_stats(labels, n_labels, min_area=10)

    assert n_labels >= 2
    assert len(stats) >= 2
    for s in stats:
        for attr in ("label", "pixel_count", "cx", "cy", "bbox_x", "bbox_y", "bbox_w", "bbox_h"):
            assert hasattr(s, attr)


def make_bracket(w: int = 200, h: int = 160, cx: float = 100.0, cy: float = 80.0,
                 angle: float = 0.0) -> np.ndarray:
    """Anti-aliased L-bracket, rendered from its signed distance function."""
    import math

    ys, xs = np.mgrid[0:h, 0:w]
    dx, dy = xs - cx, ys - cy
    cs, sn = math.cos(angle), math.sin(angle)
    mx, my = cs * dx + sn * dy, -sn * dx + cs * dy

    def sdf_box(px, py, hx, hy):
        ax, ay = np.abs(px) - hx, np.abs(py) - hy
        outside = np.sqrt(np.maximum(ax, 0) ** 2 + np.maximum(ay, 0) ** 2)
        return outside + np.minimum(np.maximum(ax, ay), 0)

    sdf = np.minimum(sdf_box(mx, my + 22, 30, 8), sdf_box(mx + 22, my, 8, 30))
    t = np.clip((1.0 - sdf) / 2.0, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return (40 + 170 * t).astype(np.uint8)


BRACKET_ROI = (58.0, 38.0, 84.0, 84.0)


def test_shape_model_reports_its_structure():
    model = vm.ShapeModel(make_bracket(), BRACKET_ROI)
    assert model.num_levels >= 2
    assert len(model.point_counts) == model.num_levels
    assert model.point_counts[0] > 20
    ox, oy = model.origin
    # The reference point is the level-0 centroid, so it sits inside the ROI.
    assert BRACKET_ROI[0] <= ox <= BRACKET_ROI[0] + BRACKET_ROI[2]
    assert BRACKET_ROI[1] <= oy <= BRACKET_ROI[1] + BRACKET_ROI[3]
    pts = model.reference_points()
    assert len(pts) == model.point_counts[0]
    assert all(len(p) == 2 for p in pts)


def test_shape_matcher_recovers_a_rotated_instance():
    import math

    reference = make_bracket()
    model = vm.ShapeModel(reference, BRACKET_ROI, vm.ShapeModelConfig(max_points=400))

    truth = math.radians(35.0)
    scene = make_bracket(cx=115.0, cy=70.0, angle=truth)
    matcher = vm.ShapeMatcher(
        vm.ShapeSearchConfig(min_score=0.6, refinement="least_squares")
    )
    matches = matcher.find(scene, model)

    assert len(matches) == 1
    m = matches[0]
    assert m.score > 0.9
    assert abs(math.degrees(m.angle) - 35.0) < 1.5
    assert abs(m.scale - 1.0) < 0.02
    assert m.support > 0
    assert isinstance(matcher.truncated, bool)

    # `matrix` bakes in the model origin, so a reference point maps straight in.
    mat = np.array(m.matrix(model.origin))
    mapped = mat @ np.array([100.0, 80.0, 1.0])
    assert abs(mapped[0] - 115.0) < 2.0
    assert abs(mapped[1] - 70.0) < 2.0


def test_find_shape_model_free_function_matches_the_object_api():
    import math

    reference = make_bracket()
    scene = make_bracket(cx=115.0, cy=70.0, angle=math.radians(35.0))
    search = vm.ShapeSearchConfig(min_score=0.6)

    out = vm.find_shape_model(reference, BRACKET_ROI, scene, None, search)
    assert len(out) == 1

    model = vm.ShapeModel(reference, BRACKET_ROI)
    obj = vm.ShapeMatcher(search).find(scene, model)
    assert len(obj) == 1
    assert abs(out[0].x - obj[0].x) < 1e-3
    assert abs(out[0].angle - obj[0].angle) < 1e-4


def test_polarity_match_rejects_inverted_contrast():
    reference = make_bracket()
    inverted = 250 - make_bracket()

    strict = vm.ShapeModel(reference, BRACKET_ROI)
    lenient = vm.ShapeModel(
        reference, BRACKET_ROI, vm.ShapeModelConfig(polarity="ignore_global")
    )
    matcher = vm.ShapeMatcher(vm.ShapeSearchConfig(min_score=0.6))

    assert matcher.find(inverted, strict) == []
    found = matcher.find(inverted, lenient)
    assert len(found) == 1 and found[0].score > 0.85


def test_config_validation_errors():
    img = make_step_image()

    with pytest.raises(ValueError):
        vm.detect_edges_u8(img, vm.EdgeConfig(smooth_kind="bad"))

    # A zero-extent ROI is invalid config, not an empty result.
    with pytest.raises(ValueError):
        vm.ShapeModel(img, (0.0, 0.0, 0.0, 10.0))

    # ...and so is an ROI outside the image.
    with pytest.raises(ValueError):
        vm.ShapeModel(img, (1000.0, 1000.0, 20.0, 20.0))

    model = vm.ShapeModel(make_bracket(), BRACKET_ROI)
    with pytest.raises(ValueError):
        vm.ShapeMatcher(vm.ShapeSearchConfig(min_score=1.5)).find(img, model)
    with pytest.raises(ValueError):
        vm.ShapeMatcher(vm.ShapeSearchConfig(refinement="magic")).find(img, model)
    with pytest.raises(ValueError):
        vm.ShapeModel(
            make_bracket(), BRACKET_ROI, vm.ShapeModelConfig(polarity="sideways")
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_shape_model_save_load_roundtrip(tmp_path):
    """A persisted model must match at the identical pose and score."""
    model = vm.ShapeModel(make_bracket(), BRACKET_ROI)

    path = str(tmp_path / "bracket_model.json")
    model.save(path)
    restored = vm.ShapeModel.load(path)
    assert restored.num_levels == model.num_levels
    assert restored.point_counts == model.point_counts

    scene = make_bracket(cx=115.0, cy=70.0)
    matcher = vm.ShapeMatcher()
    a = matcher.find(scene, model)
    b = matcher.find(scene, restored)
    assert len(a) == len(b) == 1
    assert a[0].score == b[0].score
    assert (a[0].x, a[0].y) == (b[0].x, b[0].y)

    # A tampered format_version must be refused.
    # Derive the needle from the module constant: hard-coding version 1 made
    # this assertion silently stop testing anything when the format was bumped.
    needle = f'"format_version":{vm.SHAPE_MODEL_FORMAT_VERSION}'
    original = open(path).read()
    assert needle in original, f"envelope shape changed: {needle} not found"
    doc = original.replace(needle, '"format_version":999', 1)
    assert doc != original, "version substitution did not fire"
    bad = str(tmp_path / "bad.json")
    open(bad, "w").write(doc)
    try:
        vm.ShapeModel.load(bad)
        assert False, "expected ValueError for unsupported format version"
    except ValueError:
        pass

