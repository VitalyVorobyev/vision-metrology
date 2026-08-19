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
    assert hasattr(vm, "detect_edges")
    assert not hasattr(vm, "detect_edges_u8"), "the _u8 suffix is gone: detect dispatches on dtype"
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
    edgels_obj = det.detect(img)
    edgels_fn = vm.detect_edges(img, cfg)

    assert len(edgels_obj) > 0
    assert len(edgels_obj) == len(edgels_fn)

    e = edgels_obj[0]
    for attr in ("x", "y", "nx", "ny", "strength"):
        assert hasattr(e, attr)


def test_edge_detector_no_edges_on_blank():
    img = np.full((64, 64), 128, dtype=np.uint8)
    edgels = vm.detect_edges(img, vm.EdgeConfig())
    assert len(edgels) == 0


def test_edge_detector_dtype_dispatch_agrees_on_a_synthetic_edge():
    """u8, u16 and f32 views of the same edge must agree to the bit."""
    img8 = make_step_image()
    img16 = img8.astype(np.uint16)
    img32 = img8.astype(np.float32)

    results = [vm.EdgeDetector().detect(im) for im in (img8, img16, img32)]
    counts = [len(r) for r in results]
    assert counts[0] == counts[1] == counts[2] > 0
    assert results[0][0].x == results[1][0].x == results[2][0].x


def test_edge_detector_unsupported_dtype_names_the_supported_ones():
    bad = np.zeros((16, 16), dtype=np.int32)
    with pytest.raises(ValueError, match="uint8|uint16|float32"):
        vm.EdgeDetector().detect(bad)


def test_line_detector_object_and_function():
    h, w = 64, 64
    img = np.zeros((h, w), dtype=np.uint8)
    img[32:, :] = 255

    cfg = vm.LsdConfig()
    det = vm.LsdDetector(cfg)
    segs_obj = det.detect(img)
    segs_fn = vm.detect_line_segments(img, cfg)

    assert isinstance(segs_obj, list)
    assert isinstance(segs_fn, list)
    assert len(segs_obj) == len(segs_fn)

    if segs_fn:
        s = segs_fn[0]
        for attr in ("x1", "y1", "x2", "y2", "width", "nfa", "angle", "length"):
            assert hasattr(s, attr)


def test_line_detector_dtype_dispatch_agrees():
    h, w = 64, 64
    img8 = np.zeros((h, w), dtype=np.uint8)
    img8[32:, :] = 255
    img16 = img8.astype(np.uint16)
    img32 = img8.astype(np.float32)

    counts = [len(vm.LsdDetector().detect(im)) for im in (img8, img16, img32)]
    assert counts[0] == counts[1] == counts[2] > 0


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


def test_shape_matcher_dtype_dispatch_agrees_with_u8():
    """ShapeModel.build and ShapeMatcher.find both dispatch on dtype."""
    reference = make_bracket()
    scene = make_bracket(cx=115.0, cy=70.0)

    model8 = vm.ShapeModel(reference, BRACKET_ROI)
    model16 = vm.ShapeModel(reference.astype(np.uint16), BRACKET_ROI)
    model32 = vm.ShapeModel(reference.astype(np.float32), BRACKET_ROI)
    assert model8.point_counts == model16.point_counts == model32.point_counts

    matcher = vm.ShapeMatcher()
    m8 = matcher.find(scene, model8)
    m16 = matcher.find(scene.astype(np.uint16), model16)
    m32 = matcher.find(scene.astype(np.float32), model32)
    assert len(m8) == len(m16) == len(m32) == 1
    assert abs(m8[0].x - m16[0].x) < 1e-3
    assert abs(m8[0].x - m32[0].x) < 1e-3


def test_shape_model_unsupported_dtype_names_the_supported_ones():
    bad = np.zeros((16, 16), dtype=np.int32)
    with pytest.raises(ValueError, match="uint8|uint16|float32"):
        vm.ShapeModel(bad, BRACKET_ROI)


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
        vm.detect_edges(img, vm.EdgeConfig(smooth_kind="bad"))

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


def test_contrast_is_a_tagged_type_not_a_bare_float():
    raw = vm.Contrast.raw(12.0)
    frac = vm.Contrast.fraction_of_range(0.1)
    assert "raw" in repr(raw)
    assert "fraction_of_range" in repr(frac)
    # Both variants plug into a config without error.
    vm.ShapeModelConfig(min_contrast=raw)
    vm.ShapeSearchConfig(min_contrast=frac)


def test_shape_search_config_has_nested_tuning():
    tuning = vm.ShapeSearchTuning(greediness=0.3, max_candidates=64, last_level=1)
    cfg = vm.ShapeSearchConfig(min_score=0.6, tuning=tuning)
    assert abs(cfg.tuning.greediness - 0.3) < 1e-6
    assert cfg.tuning.max_candidates == 64
    assert cfg.tuning.last_level == 1
    # The default is independent and unaffected.
    assert vm.ShapeSearchConfig().tuning.greediness != 0.3


def test_shape_model_config_carries_an_edge_config():
    cfg = vm.ShapeModelConfig(edge=vm.EdgeConfig(smooth_kind="none"))
    assert cfg.edge.smooth_kind == "none"
    model = vm.ShapeModel(make_bracket(), BRACKET_ROI, cfg)
    assert model.num_levels >= 1


def test_shape_search_config_roi_restricts_matches():
    import math

    # Two brackets, far enough apart that a wide search finds both.
    w, h = 320, 240
    left = make_bracket(w, h, cx=90.0, cy=90.0)
    right_only = make_bracket(w, h, cx=230.0, cy=150.0)
    scene = np.maximum(left, right_only)

    model = vm.ShapeModel(make_bracket(), BRACKET_ROI)
    wide = vm.ShapeMatcher(vm.ShapeSearchConfig(min_score=0.6, max_matches=5))
    both = wide.find(scene, model)
    assert len(both) == 2

    # Restrict the ROI to a box around the left instance only.
    narrow = vm.ShapeMatcher(
        vm.ShapeSearchConfig(min_score=0.6, max_matches=5, roi=(40.0, 40.0, 100.0, 100.0))
    )
    left_only = narrow.find(scene, model)
    assert len(left_only) == 1
    assert abs(left_only[0].x - 90.0) < 15.0
    assert abs(left_only[0].y - 90.0) < 15.0

    # A rotation range that excludes the model's own (unrotated) angle finds nothing.
    off_angle = vm.ShapeMatcher(
        vm.ShapeSearchConfig(
            min_score=0.6, angle_range=(math.radians(90.0), math.radians(150.0))
        )
    )
    assert off_angle.find(scene, model) == []


def make_disc(w: int, h: int, cx: float, cy: float, r: float) -> np.ndarray:
    """Anti-aliased disc: the true edge sits at exactly `r`, every direction."""
    ys, xs = np.mgrid[0:h, 0:w]
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    cover = np.clip(r + 0.5 - d, 0.0, 1.0)
    return (20 + 180 * cover).astype(np.uint8)


def test_caliper_rect_finds_a_step_edge():
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 30:] = 200
    cal = vm.Caliper.rect((32.0, 32.0), 0.0, 20.0, 10.0)
    edges = cal.measure(img)
    assert len(edges) == 1
    assert abs(edges[0].x - 29.5) < 0.05
    assert edges[0].polarity == "rising"


def test_caliper_measure_raises_reject_reason_on_a_flat_field():
    flat = np.full((64, 64), 128, dtype=np.uint8)
    cal = vm.Caliper.rect((32.0, 32.0), 0.0, 20.0, 4.0)
    with pytest.raises(vm.MeasureRejected) as exc_info:
        cal.measure(flat)
    assert exc_info.value.args[0] == "no_edge"


def test_caliper_radial_and_measure_pairs():
    disc = make_disc(160, 160, 80.0, 80.0, 40.0)
    cal = vm.Caliper.radial((80.0, 80.0), 40.0, 0.0, 8.0, 3.0)
    edges = cal.measure(disc)
    assert len(edges) == 1
    assert abs(edges[0].x - 120.0) < 0.1  # centre.x + radius

    bar = np.zeros((96, 128), dtype=np.uint8)
    bar[:, 40:70] = 200
    pairs = vm.Caliper.rect((64.0, 48.0), 0.0, 50.0, 10.0).measure_pairs(bar)
    assert len(pairs) == 1
    assert abs(pairs[0].width - 30.0) < 0.05


def test_caliper_dtype_dispatch_agrees_on_a_synthetic_edge():
    """u8, u16 and f32 views of the same edge must agree to the bit."""
    img8 = np.zeros((64, 64), dtype=np.uint8)
    img8[:, 30:] = 200
    img16 = img8.astype(np.uint16)
    img32 = img8.astype(np.float32)

    xs = [
        vm.Caliper.rect((32.0, 32.0), 0.0, 20.0, 6.0).measure(im)[0].x
        for im in (img8, img16, img32)
    ]
    assert xs[0] == xs[1] == xs[2]


def test_caliper_unsupported_dtype_names_the_supported_ones():
    bad = np.zeros((16, 16), dtype=np.int32)
    with pytest.raises(ValueError, match="uint8|uint16|float32"):
        vm.Caliper.rect((8.0, 8.0), 0.0, 4.0, 2.0).measure(bad)


def test_fit_line_object_and_function():
    pts = np.array([[float(i), 2.0] for i in range(10)], dtype=np.float32)
    obj = vm.Fitter().fit_line(pts)
    fn = vm.fit_line(pts, vm.FitConfig())
    assert obj is not None and fn is not None
    assert abs(obj.dy) < 1e-4, "should be horizontal"
    assert obj.rms < 1e-4
    assert obj.n_used == 10


def test_metrology_model_measures_a_circle():
    """The measurement chain: synthetic disc -> radial calipers -> circle fit."""
    c = (80.0, 80.0)
    disc = make_disc(160, 160, c[0], c[1], 40.0)

    model = vm.MetrologyModel()
    model.add(vm.MetrologyObject(vm.MetrologyShape.circle((0.0, 0.0), 40.0)))
    assert model.num_objects == 1

    results = model.apply(disc, c[0], c[1])
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, vm.MetrologyResult)
    assert r.kind == "circle"
    assert abs(r.circle.r - 40.0) < 0.05
    assert r.rms < 0.05
    assert len(r.hits) == 32


def test_metrology_model_reports_a_failed_object_without_dropping_others():
    flat = np.full((64, 64), 128, dtype=np.uint8)
    model = vm.MetrologyModel()
    model.add(vm.MetrologyObject(vm.MetrologyShape.circle((0.0, 0.0), 20.0)))
    model.add(vm.MetrologyObject(vm.MetrologyShape.line((10.0, 10.0), (10.0, 50.0))))
    outcomes = model.apply(flat, 32.0, 32.0)
    assert len(outcomes) == 2
    assert all(isinstance(o, vm.MetrologyError) for o in outcomes)
    assert all(isinstance(o.message, str) and o.message for o in outcomes)


def test_contour_graph_build_and_smooth():
    disc = make_disc(160, 160, 80.0, 80.0, 40.0)
    g = vm.build_contour_graph(disc)
    assert g.num_edges >= 1
    assert g.num_junctions == 0

    polylines = g.polylines()
    assert len(polylines) == g.num_edges
    poly = polylines[0]
    assert poly.ndim == 2 and poly.shape[1] == 2
    assert poly.dtype == np.float32

    smoothed = vm.smooth_polyline(poly, 1.5)
    assert smoothed.shape == poly.shape


def test_morph_basics_round_trip_shapes():
    disc = make_disc(96, 96, 48.0, 48.0, 30.0)
    binary = (disc > 100).astype(np.uint8) * 255

    for fn in (vm.erode, vm.dilate, vm.open, vm.close):
        out = fn(binary, shape="disk", radius=2)
        assert out.shape == binary.shape
        assert out.dtype == np.uint8

    thinned = vm.thin(binary)
    assert thinned.shape == binary.shape

    dist = vm.chamfer_distance(binary)
    assert dist.shape == binary.shape
    assert dist.dtype == np.float32
    assert dist.max() > 0


def test_package_ships_py_typed_and_a_stub_matching_the_runtime_surface():
    """`py.typed` and `__init__.pyi` must be installed, and the stub's names
    must be a subset of what actually exists at runtime (no promising an API
    the extension doesn't have)."""
    import importlib.resources as res
    import re

    root = res.files("vision_metrology")
    assert (root / "py.typed").is_file()

    stub_text = (root / "__init__.pyi").read_text()
    class_names = re.findall(r"^class (\w+)", stub_text, re.MULTILINE)
    func_names = re.findall(r"^def (\w+)\(", stub_text, re.MULTILINE)

    assert "EdgeDetector" in class_names
    assert "MetrologyModel" in class_names
    assert "Caliper" in class_names
    assert "detect_edges" in func_names
    assert "fit_line" in func_names

    for name in class_names + func_names:
        assert hasattr(vm, name), f"{name} is declared in the stub but missing at runtime"


def test_shape_model_save_load_roundtrip(tmp_path):
    """A persisted model must match at the identical pose and score."""
    model = vm.ShapeModel(make_bracket(), BRACKET_ROI)

    path = str(tmp_path / "bracket.model")
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

    # The stored document is opaque and versioned. The only thing the format
    # promises a caller is that a foreign document is refused rather than
    # mis-read, so that is what this checks.
    bad = str(tmp_path / "bad.model")
    open(bad, "w").write("not a shape model at all")
    try:
        vm.ShapeModel.load(bad)
        assert False, "expected ValueError for an unreadable document"
    except ValueError:
        pass

