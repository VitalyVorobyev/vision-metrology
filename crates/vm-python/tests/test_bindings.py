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

    cfg = vm.ConicFitConfig(use_bookstein=False, ransac_iters=200, inlier_tol=1.5)
    obj_fit = vm.ConicFitter(cfg).fit_ellipse(pts)
    fn_fit = vm.fit_ellipse(pts, cfg)

    assert obj_fit is not None
    assert fn_fit is not None
    assert abs(obj_fit.cx - fn_fit.cx) < 1e-3
    assert abs(obj_fit.cy - fn_fit.cy) < 1e-3


def test_conic_fitter_too_few_points_returns_none():
    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    cfg = vm.ConicFitConfig(use_bookstein=False, ransac_iters=10, inlier_tol=1.0)
    result = vm.fit_ellipse(pts, cfg)
    assert result is None


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


def test_config_validation_errors():
    img = make_step_image()

    with pytest.raises(ValueError):
        vm.detect_edges_u8(img, vm.EdgeConfig(smooth_kind="bad"))

    with pytest.raises(ValueError):
        vm.match_rigid_model(
            np.array([[0, 0, 1, 0, 1]], dtype=np.float32),
            img,
            vm.EdgeConfig(),
            vm.RigidMatchConfig(min_score=1.5),
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
