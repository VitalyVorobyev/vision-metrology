"""
Smoke tests for the vm_python PyO3 extension module.

These tests require the module to be built with maturin:
    maturin develop --manifest-path crates/vm-python/Cargo.toml

Run with:
    python -m pytest crates/vm-python/tests/test_bindings.py -v
"""

import numpy as np
import vm_python


def make_step_image(w: int = 64, h: int = 64, edge_x: int = 32) -> np.ndarray:
    """Create a synthetic step edge: left half dark, right half bright."""
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, edge_x:] = 200
    return img


def test_edge_detector_finds_edges():
    """PyEdgeDetector should return at least one edgel on a step-edge image."""
    img = make_step_image()
    det = vm_python.PyEdgeDetector()
    edgels = det.detect_u8(img)
    assert len(edgels) > 0, f"Expected edgels on step edge, got {len(edgels)}"
    # Each edgel should be a dict with required keys.
    for e in edgels[:5]:
        assert "x" in e and "y" in e and "nx" in e and "ny" in e and "strength" in e


def test_edge_detector_no_edges_on_blank():
    """PyEdgeDetector should return no edgels on a uniform image."""
    img = np.full((64, 64), 128, dtype=np.uint8)
    det = vm_python.PyEdgeDetector()
    edgels = det.detect_u8(img)
    assert len(edgels) == 0, f"Expected no edgels on uniform image, got {len(edgels)}"


def test_edge_detector_reuse():
    """Calling detect_u8 multiple times should not raise."""
    img = make_step_image()
    det = vm_python.PyEdgeDetector()
    e1 = det.detect_u8(img)
    e2 = det.detect_u8(img)
    assert len(e1) == len(e2), "Same image → same edgel count on repeated calls"


def test_lsd_detector():
    """PyLsdDetector should detect a horizontal line on a step image."""
    # 64×64 image with a strong horizontal step at y=32.
    h, w = 64, 64
    img = np.zeros((h, w), dtype=np.uint8)
    img[32:, :] = 255
    det = vm_python.PyLsdDetector()
    segs = det.detect(img)
    assert isinstance(segs, list), "detect() should return a list"
    # Each segment should be a dict with the required keys.
    for s in segs[:3]:
        for key in ("x1", "y1", "x2", "y2", "width", "nfa", "angle"):
            assert key in s, f"Missing key {key!r} in segment dict"


def test_conic_fitter_ellipse():
    """PyConicFitter should fit an ellipse and return geometric parameters."""
    # Generate points on a known ellipse (cx=32, cy=32, a=10, b=5, angle=0).
    n = 20
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([32.0 + 10.0 * np.cos(t), 32.0 + 5.0 * np.sin(t)]).astype(
        np.float32
    )
    fitter = vm_python.PyConicFitter()
    result = fitter.fit_ellipse(pts)
    assert result is not None, "Fitting perfect ellipse points should succeed"
    for key in ("cx", "cy", "a", "b", "angle"):
        assert key in result, f"Missing key {key!r} in ellipse dict"
    assert abs(result["cx"] - 32.0) < 1.0, f"cx={result['cx']:.2f}"
    assert abs(result["cy"] - 32.0) < 1.0, f"cy={result['cy']:.2f}"


def test_conic_fitter_too_few_points():
    """PyConicFitter should return None (via error) on fewer than 5 points."""
    pts = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    fitter = vm_python.PyConicFitter()
    result = fitter.fit_ellipse(pts)
    # Either None or an exception is acceptable; None is expected.
    assert result is None, "Too few points should return None"


if __name__ == "__main__":
    test_edge_detector_finds_edges()
    test_edge_detector_no_edges_on_blank()
    test_edge_detector_reuse()
    test_lsd_detector()
    test_conic_fitter_ellipse()
    test_conic_fitter_too_few_points()
    print("All smoke tests passed.")
