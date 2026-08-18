# Run: python examples/python/fit_ellipse.py
"""Fit an ellipse to exact and noisy point sets."""

import numpy as np
import vision_metrology as vm


def main():
    cx, cy, a, b = 50.0, 40.0, 20.0, 10.0
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False).astype(np.float32)
    pts_exact = np.column_stack([
        cx + a * np.cos(t),
        cy + b * np.sin(t),
    ]).astype(np.float32)

    cfg = vm.ConicFitConfig(use_bookstein=False, ransac_iters=200, inlier_tol=1.5)

    # Object API
    fitter = vm.ConicFitter(cfg)
    result_obj = fitter.fit_ellipse(pts_exact)

    # Free-function API
    result = vm.fit_ellipse(pts_exact, cfg)

    if result is not None:
        print(f"Exact fit: {result}")
        print(f"  center error: ({abs(result.cx - cx):.3f}, {abs(result.cy - cy):.3f})")
    elif result_obj is not None:
        print(f"Object API exact fit: {result_obj}")
    else:
        print("Exact fit failed")

    rng = np.random.default_rng(42)
    pts_noisy = (pts_exact + rng.normal(0, 0.5, pts_exact.shape)).astype(np.float32)
    noisy = vm.fit_ellipse(pts_noisy, cfg)
    if noisy is not None:
        print(f"Noisy fit: {noisy}")
    else:
        print("Noisy fit failed")


if __name__ == "__main__":
    main()
