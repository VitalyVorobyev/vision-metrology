# Run: python examples/python/measure_circles.py
"""Full Python pipeline: detect circles in a synthetic image and measure radii."""

import numpy as np
import vision_metrology as vm

CIRCLES = [
    (128.0, 128.0, 40.0),
    (320.0, 160.0, 60.0),
    (220.0, 360.0, 80.0),
]


def generate_image(width: int, height: int) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.float32)
    ys, xs = np.mgrid[0:height, 0:width]
    for cx, cy, r in CIRCLES:
        dist = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        t = np.clip((r + 2.0 - dist) / 4.0, 0.0, 1.0)
        smooth = t * t * (3.0 - 2.0 * t)
        img = np.maximum(img, smooth)
    return (img * 255).round().astype(np.uint8)


def main():
    w, h = 512, 512
    print(f"Generating {w}x{h} synthetic image with 3 circles (r=40, 60, 80)...")
    img = generate_image(w, h)

    print("Running EdgeDetector...")
    edgels = vm.detect_edges_u8(img, vm.EdgeConfig())
    print(f"  Detected {len(edgels)} edgels")

    if not edgels:
        print("No edgels detected!")
        return

    pts = np.array([[e.x, e.y] for e in edgels], dtype=np.float32)

    fitter_cfg = vm.ConicFitConfig(use_bookstein=False, ransac_iters=200, inlier_tol=1.5)
    results = []
    for cx, cy, r in CIRCLES:
        dist = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        mask = (dist > r * 0.5) & (dist < r * 1.5)
        cluster = pts[mask]
        if len(cluster) < 10:
            print(f"  Circle r={r}: not enough edgels ({len(cluster)})")
            continue

        result = vm.fit_ellipse(cluster, fitter_cfg)
        if result is not None:
            r_fit = (result.a + result.b) / 2.0
            print(
                f"  Circle r={r:.0f}: measured r={r_fit:.2f}, center=({result.cx:.1f}, {result.cy:.1f})"
            )
            results.append((cx, cy, r, result))
        else:
            print(f"  Circle r={r}: fit failed")

    print(f"\nFitted {len(results)} of {len(CIRCLES)} circles")


if __name__ == "__main__":
    main()
