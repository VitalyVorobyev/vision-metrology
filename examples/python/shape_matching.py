# Run: python examples/python/shape_matching.py
"""Build a shape model from a reference ROI and locate it, rotated, in a scene."""

import math

import numpy as np
import vision_metrology as vm

W, H = 320, 256


def render_bracket(cx, cy, angle):
    """Anti-aliased L-bracket, so the synthetic edges have a real edge profile."""
    ys, xs = np.mgrid[0:H, 0:W]
    dx, dy = xs - cx, ys - cy
    cs, sn = math.cos(angle), math.sin(angle)
    mx, my = cs * dx + sn * dy, -sn * dx + cs * dy

    def sdf_box(px, py, hx, hy):
        ax, ay = np.abs(px) - hx, np.abs(py) - hy
        outside = np.sqrt(np.maximum(ax, 0) ** 2 + np.maximum(ay, 0) ** 2)
        return outside + np.minimum(np.maximum(ax, ay), 0)

    sdf = np.minimum(sdf_box(mx, my + 30, 40, 10), sdf_box(mx + 30, my, 10, 40))
    t = np.clip((1.0 - sdf) / 2.0, 0.0, 1.0)
    t = t * t * (3.0 - 2.0 * t)
    return (40 + 170 * t).astype(np.uint8)


def main():
    reference = render_bracket(160, 128, 0.0)
    roi = (104, 72, 112, 112)  # x, y, width, height

    model = vm.ShapeModel(reference, roi, vm.ShapeModelConfig(max_points=400))
    print(model)

    # Same object, moved and rotated by 40 degrees.
    truth_angle = math.radians(40.0)
    scene = render_bracket(200, 150, truth_angle)

    matcher = vm.ShapeMatcher(
        vm.ShapeSearchConfig(min_score=0.6, refinement="least_squares")
    )
    matches = matcher.find(scene, model)

    if not matches:
        print("no match found")
        return

    m = matches[0]
    print(f"score {m.score:.3f}  at ({m.x:.2f}, {m.y:.2f})  "
          f"angle {math.degrees(m.angle):.2f} deg  scale {m.scale:.4f}")

    # `matrix` maps reference-image coordinates into the scene, origin included.
    mat = np.array(m.matrix(model.origin))
    probe = np.array([160.0, 128.0, 1.0])  # the shape's own centre
    mapped = mat @ probe
    print(f"reference (160, 128) maps to ({mapped[0]:.2f}, {mapped[1]:.2f}); "
          f"the object was rendered at (200, 150)")

    assert abs(math.degrees(m.angle) - 40.0) < 1.0
    assert abs(mapped[0] - 200.0) < 1.5 and abs(mapped[1] - 150.0) < 1.5
    print("All assertions passed.")


if __name__ == "__main__":
    main()
