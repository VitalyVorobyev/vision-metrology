# Run: python examples/python/edge_matching.py
"""Build a rectangle model and match it in a shifted scene image."""

import numpy as np
import vision_metrology as vm


def main():
    # Model: 30x15 rectangle ring of edgels at origin
    model_pts = []
    for x in np.linspace(0, 30, 30):
        model_pts.append([x, 0.0, 0.0, -1.0, 1.0])
    for x in np.linspace(0, 30, 30):
        model_pts.append([x, 15.0, 0.0, 1.0, 1.0])
    for y in np.linspace(0, 15, 15):
        model_pts.append([0.0, y, -1.0, 0.0, 1.0])
    for y in np.linspace(0, 15, 15):
        model_pts.append([30.0, y, 1.0, 0.0, 1.0])

    model_edgels = np.array(model_pts, dtype=np.float32)

    # Scene: same rectangle translated by (80, 60)
    scene_img = np.zeros((150, 200), dtype=np.uint8)
    ox, oy, w, h = 80, 60, 30, 15
    scene_img[oy, ox : ox + w + 1] = 200
    scene_img[oy + h, ox : ox + w + 1] = 200
    scene_img[oy : oy + h + 1, ox] = 200
    scene_img[oy : oy + h + 1, ox + w] = 200

    edge_cfg = vm.EdgeConfig()
    match_cfg = vm.RigidMatchConfig()

    # Object API
    matcher = vm.RigidMatcher(edge_cfg, match_cfg)
    result_obj = matcher.match_model(model_edgels, scene_img)

    # Free-function API
    result = vm.match_rigid_model(model_edgels, scene_img, edge_cfg, match_cfg)

    if result is not None:
        print(f"Match found: {result}")
        print(f"  Translation: ({result.tx:.1f}, {result.ty:.1f})")
        print(f"  Score: {result.score:.3f}")
    elif result_obj is not None:
        print(f"Object API match found: {result_obj}")
    else:
        print("No match found (acceptable for this simple synthetic example).")


if __name__ == "__main__":
    main()
