# Run: python examples/python/edge_detection.py
"""Detect edges on a synthetic vertical step-edge image."""

import numpy as np
import vision_metrology as vm


def main():
    # 64x64 image: left half 0, right half 200
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = 200

    # Object API
    det = vm.EdgeDetector(vm.EdgeConfig())
    edgels_obj = det.detect_u8(img)
    print(f"Object API: detected {len(edgels_obj)} edgels")

    # Free-function API
    edgels_fn = vm.detect_edges_u8(img, vm.EdgeConfig())
    print(f"Function API: detected {len(edgels_fn)} edgels")

    if edgels_fn:
        e = edgels_fn[0]
        print(f"First edgel: {e}")
        print(f"  position: ({e.x:.2f}, {e.y:.2f}), strength={e.strength:.2f}")


if __name__ == "__main__":
    main()
