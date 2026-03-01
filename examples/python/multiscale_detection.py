# Run: python examples/python/multiscale_detection.py
"""Multi-scale edge detection on a step-edge image."""

import numpy as np
import vision_metrology as vm


def main():
    img = np.zeros((256, 256), dtype=np.uint8)
    img[:, 128:] = 200

    for n_levels in [1, 2, 3]:
        cfg = vm.MultiScaleConfig(num_levels=n_levels)
        edgels = vm.detect_multiscale_edges_u8(img, cfg)
        print(f"num_levels={n_levels}: {len(edgels)} edgels")


if __name__ == "__main__":
    main()
