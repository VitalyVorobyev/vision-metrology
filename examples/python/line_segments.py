# Run: python examples/python/line_segments.py
"""Detect line segments in a synthetic image with 4 lines."""

import numpy as np
import vision_metrology as vm


def main():
    img = np.full((128, 128), 128, dtype=np.uint8)
    img[20, 10:118] = 0
    img[60, 10:118] = 0
    img[10:118, 30] = 0
    img[10:118, 90] = 0

    det = vm.LsdDetector(vm.LsdConfig())
    segs = det.detect_u8(img)
    print(f"Detected {len(segs)} line segments")
    for s in segs:
        print(f"  {s}")


if __name__ == "__main__":
    main()
