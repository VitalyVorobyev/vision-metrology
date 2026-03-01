# Run: python examples/python/segmentation.py
"""Otsu threshold -> binary -> CCL -> component stats on a bimodal image."""

import numpy as np
import vision_metrology as vm


def main():
    img = np.full((64, 64), 30, dtype=np.uint8)
    img[10:21, 10:21] = 200
    img[40:56, 40:56] = 200

    # Object API
    seg = vm.Segmenter()
    threshold_obj = seg.otsu_threshold(img)

    # Free-function API
    threshold = vm.otsu_threshold(img)
    print(f"Otsu threshold: {threshold} (object API gives {threshold_obj})")

    binary = vm.threshold_binary(img, threshold)
    print(f"Binary image: {binary.sum() // 255} foreground pixels")

    label_img, n_labels = vm.label_components(binary, connectivity=8)
    print(f"Found {n_labels} connected component(s)")

    stats = vm.component_stats(label_img, n_labels, min_area=10)
    for s in stats:
        print(f"  {s}")


if __name__ == "__main__":
    main()
