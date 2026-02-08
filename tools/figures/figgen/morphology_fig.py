from __future__ import annotations

import numpy as np

from .common import save_fig
from .config import RNG
from .plot import plt


def binary_dilate3x3(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = [
        p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ]
    return np.logical_or.reduce(neighbors)


def binary_erode3x3(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    p = np.pad(mask, 1, mode="constant", constant_values=False)
    neighbors = [
        p[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
        for dy in (-1, 0, 1)
        for dx in (-1, 0, 1)
    ]
    return np.logical_and.reduce(neighbors)


def make_figure() -> None:
    h, w = 160, 220
    yy, xx = np.mgrid[0:h, 0:w]

    mask = np.zeros((h, w), dtype=bool)
    mask[(xx >= 20) & (xx <= 90) & (yy >= 25) & (yy <= 120)] = True
    mask[((xx - 150) ** 2 + (yy - 85) ** 2) <= 35**2] = True
    mask[(xx >= 45) & (xx <= 170) & (yy >= 130) & (yy <= 150)] = True

    salt = RNG.random((h, w)) < 0.007
    holes = RNG.random((h, w)) < 0.006
    noisy = (mask | salt) & (~(mask & holes))

    opened = binary_dilate3x3(binary_erode3x3(noisy))
    closed = binary_erode3x3(binary_dilate3x3(noisy))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    views = [
        (noisy, "Input (noise + holes)"),
        (opened, "Open (3x3)"),
        (closed, "Close (3x3)"),
        (mask, "Reference clean mask"),
    ]

    for ax, (img, title) in zip(axes.ravel(), views):
        ax.imshow(img.astype(np.uint8), cmap="gray", vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")

    fig.suptitle("Binary morphology illustration", fontsize=12)
    save_fig(fig, "morphology_open_close.png")
