from __future__ import annotations

import numpy as np

from .common import binomial_blur_1d, conv1d_clamp, dog_kernel_1d, save_fig
from .laser_fig import detect_best_pair
from .plot import plt


def make_figure() -> None:
    n = 96
    x = np.arange(n, dtype=np.float32)
    x_l, x_r = 20.3, 35.7

    ideal = np.where((x >= x_l) & (x <= x_r), 1.0, 0.0).astype(np.float32)
    signal = 255.0 * ideal
    signal = np.minimum(signal, 255.0)
    signal = binomial_blur_1d(signal, passes=3)

    _, dg = dog_kernel_1d(sigma=1.2)
    response = conv1d_clamp(signal, dg)

    pair = detect_best_pair(
        response,
        predicted_center=None,
        min_width=3.0,
        max_width=30.0,
        bright_on_dark=True,
    )

    left_x = right_x = center_x = None
    if pair is not None:
        left_x, right_x, _ = pair
        center_x = 0.5 * (left_x + right_x)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax0.plot(x, signal, color="#2d2d2d", lw=2, label="signal")
    ax0.set_ylabel("intensity")
    ax0.set_title("1D stripe signal with DoG edge detection")
    ax0.grid(alpha=0.25)

    ax1.plot(x, response, color="#1f77b4", lw=2, label="DoG response")
    ax1.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax1.set_ylabel("response")
    ax1.set_xlabel("x (pixel-center index)")
    ax1.grid(alpha=0.25)

    if left_x is not None and right_x is not None and center_x is not None:
        for ax in (ax0, ax1):
            ax.axvline(left_x, color="#2ca02c", ls="--", lw=1.6)
            ax.axvline(right_x, color="#d62728", ls="--", lw=1.6)
            ax.axvline(center_x, color="#9467bd", ls=":", lw=1.8)
        ax0.legend(["signal", "left edge", "right edge", "center"], loc="upper right")

    save_fig(fig, "edge1d_dog_stripe.png")
